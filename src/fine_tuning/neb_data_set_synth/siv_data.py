"""Build a curated NEB training dataset from VASP OUTCAR files.

This module is the high-level orchestrator for the NEB dataset synthesis
pipeline. It loads a YAML config, builds a per-frame catalogue from one or more
NEB sources, applies a rule-based sampler, splits the selected rows into
train/validation/test sets, and writes family-ready `extxyz` outputs plus
bookkeeping files.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shlex
import sys
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import yaml

from .outcar_extxyz import find_outcar, iter_force_tables, maybe_subtract_d3, parse_header, write_frame
from .siv_shelf import RULES, allocate_quotas


# if one changes the structure of the caches, or how they are processed, then changing the scheme_version automatially
# will disqualify old versions, rather than stale cached data throwing a spanner in the works. 
scheme_version = 1


def load_rules_yaml(path: Path) -> dict[str, object]:
    """Load and validate the rule configuration YAML.

    Besides basic checks, this also normalizes some derived metadata that
    later stages rely on, such as `source_order`.

    That is, this, this loads the yaml, but also checks that the necessary characteristics of the 
    config are present and valid. 
    firsty, the metadata that is required of the yaml:
        Outputs : where to write the output files, including a prefix for the file names.
        Sampling : how many total samples to select.
        D3 : whether to apply D3 subtraction at write time, and if so, which method and damping to use.
        Sources : a list of sources, each with a name, weight, neb_root, *and* images to use from it 
        Rule : which selection rule to apply to the catalogue of data (obtained from the sources)
         NOTE :  the available Rules to choose from have to be defined in siv_shelf.py 
        Split : how to split the selected data into train/val/test sets, including the random seed to use for shuffling.
         NOTE : this random seed is created by `f"{base_seed}:{source_name}"`, with base_seed being the number from `seed : <base_seed>`
         in the yaml. This means that the shuffling is deterministic but different across sources. This is done with pythons pseudo-random number generator
         rng.random.Random(f"{base_seed}:{source_name}")
         rnd.shuffle(data)

    Args:
        path: Path to the YAML configuration file.

    Returns:
        dict[str, object]: Parsed and validated configuration.

    Raises:
        ValueError: If the configuration is missing required keys or contains
            invalid values.
    """
    rules_path = path.expanduser()
    if not rules_path.is_absolute():
        rules_path = (Path.cwd() / rules_path).resolve()
    base_dir = rules_path.parent

    # load the cofig:
    with rules_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # The yaml checks:

    # These sections define the full pipeline: where to write, what to sample,
    # where to read from, how to split, and which rule to apply.
    required = ["outputs", "sampling", "d3", "sources", "rule", "split"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required top-level section: {key}")

    total_count = int(config["sampling"]["total_count"])
    if total_count <= 0:
        raise ValueError("sampling.total_count must be positive")

    sources = list(config["sources"])
    if not sources:
        raise ValueError("At least one source is required")

    names: set[str] = set()
    for idx, source in enumerate(sources):
        name = str(source["name"])
        if name in names:
            raise ValueError(f"Duplicate source name: {name}")
        names.add(name)
        if float(source["weight"]) <= 0.0:
            raise ValueError(f"Source weight must be positive: {name}")
        if not source.get("images"):
            raise ValueError(f"Source must declare images: {name}")
        # Preserve YAML order so later sorting stays deterministic.
        source["source_order"] = idx

    split = config["split"]
    total_pct = float(split["train_pct"]) + float(split["val_pct"]) + float(split["test_pct"])
    if not math.isclose(total_pct, 100.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split percentages must sum to 100")

    rule_name = str(config["rule"]["name"])
    if rule_name not in RULES:
        raise ValueError(f"Unknown rule: {rule_name}")

    outputs = config["outputs"]
    outputs["out_dir"] = str(_resolve_relative_path(outputs["out_dir"], base_dir))
    for source in config["sources"]:
        source["neb_root"] = str(_resolve_relative_path(source["neb_root"], base_dir))

    return config


def _resolve_relative_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def cache_path_for(cache_dir: Path, source_name: str) -> Path:
    """Return the JSON cache path (that is safe for file system (eg: no spaces)) for one source.
    That is, given a source, it returns what JSON file should hold the cached information. 

    Args:
        cache_dir: Directory where cache files are stored.
        source_name: Human-readable source name from the YAML config.

    Returns:
        Path: Sanitized cache file path for the source.
    """

    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source_name)
    return cache_dir / f"{safe}.json"


def source_file_stats(source_spec: dict[str, object]) -> list[dict[str, object]]:
    """Collect file metadata used to check the source cache. This is to check if the 
    raw input files for this source have changed since the last time the cache was built. 
    The metrics by which are used for this comparison are: 
        image: which NEB image this file belongs to
        path: exact resolved file path
        mtime_ns: last modified time
        size: file size
    this is then compared against what was stored in the cache in load_cached_catalogue, 
    and if there is any mismatch, the cache is ignored, otherwise the cache is used. 

    Args:
        source_spec: YAML source entry.

    Returns:
        list[dict[str, object]]: Per-image file fingerprints.
    """

    neb_root = Path(source_spec["neb_root"])
    stats: list[dict[str, object]] = []
    for image in source_spec["images"]:
        image_name = str(image)
        outcar = find_outcar(neb_root / image_name)
        stat = outcar.stat()
        stats.append(
            {
                "image": image_name,
                "path": str(outcar.resolve()),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    return stats


def load_cached_catalogue(cache_path: Path, source_spec: dict[str, object], file_stats: list[dict[str, object]]) -> list[dict[str, object]] | None:
    """Load a previously built source catalogue if it is still valid.

    Cached rows intentionally omit large arrays such as positions and forces to
    keep the JSON compact; those are reloaded later only for selected frames.

    Args:
        cache_path: Path to the source cache file.
        source_spec: YAML source entry.
        file_stats: Current input file fingerprints.

    Returns:
        list[dict[str, object]] | None: Cached catalogue rows, or `None` if the
        cache is missing or stale.
    """

    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    expected = {
        "scheme_version": scheme_version,
        "name": str(source_spec["name"]),
        "neb_root": str(Path(source_spec["neb_root"]).resolve()),
        "images": [str(image) for image in source_spec["images"]],
    }
    if payload.get("source") != expected:
        return None
    if payload.get("files") != file_stats: # the files block saved inside the cache corresponds to the cached file_stats  
        return None

    rows = list(payload["rows"])
    for row in rows: # set up the row dicts, to be filled later. 
        row["positions"] = None
        row["forces"] = None
    return rows


def save_cached_catalogue(cache_path: Path, source_spec: dict[str, object], file_stats: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    """ Save a source catalogue to a JSON cache. keeps only metadata, not all of the positions and forces. 
    
    """

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scheme_version": scheme_version,
        "source": {
            "scheme_version": scheme_version,
            "name": str(source_spec["name"]),
            "neb_root": str(Path(source_spec["neb_root"]).resolve()),
            "images": [str(image) for image in source_spec["images"]],
        },
        "files": file_stats,
        "rows": [
            {
                key: value
                for key, value in row.items()
                # Positions and forces dominate file size, so keep only the
                # lightweight metadata in the cache. 
                if key not in {"positions", "forces"}
            }
            for row in rows
        ],
    }
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def build_catalogue(source_spec: dict[str, object], cache_dir: Path) -> list[dict[str, object]]:
    """Build or load the per-frame catalogue for one source. This is what gives the full data set available from our 
    NEB frames. 

    Each catalogue row is one frame from one NEB image from one source. it corresponds to one force table from one OUTCAR and
    carries the metadata needed for later sampling and output writing. this metadata includes: 

        source_name : The human-readable name of the NEB source this frame came from.
        source_order : The original order of the source in the YAML file, used for stable sorting.
        source_weight : The relative sampling weight assigned to this source.
        neb_root : The root directory containing the NEB image folders for this source.
        outcar_path : The resolved filesystem path to the specific OUTCAR file for this frame’s image.
        image : The NEB image name, such as i, 00, 04, or f.
        image_order : The position of this image in the source’s configured image list.
        frame_ordinal : The zero-based index of this force table within the image’s OUTCAR.
        ionic_step : The ionic step number reported in the OUTCAR for this frame.
        frame_energy : The DFT total energy for this specific frame.
        max_atom_force : The maximum force magnitude on any atom in this frame.
        positions : The atomic positions for this frame.
        forces : The atomic force vectors for this frame.
        symbols : The ordered list of atomic species for the structure.
        lattice : The 3x3 lattice vectors for the structure.
        final_image_energy : The energy of the last frame in the same image, used as an image-level score.
        image_percentile : The normalized percentile rank of the image’s final energy within the source.
        image_bin_weight : The sampling weight assigned to that energy percentile bin.
        image_selected_quota : How many frames the rule allocated to this image.

    Args:
        source_spec: YAML source entry.
        cache_dir: Cache directory for source catalogues.

    Returns:
        list[dict[str, object]]: Catalogue rows for the source.

    Raises:
        ValueError: If an image has no force tables.
    """

    # try load a valid cahce

    file_stats = source_file_stats(source_spec) 
    cache_path = cache_path_for(cache_dir, str(source_spec["name"]))
    cached = load_cached_catalogue(cache_path, source_spec, file_stats)
    if cached is not None:
        return cached

    neb_root = Path(source_spec["neb_root"])
    catalogue: list[dict[str, object]] = []

    for image_order, image in enumerate(source_spec["images"]):
        image_name = str(image)
        outcar = find_outcar(neb_root / image_name)
        symbols, lattice = parse_header(outcar)
        image_rows: list[dict[str, object]] = []

        for frame_ordinal, (ionic_step, energy, positions, forces) in enumerate(iter_force_tables(outcar, len(symbols))):
            # Use the largest per-atom force magnitude as a simple difficulty /
            # non-equilibrium score for the frame.
            max_atom_force = max(math.sqrt(fx * fx + fy * fy + fz * fz) for fx, fy, fz in forces)
            image_rows.append(
                {
                    # Source-level metadata.
                    "source_name": str(source_spec["name"]),
                    "source_order": int(source_spec["source_order"]),
                    "source_weight": float(source_spec["weight"]),
                    "neb_root": str(neb_root.resolve()),
                    "outcar_path": str(outcar.resolve()),
                    # Image-level metadata.
                    "image": image_name,
                    "image_order": image_order,
                    # Frame-level metadata.
                    "frame_ordinal": frame_ordinal,
                    "ionic_step": ionic_step,
                    "frame_energy": float(energy),
                    "max_atom_force": float(max_atom_force),
                    # Heavy data needed only if the frame is eventually written.
                    "positions": positions,
                    "forces": forces,
                    "symbols": symbols,
                    "lattice": lattice,
                    # Filled in here or later by the rule pipeline.
                    "final_image_energy": None,
                    "image_percentile": None,
                    "image_bin_weight": None,
                    "image_selected_quota": None,
                }
            )

        if not image_rows:
            raise ValueError(f"No force tables found in {outcar}")

        # The sampling rule uses the final relaxed image energy as an image-level
        # signal, so stamp it onto every row from the image.
        final_energy = float(image_rows[-1]["frame_energy"])
        for row in image_rows:
            row["final_image_energy"] = final_energy
        catalogue.extend(image_rows)

    save_cached_catalogue(cache_path, source_spec, file_stats, catalogue)
    return catalogue


def allocate_source_counts(total_count: int, sources: list[dict[str, object]]) -> dict[str, int]:
    """Allocate the sample numbers across sources.

    Args:
        total_count: Total number of rows requested by the config.
        sources: Source specs, including `available_count`, (number of frames in the built catalogue for that source)

    Returns:
        dict[str, int]: Requested number of samples per source.
    """

    weights = {str(source["name"]): float(source["weight"]) for source in sources}
    capacities = {str(source["name"]): int(source["available_count"]) for source in sources}
    ordered = [str(source["name"]) for source in sources]
    return allocate_quotas(ordered, weights, total_count, capacities)


def select_curated_pool(catalogue_by_source: dict[str, list[dict[str, object]]], config: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Apply source-level quotas and the configured selection rule determined by siv_rules.yml.

    Args:
        catalogue_by_source: Full catalogue rows keyed by source name.
        config: Validated pipeline configuration.

    Returns:
        tuple[list[dict[str, object]], dict[str, int]]: The selected rows and
        the source allocation counts that produced them.
    """

    sources = [dict(source) for source in config["sources"]]
    for source in sources:
        source["available_count"] = len(catalogue_by_source[str(source["name"])])

    source_allocations = allocate_source_counts(int(config["sampling"]["total_count"]), sources)
    rule = RULES[str(config["rule"]["name"])]

    selected_rows: list[dict[str, object]] = []
    for source in sources: # apply selection rule per source,
        source_name = str(source["name"])
        source["allocated_count"] = source_allocations[source_name] # how many rows the source is allowed to contribute
        
        # The rule decides which rows survive inside this source.
        # I.e., obtain the 'useful' data
        selected = rule(catalogue_by_source[source_name], source, config["rule"])
        
        # Record how many rows that source was supposed to contribute before the rule picked actual frames. 
        # allows one to distinguish between: how many rows the source was supposed to contribute, vs 
        # how many rows the source actually contributed after the rule may have filtered some out.
        for row in selected: #
            row["source_allocated_count"] = source_allocations[source_name]
        
        selected_rows.extend(selected)

    # Put rows back into a stable source/image/frame ordering after the rule may
    # have rearranged them internally for ranking purposes.
    selected_rows.sort(
        key=lambda row: (
            int(row["source_order"]),
            int(row["image_order"]),
            int(row["frame_ordinal"]),
        )
    )
    return selected_rows, source_allocations


def split_pool(selected_rows: list[dict[str, object]], split_spec: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Split the selected rows into train, validation, and test sets.

    The split is performed independently within each source so the source mix is
    preserved more faithfully than with one global shuffle.

    Args:
        selected_rows: Rows kept after rule-based selection.
        split_spec: Split configuration from the YAML file.

    Returns:
        dict[str, list[dict[str, object]]]: Rows grouped by split name.
    """

    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected_rows:
        by_source[str(row["source_name"])].append(row)

    split_rows = {"train": [], "val": [], "test": []}
    split_weights = {
        "train": float(split_spec["train_pct"]),
        "val": float(split_spec["val_pct"]),
        "test": float(split_spec["test_pct"]),
    }
    base_seed = int(split_spec["seed"])

    for source_name in sorted(by_source, key=lambda name: by_source[name][0]["source_order"]):
        rows = list(by_source[source_name])
        # Seed per source so reruns are deterministic but sources are shuffled
        # independently.
        rng = random.Random(f"{base_seed}:{source_name}")
        rng.shuffle(rows)
        counts = allocate_quotas(["train", "val", "test"], split_weights, len(rows))

        start = 0
        for split_name in ["train", "val", "test"]:
            end = start + counts[split_name]
            for row in rows[start:end]:
                row["assigned_split"] = split_name
                split_rows[split_name].append(row)
            start = end

    return split_rows


def hydrate_selected_rows(selected_rows: list[dict[str, object]]) -> None:
    """Reload positions and forces for selected rows missing heavy data.

    This is primarily needed when the catalogue was loaded from cache, because
    the cache intentionally stores only metadata.

    Args:
        selected_rows: Rows that will be written to output files.

    Raises:
        ValueError: If a requested frame cannot be reloaded from its OUTCAR.
    """

    needed: dict[str, set[int]] = defaultdict(set)
    for row in selected_rows:
        if row["positions"] is None or row["forces"] is None:
            needed[str(row["outcar_path"])].add(int(row["frame_ordinal"]))

    hydrated: dict[tuple[str, int], tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[str], list[list[float]]]] = {}
    for outcar_path, ordinals in needed.items():
        outcar = Path(outcar_path)
        symbols, lattice = parse_header(outcar)
        for frame_ordinal, (_, _, positions, forces) in enumerate(iter_force_tables(outcar, len(symbols))):
            if frame_ordinal in ordinals:
                # Key by OUTCAR path plus ordinal because different images may
                # reuse the same ordinal values.
                hydrated[(outcar_path, frame_ordinal)] = (positions, forces, symbols, lattice)

    for row in selected_rows:
        if row["positions"] is not None and row["forces"] is not None:
            continue
        key = (str(row["outcar_path"]), int(row["frame_ordinal"]))
        if key not in hydrated:
            raise ValueError(f"Could not hydrate selected frame: {key}")
        positions, forces, symbols, lattice = hydrated[key]
        row["positions"] = positions
        row["forces"] = forces
        row["symbols"] = symbols
        row["lattice"] = lattice


def write_manifest(manifest_path: Path, selected_rows: list[dict[str, object]]) -> None:
    """Write a CSV manifest describing the selected frames.

    Args:
        manifest_path: Output CSV path.
        selected_rows: Selected rows to record.
    """

    fieldnames = [
        "source_name",
        "neb_root",
        "source_weight",
        "source_allocated_count",
        "image",
        "frame_ordinal",
        "ionic_step",
        "frame_energy",
        "final_image_energy",
        "image_percentile",
        "image_bin_weight",
        "image_selected_quota",
        "max_atom_force",
        "assigned_split",
        "outcar_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_summary(summary_path: Path, config: dict[str, object], selected_rows: list[dict[str, object]], split_rows: dict[str, list[dict[str, object]]], source_allocations: dict[str, int]) -> None:
    """Write a human-readable text summary of the sampling run.

    Args:
        summary_path: Output summary path.
        config: Validated pipeline configuration.
        selected_rows: All selected rows across splits.
        split_rows: Rows grouped by split.
        source_allocations: Source quotas computed before rule selection.
    """

    all_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for row in selected_rows:
        source = str(row["source_name"])
        all_counts[source] += 1
        split_counts[source][str(row["assigned_split"])] += 1

    # Keep the summary easy to inspect in a text editor or job log.
    lines = [
        f"prefix: {config['outputs']['prefix']}",
        f"requested_total_count: {int(config['sampling']['total_count'])}",
        f"selected_total_count: {len(selected_rows)}",
        f"rule: {config['rule']['name']}",
        f"d3_remove: {bool(config['d3']['remove'])}",
        f"train_count: {len(split_rows['train'])}",
        f"val_count: {len(split_rows['val'])}",
        f"test_count: {len(split_rows['test'])}",
        "",
        "source_counts:",
    ]

    for source in config["sources"]:
        name = str(source["name"])
        lines.extend(
            [
                f"  - name: {name}",
                f"    requested_weight: {float(source['weight'])}",
                f"    allocated_count: {source_allocations.get(name, 0)}",
                f"    selected_count: {all_counts.get(name, 0)}",
                f"    train_count: {split_counts[name]['train']}",
                f"    val_count: {split_counts[name]['val']}",
                f"    test_count: {split_counts[name]['test']}",
            ]
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    selected_rows: list[dict[str, object]],
    split_rows: dict[str, list[dict[str, object]]],
    config: dict[str, object],
    source_allocations: dict[str, int],
) -> None:
    """Write `extxyz` datasets and metadata files for the selected rows.

    Args:
        selected_rows: Rows chosen by the sampling rule.
        split_rows: Rows grouped by split name.
        config: Validated pipeline configuration.
        source_allocations: Source quotas computed before selection.
    """

    # Ensure heavy data is present before we start writing frames.
    hydrate_selected_rows(selected_rows)

    out_dir = Path(config["outputs"]["out_dir"])
    prefix = str(config["outputs"]["prefix"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / f"{prefix}_all.extxyz"
    train_path = out_dir / f"{prefix}_train.extxyz"
    val_path = out_dir / f"{prefix}_val.extxyz"
    test_path = out_dir / f"{prefix}_test.extxyz"
    manifest_path = out_dir / f"{prefix}_manifest.csv"
    summary_path = out_dir / f"{prefix}_summary.txt"
    resolved_path = out_dir / f"{prefix}_resolved.yml"
    commands_path = out_dir / f"{prefix}_commands.txt"

    with ExitStack() as stack:
        handles = {
            "all": stack.enter_context(all_path.open("w", encoding="utf-8")),
            "train": stack.enter_context(train_path.open("w", encoding="utf-8")),
            "val": stack.enter_context(val_path.open("w", encoding="utf-8")),
            "test": stack.enter_context(test_path.open("w", encoding="utf-8")),
        }

        for row in selected_rows:
            # Apply optional D3 subtraction at write time so the cached catalogue
            # preserves the raw OUTCAR values.
            energy, forces = maybe_subtract_d3(
                bool(config["d3"]["remove"]),
                list(row["symbols"]),
                list(row["lattice"]),
                list(row["positions"]),
                list(row["forces"]),
                float(row["frame_energy"]),
                d3_method=str(config["d3"]["method"]),
                d3_damping=str(config["d3"]["damping"]),
                d3_params_tweaks=None,
                d3_cache_api=True,
            )

            write_frame(
                handles["all"],
                list(row["symbols"]),
                list(row["lattice"]),
                list(row["positions"]),
                forces,
                energy,
                str(row["image"]),
                int(row["ionic_step"]),
            )
            write_frame(
                handles[str(row["assigned_split"])],
                list(row["symbols"]),
                list(row["lattice"]),
                list(row["positions"]),
                forces,
                energy,
                str(row["image"]),
                int(row["ionic_step"]),
            )

    # Write the lightweight bookkeeping files after the larger frame data.
    write_manifest(manifest_path, selected_rows)
    write_summary(summary_path, config, selected_rows, split_rows, source_allocations)
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    commands_path.write_text(
        "\n".join(
            [
                f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                f"cwd: {Path.cwd()}",
                "command: " + " ".join(shlex.quote(arg) for arg in sys.argv),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

def run_neb_curation(rules_path: Path) -> dict[str, object]:
    """Run the full NEB dataset synthesis pipeline for one rules file."""

    config = load_rules_yaml(rules_path)

    cache_dir = Path(config["outputs"]["out_dir"]) / ".cache"
    # Build each source catalogue independently so caching and failures stay
    # localized to a source.
    catalogue_by_source = {
        str(source["name"]): build_catalogue(source, cache_dir)
        for source in config["sources"]
    }
    # build full catalogue of all available NEB frames. 
    selected_rows, source_allocations = select_curated_pool(catalogue_by_source, config)
    # select from that a smaller subset of data. 
    split_rows = split_pool(selected_rows, config["split"]) # split into train/val/test.
    write_outputs(selected_rows, split_rows, config, source_allocations) # write the output files.
    return config


if __name__ == "__main__":
    raise SystemExit(
        "Use `mlip-ft <family> --curate-neb --inputs <path/to/siv_rules.yml>` instead."
    )
