from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_common import (
    analysis_root,
    candidate_manifest_path,
    case_manifest_path,
    clear_analysis,
    copy_structure,
    discover_reference_dirs,
    ensure_analysis_dirs,
    load_reference_metadata,
    safe_label,
    write_csv_rows,
)


SNB_TUPLE_MANIFEST = Path("/home/rnpla/projects/mlip_phonons/assets/SNB_data/manifest/tuple_manifest.csv")

CANDIDATE_FIELDS = [
    "case_label",
    "variant_label",
    "input_poscar",
    "staged_variant_poscar",
    "dft_contcar",
    "staged_dft_contcar",
    "dft_energy_eV",
    "dft_reference_dir",
    "dft_reference_section",
    "source_dft_references_dir",
]

CASE_FIELDS = [
    "case_label",
    "input_poscar",
    "staged_case_input_poscar",
    "dft_references_dir",
    "n_variants",
]

ENDPOINT_CASE_FIELDS = [
    "case_label",
    "composition",
    "phase",
    "phase_code",
    "defect",
    "charge",
    "atom_count",
    "parent_atom_count",
    "supercell",
    "confidence",
    "input_poscar",
    "staged_case_input_poscar",
    "source_dir",
    "organized_path",
    "dft_references_dir",
    "endpoint_labels",
    "n_endpoints",
    "notes",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def parse_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(v) for v in values}


def keep_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    if row.get("source_type") != "training_neb":
        return False
    if row.get("composition") != args.composition:
        return False

    filters = {
        "phase": parse_filter(args.phase),
        "defect": parse_filter(args.defect),
        "charge": parse_filter(args.charge),
        "confidence": parse_filter(args.confidence),
    }
    for key, allowed in filters.items():
        if allowed is not None and row.get(key) not in allowed:
            return False
    return True


def endpoint_reference_dirs(dft_references_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    refs: list[tuple[Path, dict[str, Any]]] = []
    for ref_dir in discover_reference_dirs(dft_references_dir):
        meta = load_reference_metadata(ref_dir)
        label = str(meta.get("label", ref_dir.name))
        if label.startswith("endpoint_"):
            refs.append((ref_dir, meta))
    return sorted(refs, key=lambda item: str(item[1].get("label", item[0].name)))


def existing_structure_path(meta: dict[str, Any], ref_dir: Path) -> Path:
    structure_path = Path(str(meta.get("structure_path", "")))
    if structure_path.exists():
        return structure_path

    organized_poscar = ref_dir / "POSCAR"
    if organized_poscar.exists():
        return organized_poscar

    raise FileNotFoundError(f"No endpoint structure found for {ref_dir}")


def write_endpoint_bunya_notes(analysis_name: str) -> None:
    root = analysis_root(analysis_name)
    notes = f"""# NEB Endpoint Preservation Transfer Notes

Analysis: `{analysis_name}`

This workflow tests whether an MLIP relaxation started from a DFT NEB endpoint
stays in that endpoint's structure basin. It is not an SnB ground-cluster test.

Transfer to Bunya if running MLIP relaxations there:

```text
{root}/snb_variant_inputs/
{root}/analysis/candidate_manifest.csv
{root}/analysis/endpoint_case_metadata.csv
/home/rnpla/projects/mlip_phonons/src/defect_landscape/analysis_pipeline/
```

Transfer back:

```text
{root}/mlip_relaxed/
```

Then run locally:

```bash
python 12_compare_neb_endpoint_preservation.py --analysis-name {analysis_name}
python 13_write_neb_endpoint_report.py --analysis-name {analysis_name}
```

Do not transfer or duplicate POTCAR files for this workflow.
"""
    (root / "bunya_transfer_notes.md").write_text(notes)


def stage_endpoint_cases(args: argparse.Namespace) -> None:
    if args.reset:
        clear_analysis(args.analysis_name)
    else:
        ensure_analysis_dirs(args.analysis_name)

    root = analysis_root(args.analysis_name)
    rows = [row for row in read_manifest(args.tuple_manifest) if keep_row(row, args)]
    if not rows:
        raise RuntimeError("No NEB endpoint cases matched the requested filters.")

    candidate_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    endpoint_case_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for row in rows:
        case_label = row["case_label"]
        case_safe = safe_label(case_label)
        organized_path = Path(row["organized_path"])
        dft_references_dir = organized_path / "dft_references"
        input_poscar = Path(row["input_POSCAR"])
        if not input_poscar.exists():
            input_poscar = organized_path / "input" / "POSCAR"
        if not input_poscar.exists():
            skipped.append(f"{case_label}: missing input POSCAR")
            continue

        endpoint_refs = endpoint_reference_dirs(dft_references_dir)
        if not endpoint_refs:
            skipped.append(f"{case_label}: no endpoint reference folders")
            continue

        staged_case_input = root / "case_inputs" / case_safe / "POSCAR"
        copy_structure(input_poscar, staged_case_input)

        endpoint_labels: list[str] = []
        for ref_dir, meta in endpoint_refs:
            endpoint_label = safe_label(str(meta.get("label", ref_dir.name)))
            endpoint_poscar = existing_structure_path(meta, ref_dir)
            staged_endpoint = root / "snb_variant_inputs" / case_safe / endpoint_label / "POSCAR"
            staged_dft = root / "dft_references" / case_safe / endpoint_label / "CONTCAR"
            copy_structure(endpoint_poscar, staged_endpoint)
            copy_structure(endpoint_poscar, staged_dft)

            meta_out = root / "dft_references" / case_safe / endpoint_label / "reference_metadata.json"
            staged_meta = dict(meta)
            staged_meta["endpoint_preservation_role"] = "mlip_start_and_dft_reference"
            staged_meta["staged_endpoint_poscar"] = str(staged_endpoint)
            staged_meta["staged_dft_reference"] = str(staged_dft)
            meta_out.write_text(json.dumps(staged_meta, indent=2, sort_keys=True) + "\n")

            candidate_rows.append(
                {
                    "case_label": case_label,
                    "variant_label": endpoint_label,
                    "input_poscar": str(endpoint_poscar),
                    "staged_variant_poscar": str(staged_endpoint),
                    "dft_contcar": str(endpoint_poscar),
                    "staged_dft_contcar": str(staged_dft),
                    "dft_energy_eV": meta.get("energy_eV", ""),
                    "dft_reference_dir": str(meta.get("evidence_dir", ref_dir)),
                    "dft_reference_section": ref_dir.parent.name,
                    "source_dft_references_dir": str(dft_references_dir),
                }
            )
            endpoint_labels.append(endpoint_label)

        case_rows.append(
            {
                "case_label": case_label,
                "input_poscar": str(input_poscar),
                "staged_case_input_poscar": str(staged_case_input),
                "dft_references_dir": str(dft_references_dir),
                "n_variants": len(endpoint_labels),
            }
        )
        endpoint_case_rows.append(
            {
                "case_label": case_label,
                "composition": row.get("composition", ""),
                "phase": row.get("phase", ""),
                "phase_code": row.get("phase_code", ""),
                "defect": row.get("defect", ""),
                "charge": row.get("charge", ""),
                "atom_count": row.get("atom_count", ""),
                "parent_atom_count": row.get("parent_atom_count", ""),
                "supercell": row.get("supercell", ""),
                "confidence": row.get("confidence", ""),
                "input_poscar": str(input_poscar),
                "staged_case_input_poscar": str(staged_case_input),
                "source_dir": row.get("source_dir", ""),
                "organized_path": str(organized_path),
                "dft_references_dir": str(dft_references_dir),
                "endpoint_labels": "|".join(endpoint_labels),
                "n_endpoints": len(endpoint_labels),
                "notes": row.get("notes", ""),
            }
        )

    if not candidate_rows:
        raise RuntimeError("No endpoint candidates were staged.")

    out_dir = root / "analysis"
    write_csv_rows(candidate_manifest_path(args.analysis_name), candidate_rows, CANDIDATE_FIELDS)
    write_csv_rows(case_manifest_path(args.analysis_name), case_rows, CASE_FIELDS)
    write_csv_rows(out_dir / "endpoint_case_metadata.csv", endpoint_case_rows, ENDPOINT_CASE_FIELDS)
    write_csv_rows(out_dir / "endpoint_prepare_skipped.csv", [{"reason": s} for s in skipped], ["reason"])
    write_endpoint_bunya_notes(args.analysis_name)

    charges = Counter(row["charge"] for row in endpoint_case_rows)
    phases = Counter(row["phase"] for row in endpoint_case_rows)
    defects = Counter(row["defect"] for row in endpoint_case_rows)
    print(f"Prepared NEB endpoint preservation analysis: {args.analysis_name}")
    print(f"Cases staged: {len(endpoint_case_rows)}")
    print(f"Endpoint structures staged: {len(candidate_rows)}")
    print(f"Charge counts: {dict(sorted(charges.items()))}")
    print(f"Phase counts: {dict(sorted(phases.items()))}")
    print(f"Defect counts: {dict(sorted(defects.items()))}")
    if skipped:
        print(f"Skipped: {len(skipped)}; see endpoint_prepare_skipped.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CsPbI3 NEB endpoint-preservation cases. Each endpoint POSCAR is staged as "
            "both the MLIP starting structure and the DFT endpoint reference."
        )
    )
    parser.add_argument("--analysis-name", required=True, help="Named folder under src/defect_landscape/runs")
    parser.add_argument("--tuple-manifest", type=Path, default=SNB_TUPLE_MANIFEST)
    parser.add_argument("--composition", default="CsPbI3")
    parser.add_argument("--phase", nargs="*", help="Optional phase filter, e.g. gamma beta")
    parser.add_argument("--defect", nargs="*", help="Optional defect filter, e.g. V_I I_int")
    parser.add_argument("--charge", nargs="*", help="Optional charge filter, e.g. -1 0 +1")
    parser.add_argument(
        "--confidence",
        nargs="*",
        default=["HIGH", "MEDIUM"],
        help="Confidence levels to stage. Defaults to HIGH MEDIUM.",
    )
    parser.add_argument("--reset", action="store_true", help="Delete and recreate runs/<analysis-name>")
    args = parser.parse_args()

    stage_endpoint_cases(args)


if __name__ == "__main__":
    main()
