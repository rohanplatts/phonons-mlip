#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


@dataclass(frozen=True)
class PathModelSpec:
    label: str
    path_source: Path
    summary_path: Path | None = None
    npz_path: Path | None = None


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"Label cannot be empty in: {value}")
    return label, path


def load_image_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))


def load_path(root: Path, prefer_contcar: bool = False) -> list:
    dirs = load_image_dirs(root)
    if not dirs:
        raise ValueError(f"No image dirs found under {root}")
    images = []
    for d in dirs:
        if prefer_contcar and (d / "CONTCAR").exists():
            images.append(read(d / "CONTCAR"))
        else:
            images.append(read(d / "POSCAR"))
    return images


def _final_images_from_traj(frames: list, expected_n_images: int | None) -> list:
    if expected_n_images is None or len(frames) <= expected_n_images:
        return frames
    if expected_n_images < 2:
        return frames[-expected_n_images:]
    if len(frames) % expected_n_images == 0:
        return frames[-expected_n_images:]
    return frames[-expected_n_images:]


def load_path_any(
    path: Path,
    prefer_contcar: bool = False,
    expected_n_images: int | None = None,
) -> list:
    if path.is_file():
        frames = list(read(path, ":"))
        return _final_images_from_traj(frames, expected_n_images)
    return load_path(path, prefer_contcar=prefer_contcar)


def _sidecar_from_source(path_source: Path, name: str) -> Path | None:
    base = path_source.parent if path_source.is_file() else path_source.parent
    candidate = base / name
    return candidate if candidate.exists() else None


def discover_path_model_specs(
    results_root: Path,
    *,
    include_labels: set[str] | None = None,
) -> list[PathModelSpec]:
    specs: list[PathModelSpec] = []
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if include_labels is not None and model_dir.name not in include_labels:
            continue
        raw_dir = model_dir / "raw"
        vasp_ci = raw_dir / "vasp_ci"
        traj = raw_dir / "neb_ci.traj"
        path_source = vasp_ci if vasp_ci.exists() else traj
        if not path_source.exists():
            continue
        specs.append(
            PathModelSpec(
                label=model_dir.name,
                path_source=path_source,
                summary_path=(raw_dir / "summary.txt") if (raw_dir / "summary.txt").exists() else None,
                npz_path=(raw_dir / "neb_raw.npz") if (raw_dir / "neb_raw.npz").exists() else None,
            )
        )
    return specs


def build_path_model_specs(
    *,
    explicit_entries: Sequence[tuple[str, Path]],
    results_root: Path | None,
    include_labels: Sequence[str],
) -> list[PathModelSpec]:
    if explicit_entries:
        return [
            PathModelSpec(
                label=label,
                path_source=path.resolve(),
                summary_path=_sidecar_from_source(path.resolve(), "summary.txt"),
                npz_path=_sidecar_from_source(path.resolve(), "neb_raw.npz"),
            )
            for label, path in explicit_entries
        ]

    if results_root is None:
        raise SystemExit("Provide either --model-root LABEL=PATH or --results-root PATH.")

    labels = set(include_labels) if include_labels else None
    specs = discover_path_model_specs(results_root.resolve(), include_labels=labels)
    if not specs:
        raise SystemExit(f"No path model artifacts found under {results_root}")
    return specs


def load_named_models(
    specs: Iterable[PathModelSpec],
    *,
    expected_n_images: int | None = None,
) -> dict[str, list]:
    models: dict[str, list] = {}
    for spec in specs:
        models[spec.label] = load_path_any(spec.path_source, expected_n_images=expected_n_images)
    return models


def build_color_map(names: list[str]) -> dict[str, str]:
    palette = [
        "#ff1f1f",
        "#b084d9",
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


def unwrap_path(images) -> np.ndarray:
    frac = np.stack([a.get_scaled_positions(wrap=False) for a in images], axis=0)
    out = np.empty_like(frac)
    out[0] = frac[0]
    for i in range(1, len(images)):
        df = frac[i] - frac[i - 1]
        df = (df + 0.5) % 1.0 - 0.5
        out[i] = out[i - 1] + df
    return out


def path_coord(images) -> np.ndarray:
    cell = np.asarray(images[0].cell.array, dtype=float)
    frac = unwrap_path(images)
    cart = frac @ cell
    s = np.zeros(len(images), dtype=float)
    for i in range(1, len(images)):
        dr = cart[i] - cart[i - 1]
        s[i] = s[i - 1] + float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))
    return s


def sample_path(images, u_grid: np.ndarray) -> np.ndarray:
    cell = np.asarray(images[0].cell.array, dtype=float)
    frac = unwrap_path(images)
    s = path_coord(images)
    u = s / s[-1] if s[-1] > 0 else s
    out = np.empty((len(u_grid), len(images[0]), 3), dtype=float)
    for k, uu in enumerate(u_grid):
        if uu <= u[0]:
            i = 0
            a = 0.0
        elif uu >= u[-1]:
            i = len(u) - 2
            a = 1.0
        else:
            i = int(np.searchsorted(u, uu) - 1)
            denom = u[i + 1] - u[i]
            a = 0.0 if denom == 0 else float((uu - u[i]) / denom)
        f = (1.0 - a) * frac[i] + a * frac[i + 1]
        out[k] = f @ cell
    return out


def species_blocks(numbers: np.ndarray) -> dict[int, np.ndarray]:
    blocks: dict[int, np.ndarray] = {}
    for z in np.unique(numbers):
        blocks[int(z)] = np.where(numbers == z)[0]
    return blocks


def hungarian_min(cost: np.ndarray) -> np.ndarray:
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")
    n = int(a.shape[0])
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1
    return assign


def displacement_error(ref, pred) -> tuple[float, float, float]:
    cell = np.asarray(ref.cell.array, dtype=float)
    rf = np.asarray(ref.get_scaled_positions(wrap=False), dtype=float)
    pf = np.asarray(pred.get_scaled_positions(wrap=False), dtype=float)
    ref_numbers = np.asarray(ref.get_atomic_numbers(), dtype=int)
    pred_numbers = np.asarray(pred.get_atomic_numbers(), dtype=int)
    if ref_numbers.shape != pred_numbers.shape:
        raise ValueError(f"Atom count mismatch: {ref_numbers.shape} vs {pred_numbers.shape}")

    per_atom = []
    for z, idx_r in species_blocks(ref_numbers).items():
        idx_p = np.where(pred_numbers == z)[0]
        if idx_p.size != idx_r.size:
            raise ValueError(f"Species count mismatch for Z={z}: {idx_r.size} vs {idx_p.size}")
        df = rf[idx_r, None, :] - pf[idx_p][None, :, :]
        df = (df + 0.5) % 1.0 - 0.5
        cost = np.linalg.norm(df @ cell, axis=2)
        assign = hungarian_min(cost)
        diff = (pf[idx_p[assign]] - rf[idx_r] + 0.5) % 1.0 - 0.5
        per_atom.append(np.linalg.norm(diff @ cell, axis=1))

    per_atom_arr = np.concatenate(per_atom, axis=0)
    rms = float(np.sqrt(np.mean(per_atom_arr**2)))
    mean = float(np.mean(per_atom_arr))
    mx = float(np.max(per_atom_arr))
    return rms, mean, mx


def load_barrier(summary_path: Path | None, npz_path: Path | None, dft_dat: Path | None = None) -> float | None:
    if summary_path is not None and summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("barrier_eV="):
                return float(line.split("=", 1)[1])
    if npz_path is not None and npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        if "e_mlip" in data.files:
            e = np.asarray(data["e_mlip"], dtype=float)
            return float(np.max(e))
    if dft_dat is not None and dft_dat.exists():
        data = np.loadtxt(dft_dat)
        data = np.atleast_2d(data)
        e = data[:, 2].astype(float) - float(data[0, 2])
        return float(np.max(e))
    return None


def compare_path(ref_images, model_images, grid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    ref_path = sample_path(ref_images, grid)
    model_path = sample_path(model_images, grid)
    errs = []
    mean_errs = []
    max_errs = []
    for r, m in zip(ref_path, model_path):
        r_atoms = ref_images[0].copy()
        m_atoms = model_images[0].copy()
        r_atoms.positions = r
        m_atoms.positions = m
        rms, mean, mx = displacement_error(r_atoms, m_atoms)
        errs.append(rms)
        mean_errs.append(mean)
        max_errs.append(mx)
    errs = np.asarray(errs, dtype=float)
    summary = {
        "mean_rms_disp_A": float(np.mean(errs)),
        "max_rms_disp_A": float(np.max(errs)),
        "mean_atom_disp_A": float(np.mean(mean_errs)),
        "max_atom_disp_A": float(np.max(max_errs)),
        "auc_rms_disp_A": float(np.trapezoid(errs, grid)),
    }
    return errs, summary


def _set_bandstyle_rcparams() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def plot(
    ref_images,
    models: dict[str, list],
    linear_images,
    out_png: Path,
    out_json: Path,
    *,
    dft_neb_dat: Path | None = None,
    barrier_sources: dict[str, tuple[Path | None, Path | None]] | None = None,
    title: str,
) -> None:
    ref_s = path_coord(ref_images)
    ref_grid = np.linspace(0.0, 1.0, 401)

    model_results: dict[str, dict[str, float]] = {}
    model_curves: dict[str, np.ndarray] = {}
    for name, images in models.items():
        errs, summary = compare_path(ref_images, images, ref_grid)
        model_curves[name] = errs
        model_results[name] = summary

    lin_errs, lin_summary = compare_path(ref_images, linear_images, ref_grid)
    model_curves["Linear interpolation"] = lin_errs
    model_results["Linear interpolation"] = lin_summary

    _set_bandstyle_rcparams()
    fig = plt.figure(figsize=(9.2, 5.3), facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")

    model_names = list(models)
    colors = {**build_color_map(model_names), "Linear interpolation": "#ff7a7a"}

    for idx, name in enumerate([*model_names, "Linear interpolation"]):
        curve = model_curves[name]
        ax.plot(
            ref_grid,
            curve,
            "--" if name == "Linear interpolation" else "-",
            lw=1.3 if name == "Linear interpolation" else 1.6,
            color=colors[name],
            alpha=0.98,
            solid_capstyle="round",
            dash_capstyle="round",
            label=name,
            zorder=2 if name == "Linear interpolation" else 3 + idx,
        )

    ax.axhline(0.0, color="#355cde", lw=0.9, ls=":", alpha=0.85, zorder=1)
    ymax = max(float(np.max(v)) for v in model_curves.values())
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ymax * 1.06 if ymax > 0 else 0.1)
    ax.set_xlabel("Reaction Progression [Å] → ")
    ax.set_ylabel("Atomic Displacement Error [Å] Relative to DFT")
    ax.set_title(title, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(which="both", direction="in", top=True, right=True, length=4.0, width=0.8)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.grid(False)

    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.35,
        handlelength=2.0,
        handletextpad=0.6,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.7)

    dft_total = float(ref_s[-1]) if ref_s.size else 1.0
    supplementary_barriers = {"DFT": load_barrier(None, None, dft_neb_dat)}
    if barrier_sources:
        for label, (summary_path, npz_path) in barrier_sources.items():
            supplementary_barriers[label] = load_barrier(summary_path, npz_path)

    fig.subplots_adjust(left=0.12, right=0.72, bottom=0.13, top=0.89)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    payload = {
        "grid_points": int(ref_grid.size),
        "dft_path_length_A": dft_total,
        "models": model_results,
        "supplementary_barriers_eV": supplementary_barriers,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_path(explicit_path: Path | None, outdir: Path, filename: str) -> Path:
    return explicit_path.resolve() if explicit_path is not None else (outdir / filename).resolve()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot NEB path fidelity relative to a DFT reference path.")

    ref_group = ap.add_argument_group("Reference input")
    ref_group.add_argument(
        "--dft-root",
        type=Path,
        required=True,
        help="DFT reference path root. May be an image directory containing 00/01/... or similar.",
    )
    ref_group.add_argument(
        "--dft-neb-dat",
        type=Path,
        default=None,
        help="Optional DFT `neb.dat` used for supplementary barrier reporting. Defaults to DFT_ROOT/neb.dat.",
    )
    ref_group.add_argument(
        "--prefer-contcar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer CONTCAR over POSCAR when loading DFT numbered image directories.",
    )

    model_group = ap.add_argument_group("Model inputs")
    model_group.add_argument(
        "--model-root",
        dest="model_roots",
        action="append",
        type=parse_named_path,
        default=[],
        help="Model path in LABEL=PATH form. PATH may be a numbered-image directory or trajectory file.",
    )
    model_group.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Discover models under RESULTS_ROOT/<model>/raw/vasp_ci or RESULTS_ROOT/<model>/raw/neb_ci.traj.",
    )
    model_group.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Optional model label filter when using --results-root. Can be passed multiple times.",
    )

    out_group = ap.add_argument_group("Outputs")
    out_group.add_argument(
        "--outdir",
        type=Path,
        default=Path("plot_summary"),
        help="Directory for generated outputs when explicit output paths are not provided.",
    )
    out_group.add_argument("--out-png", type=Path, default=None, help="Path-fidelity plot path.")
    out_group.add_argument("--out-json", type=Path, default=None, help="Path-fidelity JSON path.")

    style_group = ap.add_argument_group("Titles")
    style_group.add_argument("--title", default="NEB Path Fidelity")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    dft_root = args.dft_root.resolve()
    dft_neb_dat = args.dft_neb_dat.resolve() if args.dft_neb_dat is not None else (dft_root / "neb.dat")
    ref_images = load_path_any(dft_root, prefer_contcar=args.prefer_contcar)

    model_specs = build_path_model_specs(
        explicit_entries=args.model_roots,
        results_root=args.results_root,
        include_labels=args.include_model,
    )
    models = load_named_models(model_specs, expected_n_images=len(ref_images))
    linear_images = [ref_images[0], ref_images[-1]]
    barrier_sources = {spec.label: (spec.summary_path, spec.npz_path) for spec in model_specs}

    outdir = args.outdir.resolve()
    out_png = resolve_output_path(args.out_png, outdir, "combined_path_fidelity.png")
    out_json = resolve_output_path(args.out_json, outdir, "combined_path_fidelity.json")

    plot(
        ref_images=ref_images,
        models=models,
        linear_images=linear_images,
        out_png=out_png,
        out_json=out_json,
        dft_neb_dat=dft_neb_dat if dft_neb_dat.exists() else None,
        barrier_sources=barrier_sources,
        title=args.title,
    )

    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
