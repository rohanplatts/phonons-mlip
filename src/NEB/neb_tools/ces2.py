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


@dataclass(frozen=True)
class EnergyModelSpec:
    label: str
    npz_path: Path


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"Label cannot be empty in: {value}")
    return label, path


def load_neb_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    s = data[:, 1].astype(float)
    e = data[:, 2].astype(float)
    e = e - e[0]
    return s, e


def load_mlip_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    s = np.asarray(data["s_mlip"], dtype=float)
    e = np.asarray(data["e_mlip"], dtype=float)
    order = np.argsort(s)
    s = s[order]
    e = e[order] - e[order][0]
    return s, e


def linear_profile(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.interp(s, [s[0], s[-1]], [e[0], e[-1]])


def discover_energy_model_specs(
    results_root: Path,
    *,
    include_labels: set[str] | None = None,
) -> list[EnergyModelSpec]:
    specs: list[EnergyModelSpec] = []
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if include_labels is not None and model_dir.name not in include_labels:
            continue
        npz_path = model_dir / "raw" / "neb_raw.npz"
        if npz_path.exists():
            specs.append(EnergyModelSpec(label=model_dir.name, npz_path=npz_path))
    return specs


def build_energy_model_specs(
    *,
    explicit_entries: Sequence[tuple[str, Path]],
    results_root: Path | None,
    include_labels: Sequence[str],
) -> list[EnergyModelSpec]:
    if explicit_entries:
        return [EnergyModelSpec(label=label, npz_path=path.resolve()) for label, path in explicit_entries]

    if results_root is None:
        raise SystemExit("Provide either --model-npz LABEL=PATH or --results-root PATH.")

    labels = set(include_labels) if include_labels else None
    specs = discover_energy_model_specs(results_root.resolve(), include_labels=labels)
    if not specs:
        raise SystemExit(f"No model `raw/neb_raw.npz` files found under {results_root}")
    return specs


def load_models(specs: Iterable[EnergyModelSpec]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        models[spec.label] = load_mlip_npz(spec.npz_path)
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


def profile_rmse(ref_s: np.ndarray, ref_e: np.ndarray, s: np.ndarray, e: np.ndarray) -> float:
    interp = np.interp(ref_s, s, e)
    return float(np.sqrt(np.mean((interp - ref_e) ** 2)))


def metrics(ref_s: np.ndarray, ref_e: np.ndarray, s: np.ndarray, e: np.ndarray) -> dict[str, float]:
    return {
        "barrier_eV": float(np.max(e)),
        "deltaE_eV": float(e[-1]),
        "barrier_abs_err_eV": float(abs(np.max(e) - np.max(ref_e))),
        "deltaE_abs_err_eV": float(abs(e[-1] - ref_e[-1])),
        "profile_rmse_eV": float(profile_rmse(ref_s, ref_e, s, e)),
    }


def smooth_energy_curve(s: np.ndarray, e: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(s)
    s = s[order]
    e = e[order]

    if s.size < 2:
        return np.full_like(grid, float(np.mean(e)))

    edge_order = 2 if s.size >= 3 else 1
    slopes = np.gradient(e, s, edge_order=edge_order)

    grid_clamped = np.clip(grid, s[0], s[-1])
    idx = np.clip(np.searchsorted(s, grid_clamped) - 1, 0, s.size - 2)

    s0 = s[idx]
    s1 = s[idx + 1]
    e0 = e[idx]
    e1 = e[idx + 1]
    m0 = slopes[idx]
    m1 = slopes[idx + 1]

    span = s1 - s0
    span = np.where(span == 0, 1.0, span)
    t = (grid_clamped - s0) / span

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    return h00 * e0 + h10 * span * m0 + h01 * e1 + h11 * span * m1


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


def _style_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(which="both", direction="in", top=True, right=True, length=4.0, width=0.8)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.grid(False)


def _style_legend(ax, loc: str, *, bbox_to_anchor: tuple[float, float] | None = None) -> None:
    leg = ax.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.45,
        handlelength=2.2,
        handletextpad=0.7,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.7)


def plot_error(
    ref_s: np.ndarray,
    ref_e: np.ndarray,
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    out_png: Path,
    title: str,
) -> dict[str, dict[str, float]]:
    _set_bandstyle_rcparams()

    fig = plt.figure(figsize=(7.8, 5.6), facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")

    model_names = list(models)
    colors = build_color_map(model_names)

    summary: dict[str, dict[str, float]] = {}
    dense_s = np.linspace(float(np.min(ref_s)), float(np.max(ref_s)), 401)
    dft_smooth = smooth_energy_curve(ref_s, ref_e, dense_s)
    diff_curves: dict[str, np.ndarray] = {}

    for idx, name in enumerate(model_names):
        s, e = models[name]
        summary[name] = metrics(ref_s, ref_e, s, e)
        model_smooth = smooth_energy_curve(s, e, dense_s)
        diff_curves[name] = np.abs(model_smooth - dft_smooth)
        ax.plot(dense_s, diff_curves[name], "-", color=colors[name], lw=1.6, label=name, zorder=3 + idx)

    ax.axhline(0.0, color="#355cde", lw=0.9, ls=":", alpha=0.85, zorder=1)
    ax.set_xlabel("Reaction coordinate [Å]")
    ax.set_ylabel("Energy error [eV] relative to DFT")
    ax.set_title(title, pad=10)
    ax.set_xlim(float(np.min(ref_s)), float(np.max(ref_s)))
    ymin = 0.0
    ymax = max(np.max(v) for v in diff_curves.values()) if diff_curves else 0.0
    pad = max(0.02, 0.08 * max(0.05, ymax - ymin))
    ax.set_ylim(ymin, float(ymax) + pad)

    _style_axis(ax)
    _style_legend(ax, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.13, top=0.89)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return summary


def plot_energy_profiles(
    ref_s: np.ndarray,
    ref_e: np.ndarray,
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    out_png: Path,
    title: str,
) -> None:
    _set_bandstyle_rcparams()

    fig = plt.figure(figsize=(7.8, 5.6), facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")

    model_names = list(models)
    colors = {"DFT": "#111111", **build_color_map(model_names)}
    dense_s = np.linspace(float(np.min(ref_s)), float(np.max(ref_s)), 401)

    dft_smooth = smooth_energy_curve(ref_s, ref_e, dense_s)
    ax.plot(dense_s, dft_smooth, "-", color=colors["DFT"], lw=1.7, label="DFT", zorder=5)
    ax.plot(ref_s, ref_e, "o", color=colors["DFT"], ms=4.3, mec=colors["DFT"], mfc=colors["DFT"], zorder=6)

    for idx, name in enumerate(model_names):
        s, e = models[name]
        smooth_e = smooth_energy_curve(s, e, dense_s)
        ax.plot(dense_s, smooth_e, "-", color=colors[name], lw=1.6, label=name, zorder=3 + idx)
        ax.plot(s, e, "o", color=colors[name], ms=4.0, mec=colors[name], mfc=colors[name], zorder=3.2 + idx)

    ax.axhline(0.0, color="#355cde", lw=0.9, ls=":", alpha=0.85, zorder=1)
    ax.set_xlabel("Reaction coordinate [Å]")
    ax.set_ylabel("Energy [eV] relative to initial")
    ax.set_title(title, pad=10)

    all_curves = [ref_e] + [e for _, e in models.values()]
    ymin = min(float(np.min(v)) for v in all_curves)
    ymax = max(float(np.max(v)) for v in all_curves)
    yrange = max(0.05, ymax - ymin)

    ax.set_xlim(float(np.min(ref_s)), float(np.max(ref_s)))
    ax.set_ylim(ymin - 0.06 * yrange, ymax + 0.10 * yrange)

    _style_axis(ax)
    _style_legend(ax, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.13, top=0.89)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def resolve_output_path(explicit_path: Path | None, outdir: Path, filename: str) -> Path:
    return explicit_path.resolve() if explicit_path is not None else (outdir / filename).resolve()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot NEB energy-profile errors and energy profiles relative to a DFT reference."
    )
    ref_group = ap.add_argument_group("Reference input")
    ref_group.add_argument("--dft-neb-dat", type=Path, required=True, help="Reference DFT `neb.dat` file.")

    model_group = ap.add_argument_group("Model inputs")
    model_group.add_argument(
        "--model-npz",
        dest="model_npzs",
        action="append",
        type=parse_named_path,
        default=[],
        help="Model energy path in LABEL=PATH form. Can be passed multiple times.",
    )
    model_group.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Discover model `raw/neb_raw.npz` files under RESULTS_ROOT/<model>/raw/neb_raw.npz.",
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
    out_group.add_argument("--out-png", type=Path, default=None, help="Energy-error plot path.")
    out_group.add_argument(
        "--out-png-profiles",
        type=Path,
        default=None,
        help="Energy-profile plot path.",
    )
    out_group.add_argument("--out-json", type=Path, default=None, help="Energy metrics JSON path.")

    style_group = ap.add_argument_group("Titles")
    style_group.add_argument("--title", default="NEB Barrier Energy Accuracy")
    style_group.add_argument("--profiles-title", default="NEB Energy Profiles")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    model_specs = build_energy_model_specs(
        explicit_entries=args.model_npzs,
        results_root=args.results_root,
        include_labels=args.include_model,
    )

    ref_s, ref_e = load_neb_dat(args.dft_neb_dat.resolve())
    models = load_models(model_specs)

    outdir = args.outdir.resolve()
    out_png = resolve_output_path(args.out_png, outdir, "combined_energy_error.png")
    out_png_profiles = resolve_output_path(args.out_png_profiles, outdir, "combined_energy_profiles.png")
    out_json = resolve_output_path(args.out_json, outdir, "combined_energy_metrics.json")

    summary = plot_error(ref_s=ref_s, ref_e=ref_e, models=models, out_png=out_png, title=args.title)
    plot_energy_profiles(
        ref_s=ref_s,
        ref_e=ref_e,
        models=models,
        out_png=out_png_profiles,
        title=args.profiles_title,
    )

    payload = {
        "reference": {
            "dft_neb_dat": str(args.dft_neb_dat.resolve()),
            "barrier_eV": float(np.max(ref_e)),
            "deltaE_eV": float(ref_e[-1]),
        },
        "linear_interpolation": {
            "barrier_eV": float(np.max(linear_profile(ref_s, ref_e))),
            "deltaE_eV": float(linear_profile(ref_s, ref_e)[-1]),
        },
        "models": summary,
        "model_sources": {spec.label: str(spec.npz_path.resolve()) for spec in model_specs},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_png}")
    print(f"Wrote {out_png_profiles}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
