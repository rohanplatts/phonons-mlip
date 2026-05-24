from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import analysis_dir


COLORS = {
    "base_mace": "tab:blue",
    "finetuned_mace": "tab:orange",
    "finetuned_mace_positive": "tab:green",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required analysis output: {path}")
    return path


def parity_plot(df: pd.DataFrame, out_path: Path, title: str, zoom_eV: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = df.copy()
    if zoom_eV is not None:
        plot_df = plot_df[(plot_df["mlip_dE_eV"] <= zoom_eV) & (plot_df["dft_dE_eV"] <= zoom_eV)].copy()

    for model_label, sub in plot_df.groupby("model_label"):
        ax.scatter(
            sub["mlip_dE_eV"],
            sub["dft_dE_eV"],
            s=28,
            alpha=0.75,
            label=f"{model_label} (n={len(sub)})",
            color=COLORS.get(model_label),
            edgecolors="none",
        )

    if plot_df.empty:
        max_e = zoom_eV or 0.1
    else:
        max_e = max(float(plot_df["mlip_dE_eV"].max()), float(plot_df["dft_dE_eV"].max()), 0.01)
    if zoom_eV is not None:
        max_e = zoom_eV
    else:
        max_e *= 1.05

    ax.plot([0, max_e], [0, max_e], linestyle="--", color="black", linewidth=1.8, label="perfect energy ranking")
    ax.set_xlim(-0.002 * max_e, max_e)
    ax.set_ylim(-0.002 * max_e, max_e)
    ax.set_xlabel("MLIP relative energy within case / eV")
    ax.set_ylabel("DFT relative energy within case / eV")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def geometry_plot(mlip_clusters: pd.DataFrame, out_path: Path) -> None:
    models = list(mlip_clusters["model_label"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)

    rng = np.random.default_rng(7)
    for ax, metric, ylabel in [
        (axes[0], "mapped_rms_A", "Mapped MLIP-DFT RMS displacement / A"),
        (axes[1], "mapped_max_A", "Mapped MLIP-DFT max displacement / A"),
    ]:
        values_by_model = [mlip_clusters.loc[mlip_clusters["model_label"] == m, metric].astype(float).to_numpy() for m in models]
        ax.boxplot(values_by_model, positions=np.arange(len(models)), widths=0.5, showfliers=False)
        for i, model_label in enumerate(models):
            vals = values_by_model[i]
            jitter = rng.normal(loc=0.0, scale=0.035, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=36,
                alpha=0.8,
                color=COLORS.get(model_label),
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_xticks(np.arange(len(models)), models, rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Geometry agreement for MLIP ground clusters mapped to DFT clusters")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def case_outcome_plot(summary: pd.DataFrame, mlip_clusters: pd.DataFrame, out_path: Path) -> None:
    case_order = sorted(summary["case_label"].unique())
    model_order = list(summary["model_label"].drop_duplicates())
    y_index = {case: i for i, case in enumerate(case_order)}

    fig, ax = plt.subplots(figsize=(10, max(6, 0.34 * len(case_order))))
    for model_i, model_label in enumerate(model_order):
        sub = mlip_clusters[mlip_clusters["model_label"] == model_label].copy()
        x = sub["mapped_rms_A"].astype(float)
        y = sub["case_label"].map(y_index).astype(float) + (model_i - 0.5) * 0.18
        ax.scatter(
            x,
            y,
            s=50,
            alpha=0.85,
            label=model_label,
            color=COLORS.get(model_label),
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_yticks(np.arange(len(case_order)), case_order)
    ax.invert_yaxis()
    ax.set_xlabel("Mapped MLIP-DFT RMS displacement / A")
    ax.set_title("Per-case geometry agreement; all shown cases map to the DFT ground cluster")
    ax.legend(frameon=True)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_metrics(variant_rankings: pd.DataFrame, mlip_clusters: pd.DataFrame, summary: pd.DataFrame, zoom_eV: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_cases": int(summary["case_label"].nunique()),
        "n_model_case_evaluations": int(len(summary)),
        "n_variant_energy_points": int(len(variant_rankings)),
        "top1_correct": int(summary["top1_ground_cluster_correct"].sum()),
        "top3_contains_dft_ground": int(summary["top3_contains_dft_ground"].sum()),
        "one_dft_policy": int((summary["recommended_policy"] == "one_dft").sum()),
        "points_outside_low_energy_zoom": int(
            ((variant_rankings["mlip_dE_eV"] > zoom_eV) | (variant_rankings["dft_dE_eV"] > zoom_eV)).sum()
        ),
        "by_model": {},
    }

    for model_label, sub in variant_rankings.groupby("model_label"):
        geom = mlip_clusters[mlip_clusters["model_label"] == model_label]
        model_summary = summary[summary["model_label"] == model_label]
        metrics["by_model"][model_label] = {
            "variant_energy_mae_eV": float(sub["abs_dE_error_eV"].mean()),
            "variant_energy_median_abs_error_eV": float(sub["abs_dE_error_eV"].median()),
            "top1_correct": int(model_summary["top1_ground_cluster_correct"].sum()),
            "n_cases": int(len(model_summary)),
            "mapped_rms_mean_A": float(geom["mapped_rms_A"].mean()),
            "mapped_rms_median_A": float(geom["mapped_rms_A"].median()),
            "mapped_rms_max_A": float(geom["mapped_rms_A"].max()),
            "mapped_max_mean_A": float(geom["mapped_max_A"].mean()),
            "mapped_max_max_A": float(geom["mapped_max_A"].max()),
        }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MLIP-vs-DFT energy and geometry summaries for an analysis run.")
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--zoom-eV", type=float, default=0.08, help="Low-energy zoom limit for the parity plot.")
    args = parser.parse_args()

    out_dir = analysis_dir(args.analysis_name)
    variant_rankings = pd.read_csv(require_file(out_dir / "variant_energy_rankings.csv"))
    mlip_clusters = pd.read_csv(require_file(out_dir / "mlip_clusters.csv"))
    summary = pd.read_csv(require_file(out_dir / "case_model_summary.csv"))

    parity_plot(
        variant_rankings,
        out_dir / "combined_mlip_vs_dft.png",
        "MLIP vs DFT relative energies across SnB variants",
    )
    parity_plot(
        variant_rankings,
        out_dir / "combined_mlip_vs_dft_low_energy_zoom.png",
        f"MLIP vs DFT relative energies, low-energy region <= {args.zoom_eV:.2f} eV",
        zoom_eV=args.zoom_eV,
    )
    geometry_plot(mlip_clusters, out_dir / "mlip_dft_geometry_summary.png")
    case_outcome_plot(summary, mlip_clusters, out_dir / "per_case_ground_cluster_geometry.png")

    metrics = make_metrics(variant_rankings, mlip_clusters, summary, args.zoom_eV)
    with (out_dir / "plot_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote plots and metrics to {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
