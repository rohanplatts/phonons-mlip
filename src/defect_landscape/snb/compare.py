from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dft_parse import check_dft
from .io import analysis_dir, read_csv_rows, write_csv_rows
from .structure_metrics import compare_structures, get_matcher, structure_matcher_fit


MLIP_VS_DFT_FIELDS = [
    "case_name",
    "model_name",
    "model_label",
    "n_dft_refinements",
    "n_complete",
    "mlip_ground_selection_id",
    "dft_ground_selection_id",
    "mlip_ground_dft_cluster_id",
    "dft_ground_cluster_id",
    "top1_correct",
    "top3_contains_dft_ground",
    "dft_energy_penalty_eV",
    "relative_energy_mae_eV",
    "ground_geometry_rms_A",
    "ground_geometry_max_A",
]

GEOMETRY_FIELDS = [
    "case_name",
    "selection_id",
    "model_name",
    "model_label",
    "cluster_id",
    "best_candidate_id",
    "best_mlip_dE_eV",
    "dft_dE_eV",
    "dft_cluster_id",
    "structure_matcher_fit",
    "mlip_to_dft_mean_A",
    "mlip_to_dft_rms_A",
    "mlip_to_dft_max_A",
    "cell_max_abs_diff_A",
    "representative_contcar",
    "dft_final_structure",
    "dft_dir",
    "per_species_json",
]


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _cluster_dft_rows(rows: list[dict[str, Any]], *, ltol: float, stol: float, angle_tol: float) -> None:
    from pymatgen.core import Structure

    matcher = get_matcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    clusters: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["final_energy_eV"])):
        structure = Structure.from_file(row["final_structure"])
        assigned = False
        for cluster in clusters:
            if matcher.fit(structure, cluster["representative_structure"]):
                row["dft_cluster_id"] = cluster["cluster_id"]
                cluster["members"].append(row)
                assigned = True
                break
        if not assigned:
            cluster_id = len(clusters)
            row["dft_cluster_id"] = cluster_id
            clusters.append({"cluster_id": cluster_id, "representative_structure": structure, "members": [row]})


def compare_dft(
    *,
    results_root: str | Path,
    case_name: str,
    matcher_ltol: float = 0.2,
    matcher_stol: float = 0.3,
    matcher_angle_tol: float = 5.0,
) -> Path:
    dft_manifest = analysis_dir(results_root, case_name) / "dft_validation_manifest.csv"
    if not dft_manifest.exists():
        raise FileNotFoundError(f"No DFT validation manifest found: {dft_manifest}")

    status_path = check_dft(results_root=results_root, case_name=case_name)
    status_rows = read_csv_rows(status_path)
    complete = [
        row
        for row in status_rows
        if _as_float(row.get("final_energy_eV")) is not None and row.get("final_structure") and Path(row["final_structure"]).exists()
    ]

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in complete:
        by_model.setdefault(row["model_name"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "case_name": case_name,
        "matcher": {
            "ltol": matcher_ltol,
            "stol": matcher_stol,
            "angle_tol": matcher_angle_tol,
            "primitive_cell": False,
            "scale": False,
            "attempt_supercell": False,
        },
        "models": {},
    }

    for model, rows in sorted(by_model.items()):
        for row in rows:
            row["final_energy_eV"] = float(row["final_energy_eV"])
            row["best_mlip_dE_eV"] = float(row["best_mlip_dE_eV"])
        min_dft = min(float(row["final_energy_eV"]) for row in rows)
        for row in rows:
            row["dft_dE_eV"] = float(row["final_energy_eV"]) - min_dft

        _cluster_dft_rows(rows, ltol=matcher_ltol, stol=matcher_stol, angle_tol=matcher_angle_tol)
        dft_ground = min(rows, key=lambda row: float(row["final_energy_eV"]))
        dft_ground_cluster = int(dft_ground["dft_cluster_id"])
        mlip_ranked = sorted(rows, key=lambda row: float(row["best_mlip_dE_eV"]))
        mlip_ground = mlip_ranked[0]
        top1_correct = int(mlip_ground["dft_cluster_id"]) == dft_ground_cluster
        top3_contains = any(int(row["dft_cluster_id"]) == dft_ground_cluster for row in mlip_ranked[:3])
        penalty = float(mlip_ground["dft_dE_eV"])
        mae = sum(abs(float(row["best_mlip_dE_eV"]) - float(row["dft_dE_eV"])) for row in rows) / len(rows)

        ground_stats = compare_structures(mlip_ground["representative_contcar"], mlip_ground["final_structure"])
        summary_rows.append(
            {
                "case_name": case_name,
                "model_name": model,
                "model_label": mlip_ground["model_label"],
                "n_dft_refinements": len(rows),
                "n_complete": len(rows),
                "mlip_ground_selection_id": mlip_ground["selection_id"],
                "dft_ground_selection_id": dft_ground["selection_id"],
                "mlip_ground_dft_cluster_id": mlip_ground["dft_cluster_id"],
                "dft_ground_cluster_id": dft_ground_cluster,
                "top1_correct": top1_correct,
                "top3_contains_dft_ground": top3_contains,
                "dft_energy_penalty_eV": penalty,
                "relative_energy_mae_eV": mae,
                "ground_geometry_rms_A": ground_stats["rms_A"],
                "ground_geometry_max_A": ground_stats["max_A"],
            }
        )

        model_geometry: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: float(item["best_mlip_dE_eV"])):
            stats = compare_structures(row["representative_contcar"], row["final_structure"])
            matcher_fit = structure_matcher_fit(
                row["representative_contcar"],
                row["final_structure"],
                ltol=matcher_ltol,
                stol=matcher_stol,
                angle_tol=matcher_angle_tol,
            )
            geometry_row = {
                "case_name": case_name,
                "selection_id": row["selection_id"],
                "model_name": model,
                "model_label": row["model_label"],
                "cluster_id": row["cluster_id"],
                "best_candidate_id": row["best_candidate_id"],
                "best_mlip_dE_eV": row["best_mlip_dE_eV"],
                "dft_dE_eV": row["dft_dE_eV"],
                "dft_cluster_id": row["dft_cluster_id"],
                "structure_matcher_fit": matcher_fit,
                "mlip_to_dft_mean_A": stats["mean_A"],
                "mlip_to_dft_rms_A": stats["rms_A"],
                "mlip_to_dft_max_A": stats["max_A"],
                "cell_max_abs_diff_A": stats["cell_max_abs_diff_A"],
                "representative_contcar": row["representative_contcar"],
                "dft_final_structure": row["final_structure"],
                "dft_dir": row["dft_dir"],
                "per_species_json": stats["per_species_json"],
            }
            geometry_rows.append(geometry_row)
            model_geometry.append(geometry_row)

        metrics["models"][model] = {
            "n_dft_refinements": len(rows),
            "top1_correct": top1_correct,
            "top3_contains_dft_ground": top3_contains,
            "dft_energy_penalty_eV": penalty,
            "relative_energy_mae_eV": mae,
            "mlip_ground_selection_id": mlip_ground["selection_id"],
            "dft_ground_selection_id": dft_ground["selection_id"],
            "dft_ground_cluster_id": dft_ground_cluster,
            "geometry": model_geometry,
        }

    out_dir = analysis_dir(results_root, case_name)
    summary_path = out_dir / "mlip_vs_dft.csv"
    geometry_path = out_dir / "geometry_comparisons.csv"
    metrics_path = out_dir / "ranking_metrics.json"
    write_csv_rows(summary_path, summary_rows, MLIP_VS_DFT_FIELDS)
    write_csv_rows(geometry_path, geometry_rows, GEOMETRY_FIELDS)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return summary_path
