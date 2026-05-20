from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import analysis_dir, read_csv_rows, safe_model_label, write_csv_rows
from .relax import collect_relaxation_results
from .structure_metrics import compare_structures, get_matcher


CLUSTER_FIELDS = [
    "model_name",
    "model_label",
    "cluster_id",
    "n_members",
    "best_candidate_id",
    "best_mlip_energy_eV",
    "best_mlip_dE_eV",
    "representative_contcar",
    "members",
]

MEMBER_FIELDS = [
    "model_name",
    "model_label",
    "cluster_id",
    "candidate_id",
    "mlip_energy_eV",
    "mlip_dE_eV",
    "relaxed_contcar",
    "distance_to_representative_rms_A",
    "distance_to_representative_max_A",
]


def available_models(results_root: str | Path, case_name: str) -> list[str]:
    rows = collect_relaxation_results(results_root, case_name)
    return sorted({row["model_name"] for row in rows})


def cluster_model(
    *,
    model_name: str,
    results_root: str | Path,
    case_name: str,
    matcher_ltol: float = 0.2,
    matcher_stol: float = 0.3,
    matcher_angle_tol: float = 5.0,
) -> Path:
    from pymatgen.core import Structure

    results = [row for row in collect_relaxation_results(results_root, case_name, model_name) if row["model_name"] == model_name]
    if not results:
        raise RuntimeError(f"No relaxation results found for {model_name} in case {case_name}")
    min_energy = min(float(row["energy_eV"]) for row in results)
    items = sorted(
        [
            {
                **row,
                "mlip_dE_eV": float(row["energy_eV"]) - min_energy,
            }
            for row in results
        ],
        key=lambda row: float(row["energy_eV"]),
    )

    matcher = get_matcher(ltol=matcher_ltol, stol=matcher_stol, angle_tol=matcher_angle_tol)
    clusters: list[dict[str, Any]] = []
    for item in items:
        structure = Structure.from_file(item["relaxed_contcar"])
        assigned = False
        for cluster in clusters:
            if matcher.fit(structure, cluster["representative_structure"]):
                cluster["members"].append(item)
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "cluster_id": len(clusters),
                    "representative_structure": structure,
                    "members": [item],
                }
            )

    cluster_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        best = min(cluster["members"], key=lambda row: float(row["energy_eV"]))
        cluster_id = int(cluster["cluster_id"])
        rep = best["relaxed_contcar"]
        for member in cluster["members"]:
            stats = compare_structures(member["relaxed_contcar"], rep)
            member_rows.append(
                {
                    "model_name": model_name,
                    "model_label": safe_model_label(model_name),
                    "cluster_id": cluster_id,
                    "candidate_id": member["candidate_id"],
                    "mlip_energy_eV": member["energy_eV"],
                    "mlip_dE_eV": member["mlip_dE_eV"],
                    "relaxed_contcar": member["relaxed_contcar"],
                    "distance_to_representative_rms_A": stats["rms_A"],
                    "distance_to_representative_max_A": stats["max_A"],
                }
            )
        cluster_rows.append(
            {
                "model_name": model_name,
                "model_label": safe_model_label(model_name),
                "cluster_id": cluster_id,
                "n_members": len(cluster["members"]),
                "best_candidate_id": best["candidate_id"],
                "best_mlip_energy_eV": best["energy_eV"],
                "best_mlip_dE_eV": best["mlip_dE_eV"],
                "representative_contcar": rep,
                "members": "|".join(sorted(member["candidate_id"] for member in cluster["members"])),
            }
        )

    out_dir = analysis_dir(results_root, case_name)
    cluster_path = out_dir / f"clusters_{safe_model_label(model_name)}.csv"
    member_path = out_dir / f"cluster_members_{safe_model_label(model_name)}.csv"
    write_csv_rows(cluster_path, cluster_rows, CLUSTER_FIELDS)
    write_csv_rows(member_path, member_rows, MEMBER_FIELDS)
    (out_dir / f"cluster_settings_{safe_model_label(model_name)}.json").write_text(
        json.dumps(
            {
                "matcher_ltol": matcher_ltol,
                "matcher_stol": matcher_stol,
                "matcher_angle_tol": matcher_angle_tol,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return cluster_path

