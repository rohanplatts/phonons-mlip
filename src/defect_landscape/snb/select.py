from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import analysis_dir, case_root, copy_structure, read_csv_rows, safe_label, safe_model_label, write_csv_rows
from .structure_metrics import structure_matcher_fit


SELECTED_FIELDS = [
    "selection_id",
    "case_name",
    "model_name",
    "model_label",
    "cluster_id",
    "best_candidate_id",
    "best_mlip_energy_eV",
    "best_mlip_dE_eV",
    "representative_contcar",
    "selected_poscar",
    "source_models",
    "source_model_clusters",
]


def cluster_files(results_root: str | Path, case_name: str, model_name: str | None = None) -> list[Path]:
    root = analysis_dir(results_root, case_name)
    if model_name:
        return [root / f"clusters_{safe_model_label(model_name)}.csv"]
    return sorted(root.glob("clusters_*.csv"))


def _load_cluster_rows(results_root: str | Path, case_name: str, model_name: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in cluster_files(results_root, case_name, model_name):
        if path.exists():
            rows.extend(read_csv_rows(path))
    if not rows:
        raise FileNotFoundError(f"No cluster CSVs found for case {case_name}")
    return rows


def _select_for_model(
    rows: list[dict[str, Any]],
    *,
    energy_window_eV: float,
    max_clusters: int,
    top_k_clusters: int | None,
    always_include_ground: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["best_mlip_dE_eV"]))
    selected = [row for row in ordered if float(row["best_mlip_dE_eV"]) <= energy_window_eV]
    if top_k_clusters is not None:
        selected = ordered[:top_k_clusters]
    selected = selected[:max_clusters]
    if always_include_ground and ordered and ordered[0] not in selected:
        selected = [ordered[0], *selected]
    return selected


def _dedupe_union(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    union: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (float(r["best_mlip_dE_eV"]), r["model_name"], int(r["cluster_id"]))):
        matched = None
        for existing in union:
            if structure_matcher_fit(row["representative_contcar"], existing["representative_contcar"]):
                matched = existing
                break
        if matched is None:
            new = dict(row)
            new["source_models"] = row["model_name"]
            new["source_model_clusters"] = f"{row['model_name']}:{row['cluster_id']}"
            union.append(new)
        else:
            models = set(str(matched["source_models"]).split("|"))
            models.add(row["model_name"])
            clusters = set(str(matched["source_model_clusters"]).split("|"))
            clusters.add(f"{row['model_name']}:{row['cluster_id']}")
            matched["source_models"] = "|".join(sorted(models))
            matched["source_model_clusters"] = "|".join(sorted(clusters))
    return union


def select_clusters(
    *,
    results_root: str | Path,
    case_name: str,
    model_name: str | None = None,
    energy_window_eV: float = 0.50,
    max_clusters: int = 10,
    top_k_clusters: int | None = None,
    always_include_ground: bool = True,
    union_across_models: bool = False,
    overwrite: bool = False,
) -> Path:
    rows = _load_cluster_rows(results_root, case_name, model_name)
    selected: list[dict[str, Any]] = []
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)
    for model in sorted(by_model):
        selected.extend(
            _select_for_model(
                by_model[model],
                energy_window_eV=energy_window_eV,
                max_clusters=max_clusters,
                top_k_clusters=top_k_clusters,
                always_include_ground=always_include_ground,
            )
        )

    if union_across_models:
        selected = _dedupe_union(selected)

    root = case_root(results_root, case_name)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        if union_across_models:
            selection_id = f"union_cluster_{idx:03d}"
            source_models = row.get("source_models", row["model_name"])
            source_clusters = row.get("source_model_clusters", f"{row['model_name']}:{row['cluster_id']}")
        else:
            selection_id = f"{safe_model_label(row['model_name'])}_cluster_{int(row['cluster_id']):03d}"
            source_models = row["model_name"]
            source_clusters = f"{row['model_name']}:{row['cluster_id']}"
        selected_poscar = root / "selected_for_dft" / safe_label(selection_id) / "POSCAR"
        copy_structure(row["representative_contcar"], selected_poscar, overwrite=True)
        out_rows.append(
            {
                "selection_id": selection_id,
                "case_name": case_name,
                "model_name": row["model_name"],
                "model_label": safe_model_label(row["model_name"]),
                "cluster_id": row["cluster_id"],
                "best_candidate_id": row["best_candidate_id"],
                "best_mlip_energy_eV": row["best_mlip_energy_eV"],
                "best_mlip_dE_eV": row["best_mlip_dE_eV"],
                "representative_contcar": row["representative_contcar"],
                "selected_poscar": str(selected_poscar),
                "source_models": source_models,
                "source_model_clusters": source_clusters,
            }
        )

    out = analysis_dir(results_root, case_name) / "selected_for_dft_manifest.csv"
    write_csv_rows(out, out_rows, SELECTED_FIELDS)
    return out
