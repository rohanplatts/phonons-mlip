from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    RUNS_ROOT,
    analysis_dir,
    analysis_root,
    candidate_manifest_path,
    compare_structures,
    read_csv_rows,
    result_json_path,
    safe_label,
)


DEFAULT_TOLERANCES = [
    ("default", 0.20, 0.30, 5.0),
    ("strict_1", 0.10, 0.20, 3.0),
    ("strict_2", 0.05, 0.10, 2.0),
    ("strict_3", 0.025, 0.05, 1.0),
    ("strict_4", 0.010, 0.03, 0.5),
]


@dataclass(frozen=True)
class Tolerance:
    label: str
    ltol: float
    stol: float
    angle_tol: float

    @property
    def setting(self) -> str:
        return f"{self.label}: ltol={self.ltol:g}, stol={self.stol:g}, angle_tol={self.angle_tol:g}"


def parse_tolerances(items: list[str] | None) -> list[Tolerance]:
    if not items:
        return [Tolerance(*item) for item in DEFAULT_TOLERANCES]
    tolerances = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"Expected --tolerance label:ltol:stol:angle_tol, got {item!r}")
        label, ltol, stol, angle_tol = parts
        tolerances.append(Tolerance(label, float(ltol), float(stol), float(angle_tol)))
    return tolerances


def get_matcher(tol: Tolerance):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(
        ltol=tol.ltol,
        stol=tol.stol,
        angle_tol=tol.angle_tol,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )


def structure_cache() -> dict[str, Any]:
    return {}


def load_structure(path: str | Path, cache: dict[str, Any]):
    from pymatgen.core import Structure

    key = str(Path(path).resolve())
    if key not in cache:
        cache[key] = Structure.from_file(key)
    return cache[key]


def fit_files(file_a: str | Path, file_b: str | Path, matcher: Any, cache: dict[str, Any]) -> bool:
    try:
        return bool(matcher.fit(load_structure(file_a, cache), load_structure(file_b, cache)))
    except Exception:
        return False


def discover_models(analysis_name: str) -> list[str]:
    root = analysis_root(analysis_name) / "mlip_relaxed"
    if not root.exists():
        raise FileNotFoundError(f"No mlip_relaxed directory found for {analysis_name}: {root}")
    models = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not models:
        raise RuntimeError(f"No model result directories found in {root}")
    return models


def load_candidates(analysis_name: str, case_filter: str | None = None) -> pd.DataFrame:
    rows = read_csv_rows(candidate_manifest_path(analysis_name))
    if not rows:
        raise FileNotFoundError(f"No candidate manifest found: {candidate_manifest_path(analysis_name)}")
    df = pd.DataFrame(rows)
    df["dft_energy_eV"] = df["dft_energy_eV"].astype(float)
    if case_filter:
        df = df[df["case_label"] == case_filter].copy()
    if df.empty:
        raise RuntimeError(f"No candidates matched case filter {case_filter!r} in {analysis_name}")
    return df


def load_model_results(analysis_name: str, model_label: str, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cand in candidates.iterrows():
        path = result_json_path(analysis_name, model_label, cand["case_label"], cand["variant_label"])
        if not path.exists():
            continue
        with path.open() as handle:
            result = json.load(handle)
        rows.append(
            {
                "case_label": cand["case_label"],
                "variant_label": cand["variant_label"],
                "model_label": model_label,
                "mlip_energy_eV": float(result["energy_eV"]),
                "mlip_contcar": result["relaxed_contcar"],
                "result_json": str(path),
            }
        )
    return pd.DataFrame(rows)


def make_clusters(
    items: list[dict[str, Any]],
    *,
    structure_key: str,
    energy_key: str,
    prefix: str,
    matcher: Any,
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: float(row[energy_key])):
        structure = load_structure(item[structure_key], cache)
        matched = None
        for cluster in clusters:
            if matcher.fit(structure, cluster["representative_structure"]):
                matched = cluster
                break
        if matched is None:
            clusters.append(
                {
                    f"{prefix}_cluster_id": len(clusters),
                    "representative_structure": structure,
                    "members": [item],
                }
            )
        else:
            matched["members"].append(item)

    out = []
    for cluster in clusters:
        best = min(cluster["members"], key=lambda row: float(row[energy_key]))
        out.append(
            {
                f"{prefix}_cluster_id": int(cluster[f"{prefix}_cluster_id"]),
                "representative_variant": best["variant_label"],
                "representative_file": best[structure_key],
                "best_energy_eV": float(best[energy_key]),
                "n_members": len(cluster["members"]),
                "members": sorted(member["variant_label"] for member in cluster["members"]),
            }
        )
    return sorted(out, key=lambda row: float(row["best_energy_eV"]))


def nearest_dft_cluster(mlip_file: str | Path, dft_clusters: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for cluster in dft_clusters:
        stats = compare_structures(Path(mlip_file), Path(cluster["representative_file"]))
        rows.append(
            {
                "dft_cluster_id": int(cluster["dft_cluster_id"]),
                "dft_representative_variant": cluster["representative_variant"],
                "rms_A": stats["rms_A"],
                "max_A": stats["max_A"],
                "mean_A": stats["mean_A"],
            }
        )
    return sorted(rows, key=lambda row: (row["rms_A"], row["max_A"]))[0]


def cluster_gap(clusters: list[dict[str, Any]]) -> float:
    if len(clusters) < 2:
        return np.nan
    return float(clusters[1]["best_energy_eV"] - clusters[0]["best_energy_eV"])


def analyse_one(
    *,
    analysis_name: str,
    models: list[str] | None,
    tolerances: list[Tolerance],
    case_filter: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = load_candidates(analysis_name, case_filter)
    model_labels = models or discover_models(analysis_name)
    cache = structure_cache()
    detailed_rows: list[dict[str, Any]] = []
    dft_cluster_rows: list[dict[str, Any]] = []

    print(f"{analysis_name}: {candidates['case_label'].nunique()} cases, models={', '.join(model_labels)}", flush=True)

    for strictness_rank, tol in enumerate(tolerances):
        matcher = get_matcher(tol)
        print(f"  tolerance {tol.setting}", flush=True)
        for case_label, case_df in candidates.groupby("case_label"):
            dft_items = [
                {
                    "case_label": row["case_label"],
                    "variant_label": row["variant_label"],
                    "dft_contcar": row["staged_dft_contcar"],
                    "dft_energy_eV": float(row["dft_energy_eV"]),
                }
                for _, row in case_df.iterrows()
            ]
            dft_clusters = make_clusters(
                dft_items,
                structure_key="dft_contcar",
                energy_key="dft_energy_eV",
                prefix="dft",
                matcher=matcher,
                cache=cache,
            )
            dft_ground = dft_clusters[0]
            dft_ground_id = int(dft_ground["dft_cluster_id"])
            for dft_cluster in dft_clusters:
                dft_cluster_rows.append(
                    {
                        "analysis_name": analysis_name,
                        "strictness_rank": strictness_rank,
                        "tolerance_label": tol.label,
                        "ltol": tol.ltol,
                        "stol": tol.stol,
                        "angle_tol": tol.angle_tol,
                        "case_label": case_label,
                        "dft_cluster_id": dft_cluster["dft_cluster_id"],
                        "dft_representative_variant": dft_cluster["representative_variant"],
                        "dft_cluster_energy_eV": dft_cluster["best_energy_eV"],
                        "dft_cluster_dE_eV": dft_cluster["best_energy_eV"] - dft_ground["best_energy_eV"],
                        "n_members": dft_cluster["n_members"],
                        "members": "|".join(dft_cluster["members"]),
                        "is_dft_ground_cluster": int(dft_cluster["dft_cluster_id"]) == dft_ground_id,
                        "representative_file": dft_cluster["representative_file"],
                    }
                )

            for model_label in model_labels:
                model_results = load_model_results(analysis_name, model_label, case_df)
                if model_results.empty:
                    detailed_rows.append(
                        {
                            "analysis_name": analysis_name,
                            "strictness_rank": strictness_rank,
                            "tolerance_label": tol.label,
                            "ltol": tol.ltol,
                            "stol": tol.stol,
                            "angle_tol": tol.angle_tol,
                            "case_label": case_label,
                            "model_label": model_label,
                            "error": "missing_mlip_results",
                        }
                    )
                    continue
                mlip_items = [
                    {
                        "case_label": row["case_label"],
                        "variant_label": row["variant_label"],
                        "mlip_contcar": row["mlip_contcar"],
                        "mlip_energy_eV": float(row["mlip_energy_eV"]),
                    }
                    for _, row in model_results.iterrows()
                ]
                mlip_clusters = make_clusters(
                    mlip_items,
                    structure_key="mlip_contcar",
                    energy_key="mlip_energy_eV",
                    prefix="mlip",
                    matcher=matcher,
                    cache=cache,
                )
                mlip_ground = mlip_clusters[0]
                mlip_ground_file = mlip_ground["representative_file"]
                strict_fit_ids = [
                    int(cluster["dft_cluster_id"])
                    for cluster in dft_clusters
                    if fit_files(mlip_ground_file, cluster["representative_file"], matcher, cache)
                ]
                nearest = nearest_dft_cluster(mlip_ground_file, dft_clusters)
                strict_agrees = dft_ground_id in strict_fit_ids
                nearest_agrees = int(nearest["dft_cluster_id"]) == dft_ground_id
                direct_ground_fit = fit_files(mlip_ground_file, dft_ground["representative_file"], matcher, cache)

                detailed_rows.append(
                    {
                        "analysis_name": analysis_name,
                        "strictness_rank": strictness_rank,
                        "tolerance_label": tol.label,
                        "ltol": tol.ltol,
                        "stol": tol.stol,
                        "angle_tol": tol.angle_tol,
                        "case_label": case_label,
                        "model_label": model_label,
                        "n_snb_candidates": int(len(case_df)),
                        "n_dft_clusters": int(len(dft_clusters)),
                        "n_mlip_clusters": int(len(mlip_clusters)),
                        "dft_ground_cluster_id": dft_ground_id,
                        "dft_ground_variant": dft_ground["representative_variant"],
                        "dft_ground_cluster_members": "|".join(dft_ground["members"]),
                        "dft_cluster_gap_eV": cluster_gap(dft_clusters),
                        "mlip_ground_cluster_id": int(mlip_ground["mlip_cluster_id"]),
                        "mlip_ground_variant": mlip_ground["representative_variant"],
                        "mlip_ground_cluster_members": "|".join(mlip_ground["members"]),
                        "mlip_cluster_gap_eV": cluster_gap(mlip_clusters),
                        "strict_fit_dft_cluster_ids": "|".join(str(i) for i in strict_fit_ids),
                        "strict_match_to_dft_ground": bool(strict_agrees),
                        "direct_mlip_ground_to_dft_ground_fit": bool(direct_ground_fit),
                        "nearest_dft_cluster_id_by_rms": int(nearest["dft_cluster_id"]),
                        "nearest_dft_representative_variant": nearest["dft_representative_variant"],
                        "nearest_dft_cluster_is_ground": bool(nearest_agrees),
                        "nearest_rms_A": nearest["rms_A"],
                        "nearest_max_A": nearest["max_A"],
                        "mlip_ground_file": mlip_ground_file,
                        "dft_ground_file": dft_ground["representative_file"],
                        "error": "",
                    }
                )

    return pd.DataFrame(detailed_rows), pd.DataFrame(dft_cluster_rows)


def make_summary(detail: pd.DataFrame) -> pd.DataFrame:
    clean = detail[detail["error"].fillna("") == ""].copy()
    if clean.empty:
        return clean
    grouped = clean.groupby(["analysis_name", "strictness_rank", "tolerance_label", "ltol", "stol", "angle_tol", "model_label"])
    rows = []
    for key, group in grouped:
        analysis_name, strictness_rank, tolerance_label, ltol, stol, angle_tol, model_label = key
        rows.append(
            {
                "analysis_name": analysis_name,
                "strictness_rank": strictness_rank,
                "tolerance_label": tolerance_label,
                "ltol": ltol,
                "stol": stol,
                "angle_tol": angle_tol,
                "model_label": model_label,
                "n_model_cases": len(group),
                "n_strict_match_to_dft_ground": int(group["strict_match_to_dft_ground"].astype(bool).sum()),
                "n_nearest_dft_cluster_is_ground": int(group["nearest_dft_cluster_is_ground"].astype(bool).sum()),
                "mean_n_dft_clusters": float(group["n_dft_clusters"].mean()),
                "max_n_dft_clusters": int(group["n_dft_clusters"].max()),
                "mean_n_mlip_clusters": float(group["n_mlip_clusters"].mean()),
                "max_n_mlip_clusters": int(group["n_mlip_clusters"].max()),
                "strict_fail_cases": "|".join(sorted(group.loc[~group["strict_match_to_dft_ground"].astype(bool), "case_label"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_name", "strictness_rank", "model_label"])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    visible = df.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in visible.iterrows()]
    if len(df) > max_rows:
        body.append(f"\n_Only showing first {max_rows} of {len(df)} rows. See CSV for complete table._")
    return "\n".join([header, sep, *body])


def write_markdown(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    failures = detail[(detail["error"].fillna("") == "") & (~detail["strict_match_to_dft_ground"].astype(bool))].copy()
    compact_case = detail[detail["error"].fillna("") == ""].copy()
    compact_case = compact_case[
        [
            "analysis_name",
            "tolerance_label",
            "ltol",
            "stol",
            "angle_tol",
            "case_label",
            "model_label",
            "n_dft_clusters",
            "n_mlip_clusters",
            "dft_ground_variant",
            "mlip_ground_variant",
            "strict_match_to_dft_ground",
            "nearest_dft_cluster_is_ground",
            "nearest_rms_A",
        ]
    ].sort_values(["analysis_name", "tolerance_label", "case_label", "model_label"])

    text = [
        "# Strict StructureMatcher Sensitivity",
        "",
        "This reuses existing DFT and MLIP-relaxed SnB structures. No relaxations are run.",
        "",
        "`strict_match_to_dft_ground` is the conservative answer: the MLIP-selected ground representative must fit the DFT ground-cluster representative under the listed StructureMatcher tolerances.",
        "",
        "`nearest_dft_cluster_is_ground` is diagnostic only: it says whether the nearest DFT cluster by direct RMS is the DFT ground cluster, even if the strict StructureMatcher fit failed.",
        "",
        "## Summary",
        "",
        markdown_table(
            summary,
            [
                "analysis_name",
                "tolerance_label",
                "model_label",
                "n_model_cases",
                "n_strict_match_to_dft_ground",
                "n_nearest_dft_cluster_is_ground",
                "mean_n_dft_clusters",
                "max_n_dft_clusters",
            ],
            max_rows=200,
        ),
        "",
        "## Strict Failures",
        "",
        markdown_table(
            failures,
            [
                "analysis_name",
                "tolerance_label",
                "case_label",
                "model_label",
                "n_dft_clusters",
                "dft_ground_variant",
                "mlip_ground_variant",
                "strict_fit_dft_cluster_ids",
                "nearest_dft_cluster_is_ground",
                "nearest_rms_A",
            ],
            max_rows=200,
        ),
        "",
        "## Compact Case Table",
        "",
        markdown_table(compact_case, list(compact_case.columns), max_rows=250),
        "",
        "See `strict_structurematcher_sensitivity.csv` for the complete case table.",
    ]
    path.write_text("\n".join(text) + "\n")


def write_outputs(out_dir: Path, detail: pd.DataFrame, dft_clusters: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = detail.sort_values(["analysis_name", "strictness_rank", "case_label", "model_label"])
    dft_clusters = dft_clusters.sort_values(["analysis_name", "strictness_rank", "case_label", "dft_cluster_id"])
    summary = make_summary(detail)
    detail.to_csv(out_dir / "strict_structurematcher_sensitivity.csv", index=False)
    summary.to_csv(out_dir / "strict_structurematcher_summary.csv", index=False)
    dft_clusters.to_csv(out_dir / "strict_structurematcher_dft_clusters.csv", index=False)
    write_markdown(out_dir / "strict_structurematcher_sensitivity.md", detail, summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recluster existing SnB DFT/MLIP structures under stricter StructureMatcher tolerances."
    )
    parser.add_argument("--analysis-name", nargs="+", required=True, help="One or more existing analysis run names.")
    parser.add_argument("--models", nargs="+", help="Optional model labels. Defaults to all mlip_relaxed subdirectories.")
    parser.add_argument("--case", help="Optional single case label filter.")
    parser.add_argument(
        "--tolerance",
        action="append",
        help="Custom tolerance as label:ltol:stol:angle_tol. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional combined output directory. Defaults to each analysis/analysis folder, plus a combined folder for multiple analyses.",
    )
    args = parser.parse_args()

    tolerances = parse_tolerances(args.tolerance)
    all_detail = []
    all_dft_clusters = []
    for analysis_name in args.analysis_name:
        detail, dft_clusters = analyse_one(
            analysis_name=analysis_name,
            models=args.models,
            tolerances=tolerances,
            case_filter=args.case,
        )
        if args.case:
            per_analysis_out = args.out_dir or (analysis_dir(analysis_name) / f"strict_structurematcher_{safe_label(args.case)}")
        else:
            per_analysis_out = analysis_dir(analysis_name)
        write_outputs(per_analysis_out, detail, dft_clusters)
        print(f"Wrote per-analysis outputs to {per_analysis_out}", flush=True)
        all_detail.append(detail)
        all_dft_clusters.append(dft_clusters)

    if (len(all_detail) > 1 or args.out_dir) and not (args.case and len(all_detail) == 1):
        out_dir = args.out_dir or (RUNS_ROOT / "strict_structurematcher_sensitivity")
        write_outputs(out_dir, pd.concat(all_detail, ignore_index=True), pd.concat(all_dft_clusters, ignore_index=True))
        print(f"Wrote combined outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
