from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    DEFAULT_MODELS,
    DFT_AMBIGUOUS_GAP_EV,
    MLIP_AMBIGUOUS_GAP_EV,
    analysis_dir,
    candidate_manifest_path,
    compare_structures,
    get_matcher,
    read_csv_rows,
    result_json_path,
    structure_matcher_fit,
)


def load_candidates(analysis_name: str, case_filter: str | None) -> pd.DataFrame:
    rows = read_csv_rows(candidate_manifest_path(analysis_name))
    if not rows:
        raise FileNotFoundError("No candidate manifest found. Run 00_prepare_case.py first.")
    df = pd.DataFrame(rows)
    df["dft_energy_eV"] = df["dft_energy_eV"].astype(float)
    if case_filter:
        df = df[df["case_label"] == case_filter].copy()
    if df.empty:
        raise RuntimeError("No candidates matched requested case filter.")
    return df


def load_mlip_results(analysis_name: str, models: list[str], case_filter: str | None) -> pd.DataFrame:
    rows = []
    candidates = load_candidates(analysis_name, case_filter)
    for _, cand in candidates.iterrows():
        for model in models:
            result_path = result_json_path(analysis_name, model, cand["case_label"], cand["variant_label"])
            if not result_path.exists():
                continue
            with result_path.open() as f:
                result = json.load(f)
            rows.append(
                {
                    "model_label": model,
                    "case_label": cand["case_label"],
                    "variant_label": cand["variant_label"],
                    "mlip_energy_eV": float(result["energy_eV"]),
                    "mlip_max_force_eVA": float(result["max_force_eVA"]),
                    "mlip_converged": bool(result.get("converged", False)),
                    "mlip_contcar": result["relaxed_contcar"],
                    "result_json": str(result_path),
                }
            )
    if not rows:
        raise RuntimeError("No MLIP result.json files found for requested analysis/model filters.")
    return pd.DataFrame(rows)


def make_clusters(items: list[dict[str, Any]], structure_key: str, energy_key: str, prefix: str) -> list[dict[str, Any]]:
    from pymatgen.core import Structure

    clusters: list[dict[str, Any]] = []

    for item in sorted(items, key=lambda x: float(x[energy_key])):
        struct = Structure.from_file(item[structure_key])
        assigned = False
        for cluster in clusters:
            if get_matcher().fit(struct, cluster["representative_structure"]):
                cluster["members"].append(item)
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    f"{prefix}_cluster_id": len(clusters),
                    "representative_structure": struct,
                    "members": [item],
                }
            )

    out = []
    for cluster in clusters:
        best = min(cluster["members"], key=lambda x: float(x[energy_key]))
        out.append(
            {
                f"{prefix}_cluster_id": int(cluster[f"{prefix}_cluster_id"]),
                "representative_variant": best["variant_label"],
                "representative_file": best[structure_key],
                "best_energy_eV": float(best[energy_key]),
                "n_members": len(cluster["members"]),
                "members": sorted([m["variant_label"] for m in cluster["members"]]),
            }
        )
    return sorted(out, key=lambda x: x["best_energy_eV"])


def compare_cluster_pair(
    mlip_cluster: dict[str, Any],
    dft_cluster: dict[str, Any],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    mlip_file = Path(mlip_cluster["representative_file"])
    dft_file = Path(dft_cluster["representative_file"])
    key = (str(mlip_file), str(dft_file))

    if key not in cache:
        stats = compare_structures(mlip_file, dft_file)
        sm_fit = structure_matcher_fit(mlip_file, dft_file)
        cache[key] = {
            "dft_cluster_id": int(dft_cluster["dft_cluster_id"]),
            "dft_representative_variant": dft_cluster["representative_variant"],
            "structure_matcher_fit": bool(sm_fit),
            "rms_A": stats["rms_A"],
            "max_A": stats["max_A"],
            "mean_A": stats["mean_A"],
            "cell_max_abs_diff_A": stats["cell_max_abs_diff_A"],
            "per_species_json": json.dumps(stats["per_species"], sort_keys=True),
        }

    return cache[key]


def nearest_dft_cluster(
    mlip_cluster: dict[str, Any],
    dft_clusters: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    rows = [compare_cluster_pair(mlip_cluster, dft_cluster, cache) for dft_cluster in dft_clusters]
    rows = sorted(rows, key=lambda r: (not r["structure_matcher_fit"], r["rms_A"], r["max_A"]))
    return rows[0]


def gap(values: list[float]) -> float:
    vals = sorted(values)
    if len(vals) < 2:
        return np.nan
    return float(vals[1] - vals[0])


def compare(analysis_name: str, models: list[str], case_filter: str | None, distance_mode: str) -> None:
    out_dir = analysis_dir(analysis_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(analysis_name, case_filter)
    results = load_mlip_results(analysis_name, models, case_filter)
    case_groups = list(candidates.groupby("case_label"))

    dft_cluster_rows = []
    mlip_cluster_rows = []
    distance_rows = []
    ranking_rows = []
    summary_rows = []

    print(
        f"Comparing {len(case_groups)} cases with models {', '.join(models)}; "
        f"distance_mode={distance_mode}",
        flush=True,
    )

    for case_i, (case_label, case_df) in enumerate(case_groups, start=1):
        print(f"[{case_i}/{len(case_groups)}] {case_label}: clustering {len(case_df)} DFT variants", flush=True)
        dft_items = [
            {
                "case_label": row["case_label"],
                "variant_label": row["variant_label"],
                "dft_contcar": row["staged_dft_contcar"],
                "dft_energy_eV": float(row["dft_energy_eV"]),
            }
            for _, row in case_df.iterrows()
        ]
        dft_clusters = make_clusters(dft_items, "dft_contcar", "dft_energy_eV", "dft")
        dft_ground = dft_clusters[0]
        dft_gap = gap([c["best_energy_eV"] for c in dft_clusters])
        dft_min = dft_ground["best_energy_eV"]
        print(
            f"  DFT: {len(dft_clusters)} clusters; ground={dft_ground['representative_variant']}; "
            f"cluster_gap_eV={dft_gap:.6f}",
            flush=True,
        )

        for cluster in dft_clusters:
            dft_cluster_rows.append(
                {
                    "case_label": case_label,
                    "dft_cluster_id": cluster["dft_cluster_id"],
                    "representative_variant": cluster["representative_variant"],
                    "dft_energy_eV": cluster["best_energy_eV"],
                    "dft_dE_eV": cluster["best_energy_eV"] - dft_min,
                    "n_members": cluster["n_members"],
                    "members": "|".join(cluster["members"]),
                    "representative_file": cluster["representative_file"],
                    "is_dft_ground_cluster": cluster["dft_cluster_id"] == dft_ground["dft_cluster_id"],
                }
            )

        for model_label, model_case in results[results["case_label"] == case_label].groupby("model_label"):
            print(f"  {model_label}: clustering {len(model_case)} MLIP relaxations", flush=True)
            merged = model_case.merge(
                case_df[["case_label", "variant_label", "dft_energy_eV", "staged_dft_contcar"]],
                on=["case_label", "variant_label"],
                how="left",
            )
            mlip_items = [
                {
                    "case_label": row["case_label"],
                    "variant_label": row["variant_label"],
                    "mlip_contcar": row["mlip_contcar"],
                    "mlip_energy_eV": float(row["mlip_energy_eV"]),
                }
                for _, row in merged.iterrows()
            ]
            mlip_clusters = make_clusters(mlip_items, "mlip_contcar", "mlip_energy_eV", "mlip")
            mlip_min = mlip_clusters[0]["best_energy_eV"]
            mlip_gap = gap([c["best_energy_eV"] for c in mlip_clusters])

            mapped_top = []
            pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
            for cluster in mlip_clusters:
                nearest = nearest_dft_cluster(cluster, dft_clusters, pair_cache)
                mapped_top.append(nearest["dft_cluster_id"])

                dft_clusters_to_write = (
                    dft_clusters
                    if distance_mode == "all"
                    else [c for c in dft_clusters if int(c["dft_cluster_id"]) == int(nearest["dft_cluster_id"])]
                )
                for dft_cluster in dft_clusters_to_write:
                    pair = compare_cluster_pair(cluster, dft_cluster, pair_cache)
                    distance_rows.append(
                        {
                            "case_label": case_label,
                            "model_label": model_label,
                            "distance_mode": distance_mode,
                            "mlip_cluster_id": cluster["mlip_cluster_id"],
                            "dft_cluster_id": pair["dft_cluster_id"],
                            "mlip_representative_variant": cluster["representative_variant"],
                            "dft_representative_variant": pair["dft_representative_variant"],
                            "structure_matcher_fit": pair["structure_matcher_fit"],
                            "rms_A": pair["rms_A"],
                            "max_A": pair["max_A"],
                            "mean_A": pair["mean_A"],
                            "cell_max_abs_diff_A": pair["cell_max_abs_diff_A"],
                            "per_species_json": pair["per_species_json"],
                        }
                    )

                mlip_cluster_rows.append(
                    {
                        "case_label": case_label,
                        "model_label": model_label,
                        "mlip_cluster_id": cluster["mlip_cluster_id"],
                        "representative_variant": cluster["representative_variant"],
                        "mlip_energy_eV": cluster["best_energy_eV"],
                        "mlip_dE_eV": cluster["best_energy_eV"] - mlip_min,
                        "n_members": cluster["n_members"],
                        "members": "|".join(cluster["members"]),
                        "representative_file": cluster["representative_file"],
                        "mapped_dft_cluster_id": nearest["dft_cluster_id"],
                        "mapped_dft_representative_variant": nearest["dft_representative_variant"],
                        "mapped_structure_matcher_fit": nearest["structure_matcher_fit"],
                        "mapped_rms_A": nearest["rms_A"],
                        "mapped_max_A": nearest["max_A"],
                        "mapped_mean_A": nearest["mean_A"],
                        "mapped_per_species_json": nearest["per_species_json"],
                    }
                )

            merged["mlip_dE_eV"] = merged["mlip_energy_eV"] - merged["mlip_energy_eV"].min()
            merged["dft_dE_eV"] = merged["dft_energy_eV"] - merged["dft_energy_eV"].min()
            complete_mlip_variant_set = len(merged) == len(case_df)
            for _, row in merged.iterrows():
                ranking_rows.append(
                    {
                        "case_label": case_label,
                        "model_label": model_label,
                        "variant_label": row["variant_label"],
                        "mlip_energy_eV": row["mlip_energy_eV"],
                        "mlip_dE_eV": row["mlip_dE_eV"],
                        "dft_energy_eV": row["dft_energy_eV"],
                        "dft_dE_eV": row["dft_dE_eV"],
                        "abs_dE_error_eV": abs(row["mlip_dE_eV"] - row["dft_dE_eV"]),
                    }
                )

            dft_ground_id = int(dft_ground["dft_cluster_id"])
            mlip_ground_mapped_id = int(mapped_top[0])
            top1_correct = mlip_ground_mapped_id == dft_ground_id
            top3_contains = dft_ground_id in set(mapped_top[:3])
            top5_contains = dft_ground_id in set(mapped_top[:5])
            energy_mae = float(np.mean(np.abs(merged["mlip_dE_eV"] - merged["dft_dE_eV"])))
            dft_ambiguous = bool(np.isfinite(dft_gap) and dft_gap < DFT_AMBIGUOUS_GAP_EV)
            mlip_ambiguous = bool(np.isfinite(mlip_gap) and mlip_gap < MLIP_AMBIGUOUS_GAP_EV)
            one_dft_safe = bool(top1_correct and not dft_ambiguous and not mlip_ambiguous)
            one_dft_safe = bool(one_dft_safe and complete_mlip_variant_set)

            summary_rows.append(
                {
                    "case_label": case_label,
                    "model_label": model_label,
                    "n_variants": int(len(case_df)),
                    "n_mlip_variants": int(len(merged)),
                    "complete_mlip_variant_set": complete_mlip_variant_set,
                    "n_dft_clusters": int(len(dft_clusters)),
                    "n_mlip_clusters": int(len(mlip_clusters)),
                    "dft_ground_cluster_id": dft_ground_id,
                    "dft_ground_variant": dft_ground["representative_variant"],
                    "mlip_ground_cluster_id": int(mlip_clusters[0]["mlip_cluster_id"]),
                    "mlip_ground_variant": mlip_clusters[0]["representative_variant"],
                    "mlip_ground_mapped_dft_cluster_id": mlip_ground_mapped_id,
                    "top1_ground_cluster_correct": top1_correct,
                    "top3_contains_dft_ground": top3_contains,
                    "top5_contains_dft_ground": top5_contains,
                    "relative_energy_mae_eV": energy_mae,
                    "dft_cluster_gap_eV": dft_gap,
                    "mlip_cluster_gap_eV": mlip_gap,
                    "dft_ambiguous_gap": dft_ambiguous,
                    "mlip_ambiguous_gap": mlip_ambiguous,
                    "one_dft_refinement_safe_by_current_rule": one_dft_safe,
                    "recommended_policy": "one_dft" if one_dft_safe else ("top3_dft" if top3_contains else "review"),
                }
            )
            print(
                f"  {model_label}: {len(mlip_clusters)} MLIP clusters; "
                f"top1_correct={top1_correct}; top3_contains={top3_contains}; "
                f"policy={summary_rows[-1]['recommended_policy']}; "
                f"geometry_pairs_evaluated={len(pair_cache)}",
                flush=True,
            )

    pd.DataFrame(dft_cluster_rows).to_csv(out_dir / "dft_clusters.csv", index=False)
    pd.DataFrame(mlip_cluster_rows).to_csv(out_dir / "mlip_clusters.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(out_dir / "mlip_dft_cluster_distances.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(out_dir / "variant_energy_rankings.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "case_model_summary.csv", index=False)

    unresolved = summary[
        (summary["recommended_policy"] != "one_dft")
        | (~summary["top3_contains_dft_ground"])
        | (summary["dft_ambiguous_gap"])
        | (summary["mlip_ambiguous_gap"])
    ].copy()
    unresolved.to_csv(out_dir / "unresolved_cases.csv", index=False)

    print(f"Wrote comparison outputs to {out_dir}")
    print(summary[["case_label", "model_label", "top1_ground_cluster_correct", "top3_contains_dft_ground", "recommended_policy"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLIP-relaxed SnB clusters against existing DFT SnB clusters.")
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--case", help="Optional case label filter")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model labels to compare. Defaults to base_mace finetuned_mace.",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["mapped", "all"],
        default="mapped",
        help=(
            "mapped writes the physically relevant MLIP-cluster to mapped-DFT-cluster geometry rows. "
            "all additionally writes every MLIP-cluster/DFT-cluster pair."
        ),
    )
    args = parser.parse_args()
    compare(args.analysis_name, args.models, args.case, args.distance_mode)


if __name__ == "__main__":
    main()
