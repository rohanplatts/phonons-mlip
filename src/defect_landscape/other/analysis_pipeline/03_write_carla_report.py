from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import analysis_dir, fmt_float


def yes_no(x: bool) -> str:
    return "yes" if bool(x) else "no"


def parse_case_label(case_label: str) -> dict[str, str]:
    defect = "V_Br" if case_label.startswith("VBr_") else ("V_I" if case_label.startswith("VI_") else "")
    charge = "q0" if "_q0_" in case_label else ("q+1" if "_q+1_" in case_label else "")
    remainder = case_label
    for prefix in ["VBr_q0_", "VBr_q+1_", "VI_q0_", "VI_q+1_"]:
        if remainder.startswith(prefix):
            remainder = remainder[len(prefix) :]
            break
    if "_test" in remainder:
        site_label, test_label = remainder.rsplit("_test", 1)
        test_label = f"test{test_label}"
    else:
        site_label = remainder
        test_label = ""
    return {"defect": defect, "charge": charge, "site_label": site_label, "test_label": test_label}


def write_report(analysis_name: str) -> None:
    out_dir = analysis_dir(analysis_name)
    summary_path = out_dir / "case_model_summary.csv"
    clusters_path = out_dir / "mlip_clusters.csv"
    dft_clusters_path = out_dir / "dft_clusters.csv"
    rankings_path = out_dir / "variant_energy_rankings.csv"
    case_manifest_path = out_dir / "case_manifest.csv"
    candidate_manifest_path = out_dir / "candidate_manifest.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run 02_compare_to_existing_dft.py first.")

    summary = pd.read_csv(summary_path)
    mlip_clusters = pd.read_csv(clusters_path) if clusters_path.exists() else pd.DataFrame()
    dft_clusters = pd.read_csv(dft_clusters_path) if dft_clusters_path.exists() else pd.DataFrame()
    rankings = pd.read_csv(rankings_path) if rankings_path.exists() else pd.DataFrame()
    case_manifest = pd.read_csv(case_manifest_path) if case_manifest_path.exists() else pd.DataFrame()
    candidates = pd.read_csv(candidate_manifest_path) if candidate_manifest_path.exists() else pd.DataFrame()

    total = len(summary)
    top1 = int(summary["top1_ground_cluster_correct"].sum())
    top3 = int(summary["top3_contains_dft_ground"].sum())
    one_dft = int(summary["one_dft_refinement_safe_by_current_rule"].sum())
    review = int((summary["recommended_policy"] == "review").sum())

    by_model = {}
    for model, sub in summary.groupby("model_label"):
        model_ground = mlip_clusters[mlip_clusters["model_label"] == model] if not mlip_clusters.empty else pd.DataFrame()
        model_rankings = rankings[rankings["model_label"] == model] if not rankings.empty else pd.DataFrame()
        by_model[model] = {
            "n_cases": int(len(sub)),
            "top1_correct": int(sub["top1_ground_cluster_correct"].sum()),
            "top3_contains": int(sub["top3_contains_dft_ground"].sum()),
            "one_dft_safe": int(sub["one_dft_refinement_safe_by_current_rule"].sum()),
            "mean_relative_energy_mae_eV": float(sub["relative_energy_mae_eV"].mean()),
            "median_relative_energy_mae_eV": float(sub["relative_energy_mae_eV"].median()),
            "mean_ground_rms_A": None if model_ground.empty else float(model_ground["mapped_rms_A"].mean()),
            "median_ground_rms_A": None if model_ground.empty else float(model_ground["mapped_rms_A"].median()),
            "max_ground_rms_A": None if model_ground.empty else float(model_ground["mapped_rms_A"].max()),
            "mean_ground_max_A": None if model_ground.empty else float(model_ground["mapped_max_A"].mean()),
            "max_ground_max_A": None if model_ground.empty else float(model_ground["mapped_max_A"].max()),
            "variant_relative_energy_mae_eV": None if model_rankings.empty else float(model_rankings["abs_dE_error_eV"].mean()),
            "variant_relative_energy_median_abs_error_eV": None if model_rankings.empty else float(model_rankings["abs_dE_error_eV"].median()),
        }

    model_agreement_rows = []
    for case_label, sub in summary.groupby("case_label"):
        mapped = sorted(set(int(x) for x in sub["mlip_ground_mapped_dft_cluster_id"]))
        variants = sorted(set(str(x) for x in sub["mlip_ground_variant"]))
        all_top1 = bool(sub["top1_ground_cluster_correct"].all())
        all_top3 = bool(sub["top3_contains_dft_ground"].all())
        all_model_one_dft = bool(sub["one_dft_refinement_safe_by_current_rule"].all())
        agree_cluster = len(mapped) == 1
        consensus_one_dft = bool(all_model_one_dft and agree_cluster)
        model_agreement_rows.append(
            {
                "case_label": case_label,
                "n_models": int(len(sub)),
                "all_models_top1_correct": all_top1,
                "all_models_top3_contain_dft_ground": all_top3,
                "models_agree_on_mapped_dft_cluster": agree_cluster,
                "models_agree_on_variant": len(variants) == 1,
                "consensus_one_dft_safe": consensus_one_dft,
                "consensus_policy": "one_dft" if consensus_one_dft else ("top3_dft" if all_top3 else "review"),
                "mapped_dft_clusters": mapped,
                "mlip_ground_variants": variants,
            }
        )
    model_agreement = pd.DataFrame(model_agreement_rows)
    model_agreement.to_csv(out_dir / "case_consensus_summary.csv", index=False)

    n_cases = len(model_agreement)
    consensus_one = int(model_agreement["consensus_one_dft_safe"].sum()) if n_cases else 0
    consensus_top3 = int(model_agreement["all_models_top3_contain_dft_ground"].sum()) if n_cases else 0
    consensus_review = int((model_agreement["consensus_policy"] == "review").sum()) if n_cases else 0

    ground_rows = []
    if not mlip_clusters.empty and not dft_clusters.empty:
        for _, row in summary.sort_values(["case_label", "model_label"]).iterrows():
            mlip_match = mlip_clusters[
                (mlip_clusters["case_label"] == row["case_label"])
                & (mlip_clusters["model_label"] == row["model_label"])
                & (mlip_clusters["mlip_cluster_id"] == row["mlip_ground_cluster_id"])
            ]
            dft_match = dft_clusters[
                (dft_clusters["case_label"] == row["case_label"])
                & (dft_clusters["dft_cluster_id"] == row["mlip_ground_mapped_dft_cluster_id"])
            ]
            if mlip_match.empty or dft_match.empty:
                continue
            mlip_row = mlip_match.iloc[0]
            dft_row = dft_match.iloc[0]
            ground_rows.append(
                {
                    "case_label": row["case_label"],
                    "model_label": row["model_label"],
                    "same_dft_ground_cluster": bool(row["top1_ground_cluster_correct"]),
                    "identified_cluster_energy_error_eV": float(dft_row["dft_dE_eV"]),
                    "ground_geometry_rms_A": float(mlip_row["mapped_rms_A"]),
                    "ground_geometry_max_A": float(mlip_row["mapped_max_A"]),
                    "mlip_selected_variant": row["mlip_ground_variant"],
                    "dft_ground_variant": row["dft_ground_variant"],
                    "mapped_dft_representative_variant": mlip_row["mapped_dft_representative_variant"],
                }
            )
    ground_table = pd.DataFrame(ground_rows)
    if not ground_table.empty:
        ground_table.to_csv(out_dir / "ground_cluster_energy_geometry_summary.csv", index=False)

    tested_cases = []
    tested_defects = set()
    tested_charges = set()
    tested_sites = set()
    tested_orderings = set()
    if not case_manifest.empty:
        for _, row in case_manifest.sort_values("case_label").iterrows():
            parsed = parse_case_label(str(row["case_label"]))
            tested_defects.add(parsed["defect"])
            tested_charges.add(parsed["charge"])
            tested_sites.add(parsed["site_label"])
            tested_orderings.add(parsed["test_label"])
            tested_cases.append(
                {
                    "case_label": row["case_label"],
                    "defect": parsed["defect"],
                    "charge": parsed["charge"],
                    "site_label": parsed["site_label"],
                    "test_label": parsed["test_label"],
                    "n_snb_variants": int(row["n_variants"]),
                    "input_poscar": row["input_poscar"],
                    "dft_references_dir": row["dft_references_dir"],
                }
            )
    tested_cases_df = pd.DataFrame(tested_cases)
    if not tested_cases_df.empty:
        tested_cases_df.to_csv(out_dir / "tested_structure_summary.csv", index=False)

    charge_text = ", ".join(sorted(x for x in tested_charges if x)) or "unknown charge"
    defect_text = ", ".join(sorted(x for x in tested_defects if x)) or "unknown defect"
    site_text = ", ".join(sorted(x for x in tested_sites if x)) or "unknown site labels"
    ordering_text = ", ".join(sorted(x for x in tested_orderings if x)) or "unknown ordering labels"
    variant_text = (
        ", ".join(sorted(str(x) for x in candidates["variant_label"].dropna().unique()))
        if not candidates.empty and "variant_label" in candidates
        else "unknown SnB variants"
    )

    lines = []
    lines.append("# MLIP Ground-Cluster Validation For Carla")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can an MLIP identify the same ShakeNBreak ground cluster that DFT identifies, so that only a small number of structures need DFT refinement?")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Analysis: `{analysis_name}`")
    lines.append(f"- Scope: CsPbI2Br gamma {charge_text} ShakeNBreak cases")
    lines.append("- Validation mode: retrospective comparison against existing DFT-relaxed SnB variant folders")
    lines.append("- No new Bunya DFT jobs are generated by this workflow")
    lines.append("- This is a narrow mixed-halide vacancy test set, not a broad perovskite benchmark.")
    lines.append("")
    lines.append("## Precise Structures Tested")
    lines.append("")
    lines.append("- Composition/phase: `gamma CsPbI2Br`")
    lines.append(f"- Defects/charges: `{defect_text}` at `{charge_text}`")
    lines.append("- Cell type: one halide vacancy in the mixed-halide 160-atom parent, giving 159-atom defective cells")
    lines.append(f"- Halide/order variants: `{ordering_text}` from `Rohan_data/SNB/g_CsPbI2Br`")
    lines.append(f"- Site labels tested for each defect/test pair: `{site_text}`")
    lines.append(f"- SnB variants per case: `{variant_text}`")
    lines.append("")
    if tested_cases:
        lines.append("| case | defect | site | ordering | SnB variants |")
        lines.append("|---|---|---|---|---:|")
        for row in tested_cases:
            lines.append(
                f"| {row['case_label']} | {row['defect']} | {row['site_label']} | {row['test_label']} | {row['n_snb_variants']} |"
            )
        lines.append("")
    lines.append("")
    lines.append("## Direct Answer")
    lines.append("")
    lines.append(f"- Model/case evaluations: `{total}`")
    lines.append(f"- MLIP top-1 maps to DFT ground cluster: `{top1}/{total}`")
    lines.append(f"- MLIP top-3 contains DFT ground cluster: `{top3}/{total}`")
    lines.append(f"- Per-model current rule says one DFT refinement is safe: `{one_dft}/{total}`")
    lines.append(f"- Case-level consensus one-DFT safe, requiring model agreement: `{consensus_one}/{n_cases}`")
    lines.append(f"- Case-level consensus top-3 contains DFT ground: `{consensus_top3}/{n_cases}`")
    lines.append(f"- Case-level manual review: `{consensus_review}/{n_cases}`")
    lines.append("")
    lines.append("The practical interpretation is: use one DFT refinement only when the MLIP top cluster maps to the DFT ground cluster and both the DFT and MLIP cluster gaps are not small. Otherwise use a top-3 DFT safety net or review the case.")
    lines.append("")
    lines.append("Here, `identified_cluster_energy_error_eV` means the DFT relative energy of the DFT cluster selected by the MLIP, relative to the DFT ground cluster for that same case. It is therefore `0` when the MLIP-selected cluster is the DFT ground cluster. It is not an absolute MLIP-vs-DFT total-energy error.")
    lines.append("")
    lines.append("## By Model")
    lines.append("")
    for model, vals in by_model.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"- Cases: `{vals['n_cases']}`")
        lines.append(f"- Top-1 correct: `{vals['top1_correct']}/{vals['n_cases']}`")
        lines.append(f"- Top-3 contains DFT ground: `{vals['top3_contains']}/{vals['n_cases']}`")
        lines.append(f"- One-DFT safe by current rule: `{vals['one_dft_safe']}/{vals['n_cases']}`")
        lines.append(f"- Variant relative-energy MAE: `{fmt_float(vals['variant_relative_energy_mae_eV'], 4)} eV`")
        lines.append(f"- Variant relative-energy median absolute error: `{fmt_float(vals['variant_relative_energy_median_abs_error_eV'], 4)} eV`")
        lines.append(f"- Mean ground-cluster RMS geometry error: `{fmt_float(vals['mean_ground_rms_A'], 4)} A`")
        lines.append(f"- Median ground-cluster RMS geometry error: `{fmt_float(vals['median_ground_rms_A'], 4)} A`")
        lines.append(f"- Maximum ground-cluster RMS geometry error: `{fmt_float(vals['max_ground_rms_A'], 4)} A`")
        lines.append("")
    lines.append("## Ground-Cluster Energy/Geometry Table")
    lines.append("")
    lines.append("| case | model | same DFT ground cluster | identified cluster energy error eV | ground geometry RMS A | ground geometry max A |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, row in ground_table.iterrows():
        lines.append(
            f"| {row['case_label']} | {row['model_label']} | {yes_no(row['same_dft_ground_cluster'])} | "
            f"{fmt_float(row['identified_cluster_energy_error_eV'], 4)} | {fmt_float(row['ground_geometry_rms_A'], 4)} | "
            f"{fmt_float(row['ground_geometry_max_A'], 4)} |"
        )
    lines.append("")
    lines.append("The selected SnB variant label is deliberately not part of the main table because different SnB starting labels can relax into the same physical structure cluster. Variant labels are retained in `ground_cluster_energy_geometry_summary.csv` and `mlip_clusters.csv` for traceability.")
    lines.append("")
    lines.append("## Geometry Outputs")
    lines.append("")
    lines.append("- `ground_cluster_energy_geometry_summary.csv`: compact table backing the report table")
    lines.append("- `tested_structure_summary.csv`: exact case/site/order labels and input POSCAR paths tested")
    lines.append("- `dft_clusters.csv`: DFT ground clusters and their members")
    lines.append("- `mlip_clusters.csv`: MLIP clusters, nearest DFT cluster, StructureMatcher fit, RMS/max geometry distances")
    lines.append("- `mlip_dft_cluster_distances.csv`: MLIP-cluster to mapped DFT-cluster geometry distances by default; rerun comparison with `--distance-mode all` for every all-pairs distance")
    lines.append("- `case_consensus_summary.csv`: whether base and fine-tuned MACE agree by mapped DFT cluster")
    lines.append("")

    report_path = out_dir / "carla_ground_cluster_answer.md"
    report_path.write_text("\n".join(lines))

    payload = {
        "analysis_name": analysis_name,
        "n_model_case_evaluations": total,
        "top1_correct": top1,
        "top3_contains": top3,
        "one_dft_safe": one_dft,
        "manual_review": review,
        "n_cases": n_cases,
        "case_consensus_one_dft_safe": consensus_one,
        "case_consensus_top3_contains": consensus_top3,
        "case_consensus_manual_review": consensus_review,
        "by_model": by_model,
        "report_path": str(report_path),
    }
    if not ground_table.empty:
        payload["ground_cluster_energy_geometry_summary"] = ground_table.to_dict(orient="records")
    if not tested_cases_df.empty:
        payload["tested_structure_summary"] = tested_cases_df.to_dict(orient="records")
    (out_dir / "carla_ground_cluster_answer.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a concise Carla-facing validation report.")
    parser.add_argument("--analysis-name", required=True)
    args = parser.parse_args()
    write_report(args.analysis_name)


if __name__ == "__main__":
    main()
