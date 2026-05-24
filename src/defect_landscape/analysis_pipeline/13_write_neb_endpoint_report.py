from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from pipeline_common import analysis_dir, fmt_float


ATTENTION_COLUMNS = [
    "case_label",
    "phase",
    "defect",
    "charge",
    "atom_count",
    "parent_atom_count",
    "supercell",
    "confidence",
    "model_label",
    "endpoint_label",
    "mlip_contcar",
    "dft_endpoint_contcar",
    "structure_matcher_fit_expected",
    "matches_expected_endpoint_cluster",
    "rms_A",
    "max_A",
    "mean_A",
    "cell_max_abs_diff_A",
    "nearest_dft_endpoint_label",
    "nearest_structure_matcher_fit",
    "nearest_rms_A",
    "nearest_max_A",
    "nearest_endpoint_is_expected",
    "preserved_expected_endpoint",
    "preservation_status",
    "mlip_energy_eV",
    "mlip_max_force_eVA",
    "mlip_converged",
    "include_vdw",
    "model_name",
    "result_json",
    "per_species_json",
]

MISSING_COLUMNS = [
    "case_label",
    "endpoint_label",
    "charge",
    "model_label",
    "expected_result_json",
]


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_None._"
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    table = table.fillna("")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in table.iterrows():
        body.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def load_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str)
    except EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def write_report(analysis_name: str) -> None:
    out_dir = analysis_dir(analysis_name)
    comparisons = load_csv(out_dir / "endpoint_geometry_comparisons.csv")
    model_summary = load_csv(out_dir / "endpoint_model_summary.csv")
    case_summary = load_csv(out_dir / "endpoint_case_summary.csv")
    case_meta = load_csv(out_dir / "endpoint_case_metadata.csv")
    attention = load_csv(out_dir / "endpoint_cases_needing_attention.csv", ATTENTION_COLUMNS)
    missing = load_csv(out_dir / "endpoint_missing_relaxations.csv", MISSING_COLUMNS)

    if comparisons.empty:
        raise RuntimeError("No endpoint comparison rows found. Run 12_compare_neb_endpoint_preservation.py first.")

    numeric_cols = [
        "mean_rms_A",
        "median_rms_A",
        "max_rms_A",
        "mean_max_A",
        "max_max_A",
        "fraction_preserved",
        "fraction_matches_expected_cluster",
    ]
    for col in numeric_cols:
        if col in model_summary:
            model_summary[col] = pd.to_numeric(model_summary[col], errors="coerce").map(
                lambda x: fmt_float(x, 4)
            )

    for col in ["rms_A", "max_A", "nearest_rms_A", "nearest_max_A"]:
        if col in attention:
            attention[col] = pd.to_numeric(attention[col], errors="coerce").map(lambda x: fmt_float(x, 4))

    tested = case_meta[
        [
            "case_label",
            "phase",
            "defect",
            "charge",
            "atom_count",
            "confidence",
            "endpoint_labels",
            "input_poscar",
        ]
    ].sort_values(["charge", "phase", "defect", "case_label"])
    tested.to_csv(out_dir / "endpoint_tested_structure_summary.csv", index=False)

    report = f"""# NEB Endpoint Preservation Report

Analysis: `{analysis_name}`

This test starts from DFT NEB endpoint structures and relaxes them with MLIPs. It answers:

> If an endpoint is already a DFT-validated metastable structure, does the MLIP relaxation preserve that endpoint geometry?

This is not the same as the ShakeNBreak ground-cluster test. The SnB test asks whether the MLIP selects the same lowest-energy cluster from many distorted candidates. This endpoint test asks whether the MLIP keeps each known NEB endpoint in its original structural basin.

Strict preservation rule:

```text
preserved_expected_endpoint = StructureMatcher(MLIP_relaxed_endpoint, expected_DFT_endpoint)
                              AND nearest_DFT_endpoint_label == expected_endpoint_label
```

The geometry columns are direct atom-matched distances between the MLIP-relaxed endpoint and the expected DFT endpoint.

## Model Summary

{markdown_table(model_summary, [
    "model_label",
    "charge",
    "n_endpoint_evaluations",
    "n_preserved",
    "n_matches_expected_cluster",
    "fraction_preserved",
    "fraction_matches_expected_cluster",
    "mean_rms_A",
    "max_rms_A",
])}

## Structures Tested

{markdown_table(tested, [
    "case_label",
    "phase",
    "defect",
    "charge",
    "atom_count",
    "confidence",
    "endpoint_labels",
])}

## Cases Needing Attention

{markdown_table(attention.sort_values(["model_label", "charge", "case_label", "endpoint_label"]) if not attention.empty else attention, [
    "case_label",
    "model_label",
    "endpoint_label",
    "charge",
    "preservation_status",
    "rms_A",
    "max_A",
    "nearest_dft_endpoint_label",
    "nearest_rms_A",
    "mlip_converged",
], max_rows=80)}

## Missing Scheduled Relaxations

{markdown_table(missing, [
    "case_label",
    "endpoint_label",
    "charge",
    "model_label",
    "expected_result_json",
], max_rows=80)}

## Output Files

```text
{out_dir}/endpoint_geometry_comparisons.csv
{out_dir}/endpoint_model_summary.csv
{out_dir}/endpoint_case_summary.csv
{out_dir}/endpoint_cases_needing_attention.csv
{out_dir}/endpoint_missing_relaxations.csv
{out_dir}/endpoint_tested_structure_summary.csv
{out_dir}/neb_endpoint_preservation_report.md
```
"""
    (out_dir / "neb_endpoint_preservation_report.md").write_text(report)

    payload: dict[str, Any] = {
        "analysis_name": analysis_name,
        "n_cases": int(case_meta["case_label"].nunique()),
        "n_endpoint_structures": int(pd.to_numeric(case_meta["n_endpoints"], errors="coerce").fillna(0).sum()),
        "n_completed_endpoint_model_evaluations": int(len(comparisons)),
        "n_cases_needing_attention_rows": int(len(attention)),
        "n_missing_scheduled_relaxations": int(len(missing)),
        "report_path": str(out_dir / "neb_endpoint_preservation_report.md"),
    }
    (out_dir / "neb_endpoint_preservation_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    print(f"Wrote {out_dir / 'neb_endpoint_preservation_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a concise NEB endpoint-preservation report.")
    parser.add_argument("--analysis-name", required=True)
    args = parser.parse_args()
    write_report(args.analysis_name)


if __name__ == "__main__":
    main()
