from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    analysis_dir,
    analysis_root,
    candidate_manifest_path,
    compare_structures,
    fmt_float,
    read_csv_rows,
    result_json_path,
    structure_matcher_fit,
)


SCHEDULED_MODELS = [
    "base_mace",
    "finetuned_mace",
    "finetuned_mace_positive",
    "finetuned_mace_negative",
]

MISSING_FIELDS = [
    "case_label",
    "endpoint_label",
    "charge",
    "model_label",
    "expected_result_json",
]


def scheduled_models_for_charge(charge: str) -> list[str]:
    models = ["base_mace", "finetuned_mace"]
    if charge == "+1":
        models.append("finetuned_mace_positive")
    elif charge == "-1":
        models.append("finetuned_mace_negative")
    return models


def load_candidates(analysis_name: str) -> pd.DataFrame:
    rows = read_csv_rows(candidate_manifest_path(analysis_name))
    if not rows:
        raise FileNotFoundError("No candidate manifest found. Run 11_prepare_neb_endpoint_cases.py first.")
    df = pd.DataFrame(rows)
    df = df[df["variant_label"].str.startswith("endpoint_")].copy()
    if df.empty:
        raise RuntimeError("No endpoint candidates found in candidate_manifest.csv.")
    return df


def load_case_metadata(analysis_name: str) -> pd.DataFrame:
    meta_path = analysis_dir(analysis_name) / "endpoint_case_metadata.csv"
    rows = read_csv_rows(meta_path)
    if not rows:
        raise FileNotFoundError(f"No endpoint case metadata found: {meta_path}")
    return pd.DataFrame(rows)


def require_structure_matcher_available() -> None:
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: F401
        from pymatgen.core import Structure  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "pymatgen StructureMatcher is required for endpoint preservation. "
            "Activate mace_env or run with /home/rnpla/anaconda3/envs/mace_env/bin/python."
        ) from exc


def bool_text(value: Any) -> str:
    return "True" if bool(value) else "False"


def compare_pair(
    mlip_file: Path,
    dft_file: Path,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (str(mlip_file), str(dft_file))
    if key not in cache:
        stats = compare_structures(mlip_file, dft_file)
        cache[key] = {
            "structure_matcher_fit": bool(structure_matcher_fit(mlip_file, dft_file)),
            "rms_A": stats["rms_A"],
            "max_A": stats["max_A"],
            "mean_A": stats["mean_A"],
            "cell_max_abs_diff_A": stats["cell_max_abs_diff_A"],
            "n_atoms": stats["n_atoms"],
            "per_species_json": json.dumps(stats["per_species"], sort_keys=True),
        }
    return cache[key]


def preservation_status(expected_fit: bool, nearest_label: str, expected_label: str, nearest_fit: bool) -> str:
    if expected_fit and nearest_label == expected_label:
        return "preserved"
    if expected_fit and nearest_label != expected_label:
        return "expected_cluster_match_but_nearest_other_endpoint"
    if (not expected_fit) and nearest_fit and nearest_label != expected_label:
        return "switched_to_other_endpoint"
    if nearest_label == expected_label:
        return "nearest_expected_but_no_structure_matcher_fit"
    return "unresolved"


def compare(analysis_name: str, models: list[str], case_filter: str | None) -> None:
    require_structure_matcher_available()

    out_dir = analysis_dir(analysis_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(analysis_name)
    case_meta = load_case_metadata(analysis_name)
    if case_filter:
        candidates = candidates[candidates["case_label"] == case_filter].copy()
        case_meta = case_meta[case_meta["case_label"] == case_filter].copy()
    if candidates.empty:
        raise RuntimeError("No endpoint candidates matched the requested case filter.")

    meta_by_case = case_meta.set_index("case_label").to_dict(orient="index")
    endpoints_by_case = {
        case_label: group.to_dict(orient="records") for case_label, group in candidates.groupby("case_label")
    }

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pair_cache: dict[tuple[str, str], dict[str, Any]] = {}

    print(
        f"Comparing NEB endpoint preservation for {len(endpoints_by_case)} cases "
        f"with model filter: {', '.join(models)}",
        flush=True,
    )

    for case_i, (case_label, endpoints) in enumerate(sorted(endpoints_by_case.items()), start=1):
        meta = meta_by_case.get(case_label, {})
        charge = str(meta.get("charge", ""))
        scheduled = [model for model in scheduled_models_for_charge(charge) if model in models]
        print(f"[{case_i}/{len(endpoints_by_case)}] {case_label}: {len(endpoints)} endpoints, charge {charge}", flush=True)

        for endpoint in endpoints:
            endpoint_label = endpoint["variant_label"]
            for model_label in scheduled:
                result_path = result_json_path(analysis_name, model_label, case_label, endpoint_label)
                if not result_path.exists():
                    missing_rows.append(
                        {
                            "case_label": case_label,
                            "endpoint_label": endpoint_label,
                            "charge": charge,
                            "model_label": model_label,
                            "expected_result_json": str(result_path),
                        }
                    )
                    continue

                with result_path.open() as f:
                    result = json.load(f)

                mlip_file = Path(result["relaxed_contcar"])
                expected_file = Path(endpoint["staged_dft_contcar"])
                expected = compare_pair(mlip_file, expected_file, pair_cache)

                nearest_rows = []
                for dft_endpoint in endpoints:
                    dft_file = Path(dft_endpoint["staged_dft_contcar"])
                    stats = compare_pair(mlip_file, dft_file, pair_cache)
                    nearest_rows.append(
                        {
                            "endpoint_label": dft_endpoint["variant_label"],
                            **stats,
                        }
                    )
                nearest = sorted(
                    nearest_rows,
                    key=lambda item: (
                        not bool(item["structure_matcher_fit"]),
                        float(item["rms_A"]),
                        float(item["max_A"]),
                    ),
                )[0]

                status = preservation_status(
                    bool(expected["structure_matcher_fit"]),
                    str(nearest["endpoint_label"]),
                    endpoint_label,
                    bool(nearest["structure_matcher_fit"]),
                )

                rows.append(
                    {
                        "case_label": case_label,
                        "phase": meta.get("phase", ""),
                        "defect": meta.get("defect", ""),
                        "charge": charge,
                        "atom_count": meta.get("atom_count", ""),
                        "parent_atom_count": meta.get("parent_atom_count", ""),
                        "supercell": meta.get("supercell", ""),
                        "confidence": meta.get("confidence", ""),
                        "model_label": model_label,
                        "endpoint_label": endpoint_label,
                        "mlip_contcar": str(mlip_file),
                        "dft_endpoint_contcar": str(expected_file),
                        "structure_matcher_fit_expected": bool(expected["structure_matcher_fit"]),
                        "matches_expected_endpoint_cluster": bool(expected["structure_matcher_fit"]),
                        "rms_A": expected["rms_A"],
                        "max_A": expected["max_A"],
                        "mean_A": expected["mean_A"],
                        "cell_max_abs_diff_A": expected["cell_max_abs_diff_A"],
                        "nearest_dft_endpoint_label": nearest["endpoint_label"],
                        "nearest_structure_matcher_fit": bool(nearest["structure_matcher_fit"]),
                        "nearest_rms_A": nearest["rms_A"],
                        "nearest_max_A": nearest["max_A"],
                        "nearest_endpoint_is_expected": nearest["endpoint_label"] == endpoint_label,
                        "preserved_expected_endpoint": status == "preserved",
                        "preservation_status": status,
                        "mlip_energy_eV": float(result["energy_eV"]),
                        "mlip_max_force_eVA": float(result["max_force_eVA"]),
                        "mlip_converged": bool(result.get("converged", False)),
                        "include_vdw": bool(result.get("include_vdw", False)),
                        "model_name": result.get("model_name", ""),
                        "result_json": str(result_path),
                        "per_species_json": expected["per_species_json"],
                    }
                )

    if not rows:
        raise RuntimeError("No completed endpoint relaxations were found for the scheduled model/case combinations.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "endpoint_geometry_comparisons.csv", index=False)
    pd.DataFrame(missing_rows, columns=MISSING_FIELDS).to_csv(
        out_dir / "endpoint_missing_relaxations.csv", index=False
    )

    model_summary = (
        df.groupby(["model_label", "charge"], dropna=False)
        .agg(
            n_endpoint_evaluations=("case_label", "size"),
            n_converged=("mlip_converged", "sum"),
            n_preserved=("preserved_expected_endpoint", "sum"),
            n_matches_expected_cluster=("matches_expected_endpoint_cluster", "sum"),
            n_nearest_expected=("nearest_endpoint_is_expected", "sum"),
            mean_rms_A=("rms_A", "mean"),
            median_rms_A=("rms_A", "median"),
            max_rms_A=("rms_A", "max"),
            mean_max_A=("max_A", "mean"),
            max_max_A=("max_A", "max"),
            mean_nearest_rms_A=("nearest_rms_A", "mean"),
            max_nearest_rms_A=("nearest_rms_A", "max"),
        )
        .reset_index()
    )
    model_summary["fraction_preserved"] = (
        model_summary["n_preserved"] / model_summary["n_endpoint_evaluations"]
    )
    model_summary["fraction_matches_expected_cluster"] = (
        model_summary["n_matches_expected_cluster"] / model_summary["n_endpoint_evaluations"]
    )
    model_summary.to_csv(out_dir / "endpoint_model_summary.csv", index=False)

    case_summary = (
        df.groupby(["case_label", "model_label"], dropna=False)
        .agg(
            phase=("phase", "first"),
            defect=("defect", "first"),
            charge=("charge", "first"),
            atom_count=("atom_count", "first"),
            n_endpoints_evaluated=("endpoint_label", "size"),
            n_preserved=("preserved_expected_endpoint", "sum"),
            n_matches_expected_cluster=("matches_expected_endpoint_cluster", "sum"),
            n_nearest_expected=("nearest_endpoint_is_expected", "sum"),
            max_rms_A=("rms_A", "max"),
            max_max_A=("max_A", "max"),
            all_endpoints_preserved=("preserved_expected_endpoint", "all"),
            all_expected_clusters_matched=("matches_expected_endpoint_cluster", "all"),
        )
        .reset_index()
    )
    case_summary.to_csv(out_dir / "endpoint_case_summary.csv", index=False)

    attention = df[
        (~df["preserved_expected_endpoint"])
        | (~df["mlip_converged"])
        | (~df["matches_expected_endpoint_cluster"])
    ].copy()
    attention.to_csv(out_dir / "endpoint_cases_needing_attention.csv", index=False)

    print(f"Wrote endpoint comparison outputs to {out_dir}")
    display_cols = [
        "model_label",
        "charge",
        "n_endpoint_evaluations",
        "n_preserved",
        "n_matches_expected_cluster",
        "mean_rms_A",
        "max_rms_A",
    ]
    printable = model_summary[display_cols].copy()
    for col in ["mean_rms_A", "max_rms_A"]:
        printable[col] = printable[col].map(lambda x: fmt_float(x, 4))
    print(printable.to_string(index=False))
    if missing_rows:
        print(f"Missing scheduled relaxations: {len(missing_rows)}; see endpoint_missing_relaxations.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MLIP-relaxed NEB endpoint structures to the original DFT endpoint structures. "
            "This tests endpoint preservation, not SnB ground-cluster selection."
        )
    )
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--case", help="Optional case label filter")
    parser.add_argument(
        "--models",
        nargs="+",
        default=SCHEDULED_MODELS,
        help="Model labels to include. Charge-specific scheduling is still applied.",
    )
    args = parser.parse_args()

    compare(args.analysis_name, args.models, args.case)


if __name__ == "__main__":
    main()
