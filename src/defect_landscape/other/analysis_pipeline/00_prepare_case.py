from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_common import (
    analysis_root,
    candidate_manifest_path,
    case_manifest_path,
    clear_analysis,
    copy_structure,
    discover_reference_dirs,
    ensure_analysis_dirs,
    load_reference_metadata,
    safe_label,
    upsert_rows,
    write_csv_rows,
)


CANDIDATE_FIELDS = [
    "case_label",
    "variant_label",
    "input_poscar",
    "staged_variant_poscar",
    "dft_contcar",
    "staged_dft_contcar",
    "dft_energy_eV",
    "dft_reference_dir",
    "dft_reference_section",
    "source_dft_references_dir",
]

CASE_FIELDS = [
    "case_label",
    "input_poscar",
    "staged_case_input_poscar",
    "dft_references_dir",
    "n_variants",
]


def stage_case(analysis_name: str, case_label: str, input_poscar: Path, dft_references_dir: Path) -> None:
    ensure_analysis_dirs(analysis_name)
    write_bunya_notes(analysis_name)
    root = analysis_root(analysis_name)
    case_safe = safe_label(case_label)

    staged_case_input = root / "case_inputs" / case_safe / "POSCAR"
    copy_structure(input_poscar, staged_case_input)

    candidate_rows = []
    ref_dirs = discover_reference_dirs(dft_references_dir)
    if not ref_dirs:
        raise RuntimeError(f"No DFT reference folders found under {dft_references_dir}")

    for ref_dir in ref_dirs:
        meta = load_reference_metadata(ref_dir)
        variant_label = safe_label(meta.get("label", ref_dir.name))
        evidence_dir = Path(meta["evidence_dir"])
        variant_poscar = evidence_dir / "POSCAR"
        dft_contcar = Path(meta["structure_path"])
        dft_energy = meta.get("energy_eV", "")

        if not variant_poscar.exists():
            raise FileNotFoundError(f"Missing SnB variant POSCAR for {case_label}/{variant_label}: {variant_poscar}")
        if not dft_contcar.exists():
            raise FileNotFoundError(f"Missing DFT reference CONTCAR for {case_label}/{variant_label}: {dft_contcar}")

        staged_variant = root / "snb_variant_inputs" / case_safe / variant_label / "POSCAR"
        staged_dft = root / "dft_references" / case_safe / variant_label / "CONTCAR"
        copy_structure(variant_poscar, staged_variant)
        copy_structure(dft_contcar, staged_dft)

        section = ref_dir.parent.name
        meta_out = root / "dft_references" / case_safe / variant_label / "reference_metadata.json"
        meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

        candidate_rows.append(
            {
                "case_label": case_label,
                "variant_label": variant_label,
                "input_poscar": str(variant_poscar),
                "staged_variant_poscar": str(staged_variant),
                "dft_contcar": str(dft_contcar),
                "staged_dft_contcar": str(staged_dft),
                "dft_energy_eV": dft_energy,
                "dft_reference_dir": str(evidence_dir),
                "dft_reference_section": section,
                "source_dft_references_dir": str(dft_references_dir),
            }
        )

    upsert_rows(
        candidate_manifest_path(analysis_name),
        candidate_rows,
        CANDIDATE_FIELDS,
        key_fields=["case_label", "variant_label"],
    )

    upsert_rows(
        case_manifest_path(analysis_name),
        [
            {
                "case_label": case_label,
                "input_poscar": str(input_poscar),
                "staged_case_input_poscar": str(staged_case_input),
                "dft_references_dir": str(dft_references_dir),
                "n_variants": len(candidate_rows),
            }
        ],
        CASE_FIELDS,
        key_fields=["case_label"],
    )

    print(f"Prepared {case_label}: {len(candidate_rows)} SnB variants")
    print(f"  input: {input_poscar}")
    print(f"  dft:   {dft_references_dir}")


def write_empty_manifests(analysis_name: str) -> None:
    ensure_analysis_dirs(analysis_name)
    write_csv_rows(candidate_manifest_path(analysis_name), [], CANDIDATE_FIELDS)
    write_csv_rows(case_manifest_path(analysis_name), [], CASE_FIELDS)
    write_bunya_notes(analysis_name)


def write_bunya_notes(analysis_name: str) -> None:
    root = analysis_root(analysis_name)
    notes = f"""# Bunya Transfer Notes

Analysis: `{analysis_name}`

This workflow should not submit new DFT jobs. Existing DFT-relaxed SnB references
are staged locally as lightweight CONTCAR copies and metadata.

If running MLIP relaxations on Bunya, transfer:

```text
{root}/snb_variant_inputs/
{root}/analysis/candidate_manifest.csv
/home/rnpla/projects/mlip_phonons/src/defect_landscape/analysis_pipeline/
```

Transfer back:

```text
{root}/mlip_relaxed/
```

Then run locally:

```bash
python 02_compare_to_existing_dft.py --analysis-name {analysis_name}
python 03_write_carla_report.py --analysis-name {analysis_name}
```

Do not transfer or duplicate POTCAR files for this workflow.
"""
    (root / "bunya_transfer_notes.md").write_text(notes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one MLIP-vs-existing-DFT SnB validation case. "
            "This stages only POSCAR/CONTCAR metadata under runs/<analysis-name>."
        )
    )
    parser.add_argument("--analysis-name", required=True, help="Named folder under src/defect_landscape/runs")
    parser.add_argument("--case-label", help="Readable case id, e.g. VBr_q0_end_Br4c_test1")
    parser.add_argument("--input-poscar", type=Path, help="Original input POSCAR for this defect/site case")
    parser.add_argument("--dft-references-dir", type=Path, help="Directory containing minimum/ and alternatives/")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate runs/<analysis-name>")
    parser.add_argument("--init-empty", action="store_true", help="Create empty manifests without adding a case")

    args = parser.parse_args()

    if args.reset:
        clear_analysis(args.analysis_name)
        print(f"Reset {analysis_root(args.analysis_name)}")

    if args.init_empty:
        write_empty_manifests(args.analysis_name)
        print(f"Initialized {analysis_root(args.analysis_name)}")
        return

    missing = [
        name
        for name, value in [
            ("--case-label", args.case_label),
            ("--input-poscar", args.input_poscar),
            ("--dft-references-dir", args.dft_references_dir),
        ]
        if value is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments for case preparation: {', '.join(missing)}")

    stage_case(
        analysis_name=args.analysis_name,
        case_label=args.case_label,
        input_poscar=args.input_poscar,
        dft_references_dir=args.dft_references_dir,
    )


if __name__ == "__main__":
    main()
