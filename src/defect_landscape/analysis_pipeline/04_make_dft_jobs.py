from pathlib import Path
import shutil
import pandas as pd



ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

ANALYSIS_DIR = ROOT / "runs" / "analysis"
SELECTED_MANIFEST = ANALYSIS_DIR / "selected_for_dft_manifest.csv"

DFT_ROOT = ROOT / "runs" / "dft_validate"
TEMPLATE_DIR = ROOT / "data" / "vasp_relax_template"

TEMPLATE_FILES = ["INCAR", "KPOINTS", "POTCAR", "submit.sh"]



df = pd.read_csv(SELECTED_MANIFEST)

dft_rows = []

for i, row in df.iterrows():
    model_label = row["model_label"]
    cluster_id = int(row["cluster_id"])

    dft_id = f"{model_label}_cluster_{cluster_id:03d}"
    dft_dir = DFT_ROOT / dft_id
    dft_dir.mkdir(parents=True, exist_ok=True)

    # Copy the selected MLIP-relaxed structure as the DFT starting POSCAR.
    shutil.copy(row["selected_poscar"], dft_dir / "POSCAR")

    # Copy VASP template files if they exist.
    for filename in TEMPLATE_FILES:
        src = TEMPLATE_DIR / filename
        if src.exists():
            shutil.copy(src, dft_dir / filename)

    dft_rows.append(
        {
            "dft_id": dft_id,
            "model_label": model_label,
            "cluster_id": cluster_id,
            "best_candidate_id": row["best_candidate_id"],
            "best_mlip_dE_eV": row["best_mlip_dE_eV"],
            "dft_dir": str(dft_dir),
        }
    )

manifest = pd.DataFrame(dft_rows)
manifest.to_csv(ANALYSIS_DIR / "dft_validation_manifest.csv", index=False)

print(f"\nWrote DFT validation folders to:\n{DFT_ROOT}")
print(f"\nWrote DFT manifest:\n{ANALYSIS_DIR / 'dft_validation_manifest.csv'}")

missing = [f for f in TEMPLATE_FILES if not (TEMPLATE_DIR / f).exists()]
if missing:
    print("\nWARNING: These template files were missing and were not copied:")
    for f in missing:
        print(f"  {TEMPLATE_DIR / f}")