from pathlib import Path
import json
import shutil

import pandas as pd
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

MLIP_ROOT = ROOT / "runs" / "mlip_relaxed"
ANALYSIS_DIR = ROOT / "runs" / "analysis"
SELECTED_DIR = ROOT / "runs" / "selected_for_dft"

ENERGY_WINDOW_EV = 0.50
MAX_CLUSTERS_PER_MODEL = 10

MODELS = ["base_mace", "finetuned_mace"]


# -------------------------
# Structure matcher
# -------------------------

matcher = StructureMatcher(
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    primitive_cell=False,
    scale=False,
    attempt_supercell=False,
)


def read_result_rows(model_label):
    rows = []

    for result_file in sorted((MLIP_ROOT / model_label).rglob("result.json")):
        with open(result_file) as f:
            result = json.load(f)

        rows.append(result)

    if not rows:
        raise RuntimeError(f"No result.json files found for {model_label}")

    df = pd.DataFrame(rows)
    df["dE_mlip_eV"] = df["energy_eV"] - df["energy_eV"].min()
    df = df.sort_values("dE_mlip_eV").reset_index(drop=True)

    return df


def cluster_model(df):
    clusters = []

    for _, row in df.iterrows():
        atoms = read(row["relaxed_contcar"])
        struct = AseAtomsAdaptor.get_structure(atoms)

        assigned = False

        for cluster in clusters:
            if matcher.fit(struct, cluster["representative_structure"]):
                cluster["members"].append(row.to_dict())
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "cluster_id": len(clusters),
                    "representative_structure": struct,
                    "members": [row.to_dict()],
                }
            )

    cluster_rows = []

    for cluster in clusters:
        best = min(cluster["members"], key=lambda x: x["dE_mlip_eV"])

        cluster_rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "n_members": len(cluster["members"]),
                "best_candidate_id": best["candidate_id"],
                "best_mlip_energy_eV": best["energy_eV"],
                "best_mlip_dE_eV": best["dE_mlip_eV"],
                "representative_contcar": best["relaxed_contcar"],
            }
        )

    return pd.DataFrame(cluster_rows).sort_values("best_mlip_dE_eV").reset_index(drop=True)


# -------------------------
# Main
# -------------------------

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_DIR.mkdir(parents=True, exist_ok=True)

all_selected_rows = []

for model_label in MODELS:
    print(f"\n=== Clustering {model_label} ===")

    df = read_result_rows(model_label)
    df.to_csv(ANALYSIS_DIR / f"{model_label}_all_mlip_relaxations.csv", index=False)

    clusters = cluster_model(df)
    clusters.to_csv(ANALYSIS_DIR / f"{model_label}_clusters.csv", index=False)

    # Select unique low-energy clusters.
    selected = clusters[
        clusters["best_mlip_dE_eV"] <= ENERGY_WINDOW_EV
    ].head(MAX_CLUSTERS_PER_MODEL)

    selected.to_csv(ANALYSIS_DIR / f"{model_label}_selected_clusters.csv", index=False)

    for _, row in selected.iterrows():
        cluster_id = int(row["cluster_id"])

        out_dir = SELECTED_DIR / model_label / f"cluster_{cluster_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # This POSCAR is the MLIP-relaxed representative structure.
        # It is what we send to DFT for validation.
        shutil.copy(row["representative_contcar"], out_dir / "POSCAR")

        all_selected_rows.append(
            {
                "model_label": model_label,
                "cluster_id": cluster_id,
                "best_candidate_id": row["best_candidate_id"],
                "best_mlip_dE_eV": row["best_mlip_dE_eV"],
                "representative_contcar": row["representative_contcar"],
                "selected_poscar": str(out_dir / "POSCAR"),
            }
        )

    print(selected[["cluster_id", "n_members", "best_mlip_dE_eV", "best_candidate_id"]])

pd.DataFrame(all_selected_rows).to_csv(
    ANALYSIS_DIR / "selected_for_dft_manifest.csv",
    index=False,
)

print(f"\nWrote selected structures to:\n{SELECTED_DIR}")
print(f"\nWrote manifest:\n{ANALYSIS_DIR / 'selected_for_dft_manifest.csv'}")