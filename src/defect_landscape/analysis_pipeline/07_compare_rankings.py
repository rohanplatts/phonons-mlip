from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from pymatgen.io.vasp.outputs import Vasprun



ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

ANALYSIS_DIR = ROOT / "runs" / "analysis"
DFT_MANIFEST = ANALYSIS_DIR / "dft_validation_manifest.csv"


# -------------------------
# Energy parsing
# -------------------------

def parse_dft_energy(dft_dir):
    dft_dir = Path(dft_dir)

    vasprun = dft_dir / "vasprun.xml"
    oszicar = dft_dir / "OSZICAR"

    if vasprun.exists():
        vr = Vasprun(
            vasprun,
            parse_dos=False,
            parse_eigen=False,
            parse_potcar_file=False,
        )
        return float(vr.final_energy)

    if oszicar.exists():
        text = oszicar.read_text(errors="ignore")

        # Parse final E0 from OSZICAR lines like:
        # 1 F= ... E0= -123.456 d E = ...
        matches = re.findall(r"E0=\s*([-+0-9.Ee]+)", text)
        if matches:
            return float(matches[-1])

    return np.nan


# -------------------------
# Compare ranking
# -------------------------

df = pd.read_csv(DFT_MANIFEST)

df["dft_energy_eV"] = [parse_dft_energy(d) for d in df["dft_dir"]]
df = df.dropna(subset=["dft_energy_eV"]).copy()

if df.empty:
    raise RuntimeError("No completed DFT energies found. Need vasprun.xml or OSZICAR in DFT folders.")

df.to_csv(ANALYSIS_DIR / "dft_energies_parsed.csv", index=False)

all_metrics = {}

for model_label, sub in df.groupby("model_label"):
    sub = sub.copy()

    # Relative DFT energies within this model's selected candidate set.
    # Same composition, same charge, same cell, so this is valid.
    sub["dft_dE_eV"] = sub["dft_energy_eV"] - sub["dft_energy_eV"].min()

    sub = sub.sort_values("best_mlip_dE_eV").reset_index(drop=True)

    n = len(sub)

    if n >= 2:
        rho, _ = spearmanr(sub["best_mlip_dE_eV"], sub["dft_dE_eV"])
        tau, _ = kendalltau(sub["best_mlip_dE_eV"], sub["dft_dE_eV"])
    else:
        rho = np.nan
        tau = np.nan

    mae = float(np.mean(np.abs(sub["best_mlip_dE_eV"] - sub["dft_dE_eV"])))

    mlip_ground_cluster = int(sub.loc[sub["best_mlip_dE_eV"].idxmin(), "cluster_id"])
    dft_ground_cluster = int(sub.loc[sub["dft_dE_eV"].idxmin(), "cluster_id"])

    top1_correct = mlip_ground_cluster == dft_ground_cluster

    mlip_sorted = sub.sort_values("best_mlip_dE_eV")
    top3_contains = dft_ground_cluster in set(mlip_sorted.head(3)["cluster_id"])
    top5_contains = dft_ground_cluster in set(mlip_sorted.head(5)["cluster_id"])

    metrics = {
        "model_label": model_label,
        "n_dft_validated": int(n),
        "relative_energy_mae_eV": mae,
        "spearman_rho": None if np.isnan(rho) else float(rho),
        "kendall_tau": None if np.isnan(tau) else float(tau),
        "mlip_ground_cluster": mlip_ground_cluster,
        "dft_ground_cluster": dft_ground_cluster,
        "top1_correct": bool(top1_correct),
        "top3_contains_dft_ground": bool(top3_contains),
        "top5_contains_dft_ground": bool(top5_contains),
    }

    all_metrics[model_label] = metrics

    sub.to_csv(ANALYSIS_DIR / f"{model_label}_mlip_vs_dft.csv", index=False)

    # Plot MLIP relative energy against DFT relative energy.
    plt.figure()
    plt.scatter(sub["best_mlip_dE_eV"], sub["dft_dE_eV"])

    max_e = max(sub["best_mlip_dE_eV"].max(), sub["dft_dE_eV"].max())
    max_e = max(max_e, 0.1)

    plt.plot([0, max_e], [0, max_e], linestyle="--")
    plt.xlabel("MLIP relative energy / eV")
    plt.ylabel("DFT relative energy / eV")
    plt.title(model_label)
    plt.tight_layout()
    #plt.savefig(ANALYSIS_DIR / f"{model_label}_mlip_vs_dft.png", dpi=300)
    plt.close()


# -------------------------
# Combined MLIP-vs-DFT plot
# -------------------------

combined = []

for model_label in ["base_mace", "finetuned_mace"]:
    f = ANALYSIS_DIR / f"{model_label}_mlip_vs_dft.csv"
    if f.exists():
        tmp = pd.read_csv(f)
        combined.append(tmp)

combined = pd.concat(combined, ignore_index=True)

plt.figure()

colors = {
    "base_mace": "tab:blue",
    "finetuned_mace": "tab:orange",
}

for model_label, sub in combined.groupby("model_label"):
    plt.scatter(
        sub["best_mlip_dE_eV"],
        sub["dft_dE_eV"],
        label=model_label,
        color=colors[model_label],
    )

max_e = max(
    combined["best_mlip_dE_eV"].max(),
    combined["dft_dE_eV"].max(),
)
max_e = max(max_e, 0.1)

plt.plot([0, max_e], [0, max_e], linestyle="--", color="black", label="perfect agreement")

plt.xlabel("MLIP relative energy / eV")
plt.ylabel("DFT relative energy / eV")
plt.title("MLIP vs DFT relative energies")
plt.legend()
plt.tight_layout()

plt.savefig(ANALYSIS_DIR / "combined_mlip_vs_dft.png", dpi=300)
plt.close()


with open(ANALYSIS_DIR / "ranking_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\nRanking metrics:")
print(json.dumps(all_metrics, indent=2))

print(f"\nWrote comparison outputs to:\n{ANALYSIS_DIR}")