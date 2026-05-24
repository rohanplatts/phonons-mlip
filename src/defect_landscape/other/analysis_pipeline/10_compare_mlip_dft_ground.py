from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from ase.io import read
from ase.geometry import find_mic


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")
ANALYSIS_DIR = ROOT / "runs" / "analysis"

BASE_CSV = ANALYSIS_DIR / "base_mace_mlip_vs_dft.csv"
FT_CSV = ANALYSIS_DIR / "finetuned_mace_mlip_vs_dft.csv"
SELECTED_MANIFEST = ANALYSIS_DIR / "selected_for_dft_manifest.csv"

OUT_TXT = ANALYSIS_DIR / "human_read_union_reference_structure_distances.txt"
OUT_CSV = ANALYSIS_DIR / "union_reference_structure_distances.csv"

# Practical flags only. The actual statistics are what matter.
SAME_RMS_THRESHOLD_A = 0.05
SAME_MAX_THRESHOLD_A = 0.20


def hungarian_min(cost: np.ndarray) -> np.ndarray:
    """
    Solve square assignment problem.

    Returns assign[i] = j, meaning row i is matched to column j.
    """
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")

    n = int(a.shape[0])

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = np.empty(n, dtype=int)

    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1

    return assign


def load_rankings():
    """
    Load the DFT-relaxed union of model-selected minima.

    These are the low-energy unique MLIP minima that were selected,
    DFT-relaxed, and ranked in 07_compare_rankings.py.
    """
    if not BASE_CSV.exists():
        raise FileNotFoundError(f"Missing {BASE_CSV}. Run 07_compare_rankings.py first.")
    if not FT_CSV.exists():
        raise FileNotFoundError(f"Missing {FT_CSV}. Run 07_compare_rankings.py first.")
    if not SELECTED_MANIFEST.exists():
        raise FileNotFoundError(f"Missing {SELECTED_MANIFEST}. Run 03_cluster_select.py first.")

    base = pd.read_csv(BASE_CSV)
    ft = pd.read_csv(FT_CSV)
    selected = pd.read_csv(SELECTED_MANIFEST)

    df = pd.concat([base, ft], ignore_index=True)

    keep = [
        "model_label",
        "cluster_id",
        "representative_contcar",
        "selected_poscar",
    ]

    for col in keep:
        if col not in selected.columns:
            raise RuntimeError(f"{SELECTED_MANIFEST} is missing column: {col}")

    df = df.merge(
        selected[keep],
        on=["model_label", "cluster_id"],
        how="left",
    )

    df["label"] = [
        f"{m}_cluster_{int(c):03d}"
        for m, c in zip(df["model_label"], df["cluster_id"])
    ]

    return df


def final_dft_structure(row):
    """
    Final DFT-relaxed structure.
    """
    dft_dir = Path(row["dft_dir"])
    contcar = dft_dir / "CONTCAR"

    if contcar.exists() and contcar.stat().st_size > 0:
        return contcar

    raise FileNotFoundError(f"Missing final DFT CONTCAR: {contcar}")


def mlip_relaxed_structure(row):
    """
    MLIP-relaxed structure before DFT.

    Important:
    Do not use dft_validate/POSCAR by default, because POSCAR may have
    been overwritten by CONTCAR during VASP restarts.

    Prefer:
    1. representative_contcar from selected_for_dft_manifest.csv
    2. selected_poscar from selected_for_dft_manifest.csv
    """
    candidates = [
        row.get("representative_contcar", ""),
        row.get("selected_poscar", ""),
    ]

    for c in candidates:
        if isinstance(c, str) and c.strip():
            f = Path(c)
            if f.exists() and f.stat().st_size > 0:
                return f

    # Emergency fallback: oldest pre-restart POSCAR before restart.
    dft_dir = Path(row["dft_dir"])
    pre_poscars = sorted(dft_dir.glob("pre_restart_*/POSCAR.before_restart"))

    for f in pre_poscars:
        if f.exists() and f.stat().st_size > 0:
            return f

    raise FileNotFoundError(
        f"Could not recover MLIP-relaxed input structure for {row['label']}"
    )


def compare_structures(file_a, file_b):
    """
    Species-resolved Hungarian minimum-image comparison.

    For each species:
      - build cost matrix of MIC distances between atoms in A and B
      - assign atoms using Hungarian minimisation
      - compute total/mean/RMS/max displacement statistics
    """
    a = read(file_a)
    b = read(file_b)

    sym_a = np.array(a.get_chemical_symbols())
    sym_b = np.array(b.get_chemical_symbols())

    if sorted(sym_a) != sorted(sym_b):
        raise ValueError(f"Different compositions:\n{file_a}\n{file_b}")

    if len(a) != len(b):
        raise ValueError(f"Different atom counts:\n{file_a}\n{file_b}")

    pos_a = a.get_positions()
    pos_b = b.get_positions()

    cell = a.cell
    pbc = a.pbc

    all_dist = np.zeros(len(a), dtype=float)
    per_species = {}

    for el in sorted(set(sym_a)):
        idx_a = np.where(sym_a == el)[0]
        idx_b = np.where(sym_b == el)[0]

        xa = pos_a[idx_a]
        xb = pos_b[idx_b]

        cost = np.zeros((len(idx_a), len(idx_b)), dtype=float)

        for i in range(len(idx_a)):
            disp = xb - xa[i]
            disp_mic, _ = find_mic(disp, cell=cell, pbc=pbc)
            cost[i, :] = np.linalg.norm(disp_mic, axis=1)

        assign = hungarian_min(cost)
        species_dist = cost[np.arange(len(idx_a)), assign]

        all_dist[idx_a] = species_dist

        per_species[el] = {
            "n": int(len(idx_a)),
            "total_A": float(np.sum(species_dist)),
            "mean_A": float(np.mean(species_dist)),
            "rms_A": float(np.sqrt(np.mean(species_dist ** 2))),
            "max_A": float(np.max(species_dist)),
        }

    return {
        "n_atoms": int(len(a)),
        "total_A": float(np.sum(all_dist)),
        "mean_A": float(np.mean(all_dist)),
        "rms_A": float(np.sqrt(np.mean(all_dist ** 2))),
        "max_A": float(np.max(all_dist)),
        "per_species": per_species,
    }


def add_stats(prefix, stats, out):
    out[f"{prefix}_total_A"] = stats["total_A"]
    out[f"{prefix}_mean_A"] = stats["mean_A"]
    out[f"{prefix}_rms_A"] = stats["rms_A"]
    out[f"{prefix}_max_A"] = stats["max_A"]


def fmt(x, ndp=6):
    if x is None:
        return ""
    try:
        if not np.isfinite(x):
            return ""
        return f"{x:.{ndp}f}"
    except Exception:
        return str(x)


def main():
    df = load_rankings()

    # The best available DFT reference is the lowest DFT energy in the
    # union of all low-energy unique MLIP-selected structures.
    reference = df.loc[df["dft_energy_eV"].idxmin()].copy()
    reference_dft_file = final_dft_structure(reference)

    rows = []

    for _, row in df.iterrows():
        candidate_dft_file = final_dft_structure(row)
        candidate_mlip_file = mlip_relaxed_structure(row)

        # 1. Did the final DFT-relaxed structure match the best available DFT reference?
        dft_to_ref = compare_structures(candidate_dft_file, reference_dft_file)

        # 2. Was the MLIP-relaxed input already close to the best available DFT reference?
        mlip_to_ref = compare_structures(candidate_mlip_file, reference_dft_file)

        # 3. How much did DFT move the MLIP-relaxed structure?
        mlip_to_own_dft = compare_structures(candidate_mlip_file, candidate_dft_file)

        out = {
            "label": row["label"],
            "model_label": row["model_label"],
            "cluster_id": int(row["cluster_id"]),
            "best_candidate_id": row["best_candidate_id"],
            "best_mlip_dE_eV": float(row["best_mlip_dE_eV"]),
            "dft_energy_eV": float(row["dft_energy_eV"]),
            "dft_dE_vs_model_set_eV": float(row["dft_dE_eV"]),
            "dft_dE_vs_union_reference_eV": float(row["dft_energy_eV"] - reference["dft_energy_eV"]),
            "is_union_dft_reference": bool(row["label"] == reference["label"]),
            "is_model_mlip_ground": bool(
                row["best_mlip_dE_eV"]
                == df[df["model_label"] == row["model_label"]]["best_mlip_dE_eV"].min()
            ),
            "is_model_dft_ground": bool(
                row["dft_energy_eV"]
                == df[df["model_label"] == row["model_label"]]["dft_energy_eV"].min()
            ),
            "dft_dir": row["dft_dir"],
            "mlip_structure_file": str(candidate_mlip_file),
            "dft_structure_file": str(candidate_dft_file),
        }

        add_stats("dft_to_union_ref", dft_to_ref, out)
        add_stats("mlip_to_union_ref", mlip_to_ref, out)
        add_stats("mlip_to_own_dft", mlip_to_own_dft, out)

        out["dft_same_as_union_ref_by_threshold"] = bool(
            dft_to_ref["rms_A"] <= SAME_RMS_THRESHOLD_A
            and dft_to_ref["max_A"] <= SAME_MAX_THRESHOLD_A
        )

        out["mlip_input_close_to_union_ref_by_threshold"] = bool(
            mlip_to_ref["rms_A"] <= SAME_RMS_THRESHOLD_A
            and mlip_to_ref["max_A"] <= SAME_MAX_THRESHOLD_A
        )

        out["dft_to_union_ref_per_species_json"] = json.dumps(dft_to_ref["per_species"])
        out["mlip_to_union_ref_per_species_json"] = json.dumps(mlip_to_ref["per_species"])
        out["mlip_to_own_dft_per_species_json"] = json.dumps(mlip_to_own_dft["per_species"])

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("dft_energy_eV").reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False)

    # Human-readable report.
    lines = []
    lines.append("Union-reference structure validation")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Question")
    lines.append("-" * 80)
    lines.append("Does the MLIP-assisted workflow produce the correct defect structure?")
    lines.append("")
    lines.append("Reference used here")
    lines.append("-" * 80)
    lines.append("The reference is the lowest-energy DFT-relaxed structure among the union of")
    lines.append("low-energy unique MLIP-selected minima from base MACE and fine-tuned MACE.")
    lines.append("")
    lines.append("This is not the full direct-DFT ShakeNBreak ground state unless every")
    lines.append("ShakeNBreak distortion has also been directly DFT-relaxed.")
    lines.append("")
    lines.append(f"Union DFT reference: {reference['label']}")
    lines.append(f"  candidate: {reference['best_candidate_id']}")
    lines.append(f"  E_DFT:     {reference['dft_energy_eV']:.8f} eV")
    lines.append(f"  structure: {reference_dft_file}")
    lines.append("")

    lines.append("Distance definitions")
    lines.append("-" * 80)
    lines.append("Distances use species-resolved Hungarian assignment with minimum-image")
    lines.append("periodic distances.")
    lines.append("")
    lines.append("dft_to_union_ref:")
    lines.append("  final DFT-relaxed candidate structure compared to the union DFT reference")
    lines.append("")
    lines.append("mlip_to_union_ref:")
    lines.append("  MLIP-relaxed pre-DFT candidate structure compared to the union DFT reference")
    lines.append("")
    lines.append("mlip_to_own_dft:")
    lines.append("  MLIP-relaxed pre-DFT candidate structure compared to its own final DFT")
    lines.append("  relaxed version")
    lines.append("")

    lines.append("Main table")
    lines.append("-" * 80)

    show_cols = [
        "label",
        "best_mlip_dE_eV",
        "dft_dE_vs_union_reference_eV",
        "dft_to_union_ref_total_A",
        "dft_to_union_ref_rms_A",
        "dft_to_union_ref_max_A",
        "mlip_to_union_ref_total_A",
        "mlip_to_union_ref_rms_A",
        "mlip_to_union_ref_max_A",
        "mlip_to_own_dft_total_A",
        "mlip_to_own_dft_rms_A",
        "mlip_to_own_dft_max_A",
        "is_model_mlip_ground",
        "is_model_dft_ground",
        "is_union_dft_reference",
    ]

    show = out_df[show_cols].copy()

    float_cols = [
        c for c in show.columns
        if c.endswith("_A") or c.endswith("_eV")
    ]

    for c in float_cols:
        show[c] = show[c].map(lambda x: fmt(x, 6))

    lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("Interpretation guide")
    lines.append("-" * 80)
    lines.append("Small dft_to_union_ref distance means the final DFT structure is the same")
    lines.append("or very close to the best available DFT reference structure.")
    lines.append("")
    lines.append("Small mlip_to_union_ref distance means the MLIP-relaxed geometry itself was")
    lines.append("already close to the best available DFT reference before DFT cleaned it up.")
    lines.append("")
    lines.append("Small mlip_to_own_dft distance means DFT did not have to move the MLIP")
    lines.append("structure much during relaxation.")
    lines.append("")
    lines.append(f"Practical near-duplicate flag: RMS <= {SAME_RMS_THRESHOLD_A} Å and max <= {SAME_MAX_THRESHOLD_A} Å")
    lines.append("This is only a heuristic. The raw distances are the important result.")
    lines.append("")

    lines.append("Near-duplicate flags")
    lines.append("-" * 80)

    flags = out_df[
        [
            "label",
            "dft_same_as_union_ref_by_threshold",
            "mlip_input_close_to_union_ref_by_threshold",
        ]
    ].copy()

    lines.append(flags.to_string(index=False))
    lines.append("")

    lines.append("Files written")
    lines.append("-" * 80)
    lines.append(str(OUT_TXT))
    lines.append(str(OUT_CSV))

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote:\n  {OUT_TXT}\n  {OUT_CSV}")
    print("")
    print(f"Union DFT reference: {reference['label']}")
    print(f"E_DFT = {reference['dft_energy_eV']:.8f} eV")
    print("")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()