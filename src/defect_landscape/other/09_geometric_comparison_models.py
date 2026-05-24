from pathlib import Path
import math
import json

import numpy as np
import pandas as pd

from ase.io import read
from ase.geometry import find_mic
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")
ANALYSIS_DIR = ROOT / "runs" / "analysis"

BASE_CSV = ANALYSIS_DIR / "base_mace_mlip_vs_dft.csv"
FT_CSV = ANALYSIS_DIR / "finetuned_mace_mlip_vs_dft.csv"

OUT_TXT = ANALYSIS_DIR / "human_read_dft_structure_matching.txt"
OUT_CSV = ANALYSIS_DIR / "dft_hungarian_structure_distances.csv"


# Heuristic thresholds for saying two relaxed structures are probably the same basin.
# These are deliberately configurable, not laws of nature.
SAME_RMS_THRESHOLD_A = 0.05
SAME_MAX_THRESHOLD_A = 0.20


matcher = StructureMatcher(
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    primitive_cell=False,
    scale=False,
    attempt_supercell=False,
)


def hungarian_min(cost: np.ndarray) -> np.ndarray:
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")

    n = int(a.shape[0])

    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
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


def load_model_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 07_compare_rankings.py first.")

    df = pd.read_csv(path)

    required = [
        "model_label",
        "cluster_id",
        "best_candidate_id",
        "dft_dir",
        "dft_energy_eV",
        "dft_dE_eV",
        "best_mlip_dE_eV",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"{path} is missing column: {col}")

    df = df.copy()
    df["label"] = [
        f"{m}_cluster_{int(c):03d}"
        for m, c in zip(df["model_label"], df["cluster_id"])
    ]

    return df


def relaxed_file(dft_dir):
    dft_dir = Path(dft_dir)

    for name in ["CONTCAR", "POSCAR"]:
        f = dft_dir / name
        if f.exists() and f.stat().st_size > 0:
            return f

    raise FileNotFoundError(f"No CONTCAR or POSCAR found in {dft_dir}")


def compare_atoms_hungarian_mic(file_a, file_b):
    """
    Compare two structures by species-resolved Hungarian assignment.

    For each species:
      1. build pairwise minimum-image distances between atoms in A and atoms in B
      2. solve the assignment problem
      3. report the distribution of assigned MIC distances

    This removes atom-indexing ambiguity while respecting periodic boundaries.
    """
    a = read(file_a)
    b = read(file_b)

    sym_a = np.array(a.get_chemical_symbols())
    sym_b = np.array(b.get_chemical_symbols())

    if sorted(sym_a) != sorted(sym_b):
        raise ValueError("Structures have different compositions.")

    pos_a = a.get_positions()
    pos_b = b.get_positions()

    cell = a.cell
    pbc = a.pbc

    all_distances = np.zeros(len(a), dtype=float)
    assignment_global = np.full(len(a), -1, dtype=int)
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

        all_distances[idx_a] = species_dist
        assignment_global[idx_a] = idx_b[assign]

        per_species[el] = {
            "n": int(len(idx_a)),
            "total_A": float(np.sum(species_dist)),
            "mean_A": float(np.mean(species_dist)),
            "rms_A": float(np.sqrt(np.mean(species_dist ** 2))),
            "max_A": float(np.max(species_dist)),
        }

    cell_diff_A = float(np.max(np.abs(np.array(a.cell) - np.array(b.cell))))

    stats = {
        "n_atoms": int(len(a)),
        "total_distance_A": float(np.sum(all_distances)),
        "mean_distance_A": float(np.mean(all_distances)),
        "rms_distance_A": float(np.sqrt(np.mean(all_distances ** 2))),
        "max_distance_A": float(np.max(all_distances)),
        "cell_max_abs_diff_A": cell_diff_A,
        "per_species": per_species,
        "assignment_global": assignment_global.tolist(),
    }

    return stats


def structure_matcher_fit(file_a, file_b):
    try:
        s1 = Structure.from_file(file_a)
        s2 = Structure.from_file(file_b)
        return bool(matcher.fit(s1, s2))
    except Exception:
        return False


def compare_rows(row_a, row_b):
    file_a = relaxed_file(row_a["dft_dir"])
    file_b = relaxed_file(row_b["dft_dir"])

    stats = compare_atoms_hungarian_mic(file_a, file_b)
    sm_fit = structure_matcher_fit(file_a, file_b)

    near_duplicate = (
        stats["rms_distance_A"] <= SAME_RMS_THRESHOLD_A
        and stats["max_distance_A"] <= SAME_MAX_THRESHOLD_A
    )

    return {
        "label_a": row_a["label"],
        "label_b": row_b["label"],
        "model_a": row_a["model_label"],
        "model_b": row_b["model_label"],
        "cluster_a": int(row_a["cluster_id"]),
        "cluster_b": int(row_b["cluster_id"]),
        "candidate_a": row_a["best_candidate_id"],
        "candidate_b": row_b["best_candidate_id"],
        "dft_energy_a_eV": float(row_a["dft_energy_eV"]),
        "dft_energy_b_eV": float(row_b["dft_energy_eV"]),
        "abs_dft_energy_diff_eV": float(abs(row_a["dft_energy_eV"] - row_b["dft_energy_eV"])),
        "dft_dE_a_eV": float(row_a["dft_dE_eV"]),
        "dft_dE_b_eV": float(row_b["dft_dE_eV"]),
        "mlip_dE_a_eV": float(row_a["best_mlip_dE_eV"]),
        "mlip_dE_b_eV": float(row_b["best_mlip_dE_eV"]),
        "n_atoms": stats["n_atoms"],
        "total_distance_A": stats["total_distance_A"],
        "mean_distance_A": stats["mean_distance_A"],
        "rms_distance_A": stats["rms_distance_A"],
        "max_distance_A": stats["max_distance_A"],
        "cell_max_abs_diff_A": stats["cell_max_abs_diff_A"],
        "structure_matcher_fit": sm_fit,
        "near_duplicate_by_distance_threshold": bool(near_duplicate),
        "per_species_json": json.dumps(stats["per_species"]),
    }


def format_pair(row):
    return (
        f"{row['label_a']}  vs  {row['label_b']}\n"
        f"  DFT energy difference: {row['abs_dft_energy_diff_eV']:.6f} eV\n"
        f"  total MIC distance:    {row['total_distance_A']:.6f} Å\n"
        f"  mean MIC distance:     {row['mean_distance_A']:.6f} Å/atom\n"
        f"  RMS MIC distance:      {row['rms_distance_A']:.6f} Å/atom\n"
        f"  max MIC distance:      {row['max_distance_A']:.6f} Å\n"
        f"  StructureMatcher fit:  {row['structure_matcher_fit']}\n"
        f"  near-duplicate flag:   {row['near_duplicate_by_distance_threshold']}"
    )


def main():
    base = load_model_csv(BASE_CSV)
    ft = load_model_csv(FT_CSV)

    base_ground = base.loc[base["dft_dE_eV"].idxmin()]
    ft_ground = ft.loc[ft["dft_dE_eV"].idxmin()]

    rows = []

    # Cross-model pairwise comparisons.
    for _, rb in base.iterrows():
        for _, rf in ft.iterrows():
            rows.append(compare_rows(rb, rf))

    out = pd.DataFrame(rows)
    out = out.sort_values("rms_distance_A").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    ground_stats = compare_rows(base_ground, ft_ground)

    # Nearest fine-tuned match for each base candidate.
    nearest_ft_for_base = (
        out.sort_values("rms_distance_A")
        .groupby("label_a", as_index=False)
        .first()
        .sort_values("label_a")
    )

    # Nearest base match for each fine-tuned candidate.
    nearest_base_for_ft = (
        out.sort_values("rms_distance_A")
        .groupby("label_b", as_index=False)
        .first()
        .sort_values("label_b")
    )

    # Overall DFT minimum across the union of both candidate sets.
    union = pd.concat([base, ft], ignore_index=True)
    global_ground = union.loc[union["dft_energy_eV"].idxmin()]

    lines = []
    lines.append("DFT-relaxed minima matching using Hungarian minimum-image distances")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("-" * 78)
    lines.append("Test whether the base-MACE and fine-tuned-MACE DFT-relaxed ground-state")
    lines.append("candidates are actually the same physical defect minimum.")
    lines.append("")
    lines.append("Method")
    lines.append("-" * 78)
    lines.append("For every cross-model pair of DFT-relaxed structures:")
    lines.append("1. Read the final relaxed CONTCAR.")
    lines.append("2. For each species separately, build a pairwise cost matrix of")
    lines.append("   minimum-image distances between atoms in structure A and structure B.")
    lines.append("3. Use Hungarian minimisation to find the atom assignment with minimum")
    lines.append("   total distance.")
    lines.append("4. Report total, mean, RMS, and max assigned minimum-image distances.")
    lines.append("")
    lines.append(f"Near-duplicate heuristic used here:")
    lines.append(f"  RMS distance <= {SAME_RMS_THRESHOLD_A:.3f} Å and max distance <= {SAME_MAX_THRESHOLD_A:.3f} Å")
    lines.append("This is a practical flag, not a theorem.")
    lines.append("")

    lines.append("Global DFT-lowest candidate across the union of both model-selected sets")
    lines.append("-" * 78)
    lines.append(
        f"{global_ground['label']}: "
        f"E_DFT = {global_ground['dft_energy_eV']:.8f} eV, "
        f"candidate = {global_ground['best_candidate_id']}"
    )
    lines.append("")

    lines.append("Model-specific DFT ground-state candidates")
    lines.append("-" * 78)
    lines.append(
        f"base_mace:      {base_ground['label']}, "
        f"E_DFT = {base_ground['dft_energy_eV']:.8f} eV, "
        f"dE_DFT = {base_ground['dft_dE_eV']:.6f} eV"
    )
    lines.append(f"  candidate: {base_ground['best_candidate_id']}")
    lines.append(
        f"finetuned_mace: {ft_ground['label']}, "
        f"E_DFT = {ft_ground['dft_energy_eV']:.8f} eV, "
        f"dE_DFT = {ft_ground['dft_dE_eV']:.6f} eV"
    )
    lines.append(f"  candidate: {ft_ground['best_candidate_id']}")
    lines.append("")

    lines.append("Ground-state cross-model structure comparison")
    lines.append("-" * 78)
    lines.append(format_pair(ground_stats))
    lines.append("")

    per_species = json.loads(ground_stats["per_species_json"])
    lines.append("Per-species ground-state displacement statistics")
    lines.append("-" * 78)
    for el, s in per_species.items():
        lines.append(
            f"{el:2s}: n={s['n']:3d}, "
            f"total={s['total_A']:.6f} Å, "
            f"mean={s['mean_A']:.6f} Å, "
            f"rms={s['rms_A']:.6f} Å, "
            f"max={s['max_A']:.6f} Å"
        )
    lines.append("")

    lines.append("Nearest fine-tuned match for each base-MACE DFT structure")
    lines.append("-" * 78)
    show = nearest_ft_for_base[
        [
            "label_a",
            "label_b",
            "abs_dft_energy_diff_eV",
            "rms_distance_A",
            "mean_distance_A",
            "max_distance_A",
            "near_duplicate_by_distance_threshold",
        ]
    ].copy()
    lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("Nearest base-MACE match for each fine-tuned-MACE DFT structure")
    lines.append("-" * 78)
    show = nearest_base_for_ft[
        [
            "label_b",
            "label_a",
            "abs_dft_energy_diff_eV",
            "rms_distance_A",
            "mean_distance_A",
            "max_distance_A",
            "near_duplicate_by_distance_threshold",
        ]
    ].copy()
    lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("All cross-model pairwise distances, sorted by RMS distance")
    lines.append("-" * 78)
    show = out[
        [
            "label_a",
            "label_b",
            "abs_dft_energy_diff_eV",
            "total_distance_A",
            "mean_distance_A",
            "rms_distance_A",
            "max_distance_A",
            "structure_matcher_fit",
            "near_duplicate_by_distance_threshold",
        ]
    ].copy()
    lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("Files written")
    lines.append("-" * 78)
    lines.append(str(OUT_TXT))
    lines.append(str(OUT_CSV))

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote:\n  {OUT_TXT}\n  {OUT_CSV}")
    print("")
    print("Ground-state comparison:")
    print(format_pair(ground_stats))


if __name__ == "__main__":
    main()