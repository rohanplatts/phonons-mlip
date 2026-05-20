from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def get_matcher(ltol: float = 0.2, stol: float = 0.3, angle_tol: float = 5.0):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )


def structure_matcher_fit(
    file_a: str | Path,
    file_b: str | Path,
    *,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> bool:
    from pymatgen.core import Structure

    matcher = get_matcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    return bool(matcher.fit(Structure.from_file(file_a), Structure.from_file(file_b)))


def hungarian_min(cost: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = np.empty(cost.shape[0], dtype=int)
        assignment[row_ind] = col_ind
        return assignment
    except Exception:
        pass

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

    assignment = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assignment[i - 1] = j - 1
    return assignment


def compare_structures(file_a: str | Path, file_b: str | Path) -> dict[str, Any]:
    from ase.geometry import find_mic
    from ase.io import read

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
    all_dist = np.zeros(len(a), dtype=float)
    per_species: dict[str, Any] = {}

    for element in sorted(set(sym_a)):
        idx_a = np.where(sym_a == element)[0]
        idx_b = np.where(sym_b == element)[0]
        cost = np.zeros((len(idx_a), len(idx_b)), dtype=float)
        for i, atom_i in enumerate(idx_a):
            disp = pos_b[idx_b] - pos_a[atom_i]
            disp_mic, _ = find_mic(disp, cell=a.cell, pbc=a.pbc)
            cost[i, :] = np.linalg.norm(disp_mic, axis=1)
        assignment = hungarian_min(cost)
        distances = cost[np.arange(len(idx_a)), assignment]
        all_dist[idx_a] = distances
        per_species[str(element)] = {
            "n": int(len(idx_a)),
            "mean_A": float(np.mean(distances)),
            "rms_A": float(np.sqrt(np.mean(distances**2))),
            "max_A": float(np.max(distances)),
            "total_A": float(np.sum(distances)),
        }

    return {
        "n_atoms": int(len(a)),
        "mean_A": float(np.mean(all_dist)),
        "rms_A": float(np.sqrt(np.mean(all_dist**2))),
        "max_A": float(np.max(all_dist)),
        "total_A": float(np.sum(all_dist)),
        "cell_max_abs_diff_A": float(np.max(np.abs(np.array(a.cell) - np.array(b.cell)))),
        "per_species": per_species,
        "per_species_json": json.dumps(per_species, sort_keys=True),
    }

