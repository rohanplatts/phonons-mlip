from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:  # scipy is optional; fall back to the local implementation below.
    _linear_sum_assignment = None


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")
RUNS_ROOT = ROOT / "runs"
MODELS_ROOT = Path("/home/rnpla/projects/mlip_phonons/assets/models")

DEFAULT_MODELS = {
    "base_mace": "mace-mpa-0-medium",
    "finetuned_mace": "mace-mpa-0-medium-ft-cspbi3-neutral",
}

SAME_RMS_THRESHOLD_A = 0.05
SAME_MAX_THRESHOLD_A = 0.20
DFT_AMBIGUOUS_GAP_EV = 0.025
MLIP_AMBIGUOUS_GAP_EV = 0.050

_MATCHER = None


def get_matcher():
    global _MATCHER
    if _MATCHER is None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        _MATCHER = StructureMatcher(
            ltol=0.2,
            stol=0.3,
            angle_tol=5,
            primitive_cell=False,
            scale=False,
            attempt_supercell=False,
        )
    return _MATCHER


def analysis_root(analysis_name: str) -> Path:
    return RUNS_ROOT / analysis_name


def analysis_dir(analysis_name: str) -> Path:
    return analysis_root(analysis_name) / "analysis"


def candidate_manifest_path(analysis_name: str) -> Path:
    return analysis_dir(analysis_name) / "candidate_manifest.csv"


def case_manifest_path(analysis_name: str) -> Path:
    return analysis_dir(analysis_name) / "case_manifest.csv"


def ensure_analysis_dirs(analysis_name: str) -> None:
    root = analysis_root(analysis_name)
    for rel in [
        "analysis",
        "case_inputs",
        "snb_variant_inputs",
        "dft_references",
        "mlip_relaxed",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def clear_analysis(analysis_name: str) -> None:
    root = analysis_root(analysis_name)
    if root.exists():
        shutil.rmtree(root)
    ensure_analysis_dirs(analysis_name)


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+%-]+", "_", str(label)).strip("_")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def upsert_rows(
    path: Path,
    new_rows: list[dict[str, Any]],
    fieldnames: list[str],
    key_fields: list[str],
) -> None:
    rows = read_csv_rows(path)
    index = {tuple(row[k] for k in key_fields): row for row in rows}
    for row in new_rows:
        index[tuple(str(row[k]) for k in key_fields)] = {k: row.get(k, "") for k in fieldnames}
    ordered = sorted(index.values(), key=lambda r: tuple(str(r.get(k, "")) for k in key_fields))
    write_csv_rows(path, ordered, fieldnames)


def copy_structure(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_reference_metadata(ref_dir: Path) -> dict[str, Any]:
    meta_path = Path(ref_dir) / "reference_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing reference metadata: {meta_path}")
    with meta_path.open() as f:
        return json.load(f)


def discover_reference_dirs(dft_references_dir: Path) -> list[Path]:
    dft_references_dir = Path(dft_references_dir)
    dirs: list[Path] = []
    for section in ["minimum", "alternatives"]:
        parent = dft_references_dir / section
        if parent.exists():
            dirs.extend(sorted([p for p in parent.iterdir() if p.is_dir()]))
    return dirs


def parse_float(x: Any) -> float:
    try:
        return float(str(x).replace("D", "E"))
    except Exception:
        return math.nan


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def result_json_path(analysis_name: str, model_label: str, case_label: str, variant_label: str) -> Path:
    return (
        analysis_root(analysis_name)
        / "mlip_relaxed"
        / model_label
        / safe_label(case_label)
        / safe_label(variant_label)
        / "result.json"
    )


def mlip_contcar_path(analysis_name: str, model_label: str, case_label: str, variant_label: str) -> Path:
    return (
        analysis_root(analysis_name)
        / "mlip_relaxed"
        / model_label
        / safe_label(case_label)
        / safe_label(variant_label)
        / "CONTCAR"
    )


def structure_matcher_fit(file_a: Path, file_b: Path) -> bool:
    try:
        from pymatgen.core import Structure

        return bool(get_matcher().fit(Structure.from_file(file_a), Structure.from_file(file_b)))
    except Exception:
        return False


def hungarian_min(cost: np.ndarray) -> np.ndarray:
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")

    if _linear_sum_assignment is not None:
        row_ind, col_ind = _linear_sum_assignment(a)
        assign = np.empty(a.shape[0], dtype=int)
        assign[row_ind] = col_ind
        return assign

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


def compare_structures(file_a: Path, file_b: Path) -> dict[str, Any]:
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
    cell = a.cell
    pbc = a.pbc

    all_dist = np.zeros(len(a), dtype=float)
    per_species: dict[str, Any] = {}

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
            "rms_A": float(np.sqrt(np.mean(species_dist**2))),
            "max_A": float(np.max(species_dist)),
        }

    return {
        "n_atoms": int(len(a)),
        "total_A": float(np.sum(all_dist)),
        "mean_A": float(np.mean(all_dist)),
        "rms_A": float(np.sqrt(np.mean(all_dist**2))),
        "max_A": float(np.max(all_dist)),
        "cell_max_abs_diff_A": float(np.max(np.abs(np.array(a.cell) - np.array(b.cell)))),
        "per_species": per_species,
    }


def fmt_float(x: Any, ndp: int = 6) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{ndp}f}"
