from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io import analysis_dir, case_root, load_candidates, read_csv_rows, safe_model_label, write_csv_rows


RELAXATION_FIELDS = [
    "case_name",
    "model_name",
    "model_label",
    "candidate_id",
    "input_poscar",
    "relaxed_contcar",
    "energy_eV",
    "dE_mlip_eV",
    "max_force_eVA",
    "converged",
    "elapsed_sec",
    "include_vdw",
    "result_json",
]


def set_threads(n_threads: int | str) -> None:
    value = str(n_threads)
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = value
    try:
        import torch

        torch.set_num_threads(int(value))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def result_path(results_root: str | Path, case_name: str, model_name: str, candidate_id: str) -> Path:
    return case_root(results_root, case_name) / "mlip_relaxed" / safe_model_label(model_name) / candidate_id / "result.json"


def collect_relaxation_results(results_root: str | Path, case_name: str, model_name: str | None = None) -> list[dict[str, Any]]:
    root = case_root(results_root, case_name) / "mlip_relaxed"
    pattern = f"{safe_model_label(model_name)}/**/result.json" if model_name else "**/result.json"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        with path.open() as handle:
            rows.append(json.load(handle))
    return rows


def write_relaxation_summary(results_root: str | Path, case_name: str) -> Path:
    rows = collect_relaxation_results(results_root, case_name)
    if not rows:
        return analysis_dir(results_root, case_name) / "relaxation_results.csv"
    by_model_min: dict[str, float] = {}
    for row in rows:
        by_model_min[row["model_name"]] = min(by_model_min.get(row["model_name"], float("inf")), float(row["energy_eV"]))
    out_rows = []
    for row in rows:
        out = dict(row)
        out["model_label"] = safe_model_label(row["model_name"])
        out["dE_mlip_eV"] = float(row["energy_eV"]) - by_model_min[row["model_name"]]
        out_rows.append(out)
    out = analysis_dir(results_root, case_name) / "relaxation_results.csv"
    write_csv_rows(out, out_rows, RELAXATION_FIELDS)
    return out


def relax_model(
    *,
    model_name: str,
    results_root: str | Path,
    case_name: str,
    models_root: str | Path,
    device: str = "cuda",
    dtype: str = "float32",
    include_vdw: bool = True,
    fmax: float = 0.03,
    max_steps: int = 600,
    overwrite: bool = False,
    threads: int | str = 16,
) -> Path:
    from ase.io import read, write
    from common.get_calc import get_calc_object
    from common.relax import relax

    set_threads(threads)
    candidates = load_candidates(results_root, case_name)

    run_device = device
    if str(device).startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA requested but not available; falling back to CPU.")
                run_device = "cpu"
        except Exception:
            pass

    calc = get_calc_object(
        model_name,
        models_root=Path(models_root),
        device=run_device,
        dtype=dtype,
        include_vdw=include_vdw,
    )

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        out_dir = case_root(results_root, case_name) / "mlip_relaxed" / safe_model_label(model_name) / candidate_id
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "result.json"
        if json_path.exists() and not overwrite:
            print(f"Skipping existing: {model_name}/{candidate_id}")
            continue

        print(f"Relaxing {case_name}/{candidate_id} with {model_name}")
        atoms = read(candidate["staged_poscar"])
        atoms.calc = calc
        t0 = time.time()
        relax(
            atoms,
            fmax=fmax,
            outdir=out_dir,
            filename="trajectory.traj",
            type="FIRE",
            steps=max_steps,
        )
        elapsed = time.time() - t0

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max())
        contcar = out_dir / "CONTCAR"
        write(contcar, atoms, format="vasp", direct=True, sort=False)

        payload = {
            "case_name": case_name,
            "model_name": model_name,
            "model_label": safe_model_label(model_name),
            "candidate_id": candidate_id,
            "input_poscar": candidate["staged_poscar"],
            "relaxed_contcar": str(contcar),
            "energy_eV": energy,
            "max_force_eVA": max_force,
            "fmax_target_eVA": fmax,
            "max_steps": max_steps,
            "converged": bool(max_force <= fmax),
            "elapsed_sec": elapsed,
            "device": run_device,
            "dtype": dtype,
            "include_vdw": include_vdw,
            "result_json": str(json_path),
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"  E={energy:.8f} eV, fmax={max_force:.4f} eV/A, converged={payload['converged']}")

    return write_relaxation_summary(results_root, case_name)
