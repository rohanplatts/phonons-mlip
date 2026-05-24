from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlip_phonons.get_calc import get_calc_object
from pipeline_common import (
    DEFAULT_MODELS,
    MODELS_ROOT,
    analysis_root,
    candidate_manifest_path,
    load_json,
    mlip_contcar_path,
    read_csv_rows,
    result_json_path,
    safe_label,
)


def set_threads(n_threads: str) -> None:
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[name] = n_threads

    try:
        import torch

        torch.set_num_threads(int(n_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def load_candidates(
    analysis_name: str,
    case_label: str | None,
    variant_label: str | None,
    case_labels: list[str] | None = None,
) -> list[dict[str, str]]:
    manifest = candidate_manifest_path(analysis_name)
    rows = read_csv_rows(manifest)
    if not rows:
        raise FileNotFoundError(f"No candidates found. Run 00_prepare_case.py first: {manifest}")
    if case_label and case_labels:
        raise RuntimeError("Use either --case or --case-list, not both.")
    if case_label:
        rows = [r for r in rows if r["case_label"] == case_label]
    if case_labels:
        allowed = set(case_labels)
        rows = [r for r in rows if r["case_label"] in allowed]
    if variant_label:
        rows = [r for r in rows if r["variant_label"] == safe_label(variant_label)]
    if not rows:
        raise RuntimeError("No candidates matched the requested filters.")
    return rows


def status(
    analysis_name: str,
    model_label: str | None,
    case_label: str | None = None,
    variant_label: str | None = None,
    case_labels: list[str] | None = None,
) -> None:
    rows = load_candidates(analysis_name, case_label, variant_label, case_labels)
    models = [model_label] if model_label else list(DEFAULT_MODELS)
    print(f"Analysis: {analysis_name}")
    print(f"Candidates: {len(rows)}")
    for model in models:
        done = 0
        for row in rows:
            p = result_json_path(analysis_name, model, row["case_label"], row["variant_label"])
            if p.exists():
                done += 1
        print(f"{model}: {done}/{len(rows)} complete")


def run_relaxations(args: argparse.Namespace) -> None:
    set_threads(args.threads)
    rows = load_candidates(args.analysis_name, args.case, args.variant, args.case_list)
    model_name = args.model_name or DEFAULT_MODELS.get(args.model)
    if model_name is None:
        raise RuntimeError(f"Unknown model label {args.model}. Provide --model-name.")

    print(f"Analysis: {args.analysis_name}")
    print(f"Model: {args.model} -> {model_name}")
    print(f"Candidates: {len(rows)}")

    calc_mlip = get_calc_object(
        model_name,
        models_root=MODELS_ROOT,
        device=args.device,
        dtype=args.dtype,
        include_vdw=not args.no_vdw,
    )

    for row in rows:
        case_label = row["case_label"]
        variant_label = row["variant_label"]
        out_dir = (
            analysis_root(args.analysis_name)
            / "mlip_relaxed"
            / args.model
            / safe_label(case_label)
            / safe_label(variant_label)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        result_json = out_dir / "result.json"
        if result_json.exists() and not args.force:
            print(f"Skipping existing: {case_label}/{variant_label}")
            continue

        poscar = Path(row["staged_variant_poscar"])
        print(f"Relaxing {case_label}/{variant_label}")
        atoms = read(poscar)
        atoms.calc = calc_mlip

        t0 = time.time()
        dyn = FIRE(atoms, trajectory=str(out_dir / "trajectory.traj"))
        dyn.run(fmax=args.fmax, steps=args.max_steps)
        elapsed = time.time() - t0

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max())
        converged = bool(max_force <= args.fmax)

        contcar = mlip_contcar_path(args.analysis_name, args.model, case_label, variant_label)
        write(contcar, atoms, format="vasp", direct=True, sort=False)

        with result_json.open("w") as f:
            json.dump(
                {
                    "analysis_name": args.analysis_name,
                    "model_label": args.model,
                    "model_name": model_name,
                    "case_label": case_label,
                    "variant_label": variant_label,
                    "input_poscar": str(poscar),
                    "relaxed_contcar": str(contcar),
                    "dft_contcar": row["staged_dft_contcar"],
                    "dft_energy_eV": row["dft_energy_eV"],
                    "energy_eV": energy,
                    "max_force_eVA": max_force,
                    "fmax_target_eVA": args.fmax,
                    "max_steps": args.max_steps,
                    "converged": converged,
                    "elapsed_sec": elapsed,
                    "device": args.device,
                    "dtype": args.dtype,
                    "include_vdw": not args.no_vdw,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        print(f"  E = {energy:.8f} eV, fmax = {max_force:.4f} eV/A, converged={converged}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relax prepared SnB variants with one MLIP.")
    parser.add_argument("--analysis-name", required=True, help="Named folder under src/defect_landscape/runs")
    parser.add_argument("--model", default="base_mace", help="Model label used in output folders")
    parser.add_argument("--model-name", help="Model name understood by mlip_phonons.get_calc_object")
    parser.add_argument("--case", help="Optional case label filter")
    parser.add_argument("--case-list", nargs="+", help="Optional list of case labels to relax in one model load")
    parser.add_argument("--variant", help="Optional SnB variant label filter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", default="16")
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--no-vdw", action="store_true", help="Disable D3/vdW wrapper where supported")
    parser.add_argument("--force", action="store_true", help="Re-run completed relaxations")
    parser.add_argument("--status", action="store_true", help="Print completion status and exit")

    args = parser.parse_args()

    if args.status:
        status(args.analysis_name, args.model if args.model else None, args.case, args.variant, args.case_list)
        return

    if args.model_name is None and args.model not in DEFAULT_MODELS:
        raise SystemExit(f"Unknown --model {args.model}; use one of {sorted(DEFAULT_MODELS)} or pass --model-name")

    run_relaxations(args)


if __name__ == "__main__":
    main()
