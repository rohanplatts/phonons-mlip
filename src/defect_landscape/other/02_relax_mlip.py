import os

N_THREADS = "16"

os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS

try:
    import torch

    torch.set_num_threads(int(N_THREADS))
    torch.set_num_interop_threads(1)

except ImportError:
    pass


from pathlib import Path
import json
import numpy as np

from ase.io import read, write
from ase.optimize import FIRE

from mlip_phonons.get_calc import get_calc_object

ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

SNB_DIR = ROOT / "runs" / "snb_initials"
OUT_ROOT = ROOT / "runs" / "mlip_relaxed"

MODELS_ROOT = Path("/home/rnpla/projects/mlip_phonons/assets/models")

DEVICE = "cuda"
DTYPE = "float32"

# Edit these names to whatever your get_calc_object expects.
MODELS = {
    "base_mace": "mace-mpa-0-medium",
    "finetuned_mace": "mace-mpa-0-medium-ft-cspbi3-neutral",
}

FMAX = 0.03
MAX_STEPS = 600


# -------------------------
# Find all ShakeNBreak POSCARs
# -------------------------

poscars = sorted(SNB_DIR.rglob("POSCAR"))

if not poscars:
    raise RuntimeError(f"No POSCAR files found under {SNB_DIR}")

print(f"Found {len(poscars)} ShakeNBreak POSCARs.")


# -------------------------
# Relax with each MLIP
# -------------------------

for model_label, model_name in MODELS.items():
    print(f"\n=== Relaxing with {model_label}: {model_name} ===")

    calc_mlip = get_calc_object(
        model_name,
        models_root=MODELS_ROOT,
        device=DEVICE,
        dtype=DTYPE,
        include_vdw=True,
    )

    for poscar in poscars:
        candidate_id = poscar.parent.relative_to(SNB_DIR)
        out_dir = OUT_ROOT / model_label / candidate_id
        out_dir.mkdir(parents=True, exist_ok=True)

        result_json = out_dir / "result.json"

        # Skip if already done.
        if result_json.exists():
            print(f"Skipping existing: {out_dir}")
            continue

        print(f"Relaxing: {candidate_id}")

        atoms = read(poscar)
        atoms.calc = calc_mlip

        dyn = FIRE(atoms, trajectory=str(out_dir / "trajectory.traj"))
        dyn.run(fmax=FMAX, steps=MAX_STEPS)

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max())

        write(out_dir / "CONTCAR", atoms, format="vasp", direct=True, sort=False)

        with open(result_json, "w") as f:
            json.dump(
                {
                    "model_label": model_label,
                    "model_name": model_name,
                    "candidate_id": str(candidate_id),
                    "input_poscar": str(poscar),
                    "relaxed_contcar": str(out_dir / "CONTCAR"),
                    "energy_eV": energy,
                    "max_force_eVA": max_force,
                    "fmax_target_eVA": FMAX,
                    "max_steps": MAX_STEPS,
                },
                f,
                indent=2,
            )

print("\nDone MLIP relaxations.")