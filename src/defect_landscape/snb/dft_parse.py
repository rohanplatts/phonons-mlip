from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .io import analysis_dir, read_csv_rows, write_csv_rows


DFT_STATUS_FIELDS = [
    "dft_id",
    "case_name",
    "selection_id",
    "model_name",
    "model_label",
    "cluster_id",
    "best_candidate_id",
    "best_mlip_energy_eV",
    "best_mlip_dE_eV",
    "selected_poscar",
    "representative_contcar",
    "dft_dir",
    "final_structure",
    "final_energy_eV",
    "max_final_force_eVA",
    "n_ionic_steps",
    "vasprun_ok",
    "outcar_present",
    "oszicar_present",
    "contcar_present",
    "converged",
    "converged_electronic",
    "converged_ionic",
    "outcar_required_accuracy",
    "restart_needed",
    "restart_reason",
]


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def read_oszicar_energy(path: str | Path) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    energy = _last_float(r"E0=\s*([-+0-9.Ee]+)", text)
    return energy if energy is not None else _last_float(r"\bF=\s*([-+0-9.Ee]+)", text)


def read_outcar_energy(path: str | Path) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    return _last_float(r"free\s+energy\s+TOTEN\s+=\s*([-+0-9.Ee]+)", text)


def parse_final_max_force_from_outcar(path: str | Path) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text(errors="ignore").splitlines()
    block_forces: list[list[float]] = []
    last_forces: list[list[float]] = []
    in_block = False
    for line in lines:
        if "TOTAL-FORCE" in line and "(eV/Angst)" in line:
            in_block = True
            block_forces = []
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        parts = stripped.split()
        if len(parts) < 6:
            if block_forces:
                last_forces = block_forces
            in_block = False
            continue
        try:
            block_forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
        except ValueError:
            if block_forces:
                last_forces = block_forces
            in_block = False
    if block_forces:
        last_forces = block_forces
    if not last_forces:
        return None
    forces = np.asarray(last_forces, dtype=float)
    return float(np.linalg.norm(forces, axis=1).max())


def read_vasprun(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"vasprun_ok": False}
    try:
        from pymatgen.io.vasp.outputs import Vasprun

        vasprun = Vasprun(str(path), parse_dos=False, parse_eigen=False)
        max_force = None
        if vasprun.ionic_steps:
            forces = np.asarray(vasprun.ionic_steps[-1].get("forces", []), dtype=float)
            if forces.size:
                max_force = float(np.linalg.norm(forces, axis=1).max())
        return {
            "vasprun_ok": True,
            "final_energy_eV": float(vasprun.final_energy),
            "max_final_force_eVA": max_force,
            "n_ionic_steps": len(vasprun.ionic_steps),
            "converged": bool(vasprun.converged),
            "converged_electronic": bool(vasprun.converged_electronic),
            "converged_ionic": bool(vasprun.converged_ionic),
        }
    except Exception as exc:
        return {"vasprun_ok": False, "vasprun_error": str(exc)}


def final_structure_path(dft_dir: str | Path) -> Path | None:
    dft_dir = Path(dft_dir)
    contcar = dft_dir / "CONTCAR"
    if contcar.exists() and contcar.stat().st_size > 0:
        return contcar
    poscar = dft_dir / "POSCAR"
    if poscar.exists() and poscar.stat().st_size > 0:
        return poscar
    return None


def parse_dft_folder(dft_dir: str | Path) -> dict[str, Any]:
    dft_dir = Path(dft_dir)
    outcar = dft_dir / "OUTCAR"
    oszicar = dft_dir / "OSZICAR"
    vasprun_xml = dft_dir / "vasprun.xml"
    final_structure = final_structure_path(dft_dir)

    vasprun = read_vasprun(vasprun_xml)
    final_energy = vasprun.get("final_energy_eV")
    if final_energy is None:
        final_energy = read_outcar_energy(outcar)
    if final_energy is None:
        final_energy = read_oszicar_energy(oszicar)

    max_force = vasprun.get("max_final_force_eVA")
    if max_force is None:
        max_force = parse_final_max_force_from_outcar(outcar)

    outcar_text = outcar.read_text(errors="ignore") if outcar.exists() else ""
    outcar_required_accuracy = "reached required accuracy" in outcar_text.lower()
    n_ionic_steps = vasprun.get("n_ionic_steps")
    if n_ionic_steps is None and oszicar.exists():
        n_ionic_steps = len(re.findall(r"^\s*\d+\s+F=", oszicar.read_text(errors="ignore"), flags=re.MULTILINE))

    converged_electronic = vasprun.get("converged_electronic", "")
    converged_ionic = vasprun.get("converged_ionic", "")
    converged = vasprun.get("converged", "")
    if not vasprun.get("vasprun_ok") and outcar_required_accuracy:
        converged = True
        converged_ionic = True

    reasons: list[str] = []
    if final_structure is None:
        reasons.append("missing final structure")
    if final_energy is None or (isinstance(final_energy, float) and math.isnan(final_energy)):
        reasons.append("missing final energy")
    if vasprun_xml.exists() and not vasprun.get("vasprun_ok"):
        reasons.append("unreadable vasprun.xml")
    if converged is False:
        reasons.append("VASP not converged")
    if not any([outcar.exists(), oszicar.exists(), vasprun_xml.exists()]):
        reasons.append("missing OUTCAR/OSZICAR/vasprun.xml")

    return {
        "dft_dir": str(dft_dir),
        "final_structure": str(final_structure) if final_structure else "",
        "final_energy_eV": final_energy if final_energy is not None else "",
        "max_final_force_eVA": max_force if max_force is not None else "",
        "n_ionic_steps": n_ionic_steps if n_ionic_steps is not None else "",
        "vasprun_ok": bool(vasprun.get("vasprun_ok", False)),
        "outcar_present": outcar.exists(),
        "oszicar_present": oszicar.exists(),
        "contcar_present": (dft_dir / "CONTCAR").exists(),
        "converged": converged,
        "converged_electronic": converged_electronic,
        "converged_ionic": converged_ionic,
        "outcar_required_accuracy": outcar_required_accuracy,
        "restart_needed": bool(reasons),
        "restart_reason": "; ".join(reasons),
    }


def check_dft(*, results_root: str | Path, case_name: str) -> Path:
    manifest_path = analysis_dir(results_root, case_name) / "dft_validation_manifest.csv"
    manifest = read_csv_rows(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"No DFT validation manifest found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    for job in manifest:
        parsed = parse_dft_folder(job["dft_dir"])
        row = {**job, **parsed}
        rows.append(row)
        if parsed["restart_needed"]:
            restart_rows.append(row)

    out = analysis_dir(results_root, case_name) / "dft_status.csv"
    restart = analysis_dir(results_root, case_name) / "dft_jobs_to_restart.csv"
    write_csv_rows(out, rows, DFT_STATUS_FIELDS)
    write_csv_rows(restart, restart_rows, DFT_STATUS_FIELDS)
    return out
