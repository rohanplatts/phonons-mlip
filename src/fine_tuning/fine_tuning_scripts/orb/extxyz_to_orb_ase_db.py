#!/usr/bin/env python3
"""Convert REF_energy/REF_forces extxyz files into ORB-compatible ASE sqlite DBs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import iread


def json_safe(value: Any) -> Any:
    """Convert ASE/numpy scalar metadata into values ASE DB can serialize."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def get_energy(atoms, energy_key: str) -> float:
    if energy_key in atoms.info:
        return float(atoms.info[energy_key])
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        raise ValueError(
            f"Frame is missing energy key '{energy_key}' and has no calculator energy"
        ) from exc


def get_forces(atoms, forces_key: str) -> np.ndarray:
    if forces_key in atoms.arrays:
        forces = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
    else:
        try:
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"Frame is missing forces array '{forces_key}' and has no calculator forces"
            ) from exc
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"Forces have shape {forces.shape}, expected ({len(atoms)}, 3)")
    return forces


def convert_extxyz_to_db(
    input_path: Path,
    output_path: Path,
    *,
    energy_key: str,
    forces_key: str,
    overwrite: bool,
    charge: int | None,
    spin: int | None,
) -> int:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    db = connect(output_path)

    for atoms in iread(input_path, format="extxyz"):
        energy = get_energy(atoms, energy_key)
        forces = get_forces(atoms, forces_key)
        metadata = {
            key: json_safe(value)
            for key, value in atoms.info.items()
            if key not in {energy_key, "energy"}
        }
        metadata["source_extxyz"] = str(input_path)
        if charge is not None:
            metadata["charge"] = int(charge)
        if spin is not None:
            metadata["spin"] = int(spin)

        atoms = atoms.copy()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        db.write(atoms, data=metadata)
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extxyz files with REF_energy/REF_forces into ASE sqlite DBs for ORB."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input extxyz file.")
    parser.add_argument("--output", required=True, type=Path, help="Output .db path.")
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument("--forces-key", default="REF_forces")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Optional total charge metadata for OrbMol models.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        help="Optional spin multiplicity metadata for OrbMol models.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    args = parser.parse_args()

    count = convert_extxyz_to_db(
        args.input,
        args.output,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        overwrite=args.overwrite,
        charge=args.charge,
        spin=args.spin,
    )
    print(f"Wrote {count} frames to {args.output}")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "input": str(args.input),
                    "output": str(args.output),
                    "frames": count,
                    "energy_key": args.energy_key,
                    "forces_key": args.forces_key,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
