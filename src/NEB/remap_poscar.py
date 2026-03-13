#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

# This file lives in <repo>/src/NEB/checking_neb.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from NEB.neb_tools.neb_analysis import map_final_to_initial_by_species
from NEB.neb_tools.neb_parsers import load_yaml, resolve_path

"""
This script is used to a) produce and save a rearranged poscar_f.vasp file to 
best match the indices in the poscar_i.vasp file, but MAINLY its purpose is to compute 
basic metrics on the displacement patterns between poscar_i.vasp and poscar_f.vasp WITHOUT reordering
and poscar_i.vasp and modified_poscar_f.vasp WITH reordering.

does this by calculating the average interatomic displacement, the number of atoms with displacement greater than 
1 angstrom, and the maximum interatomig displacement. 

The algorithm used to reorder indices based on overlap is the element-wise hungarian 
algorithm implemeneted in run_neb_raw.py and called here. 

"""


def _metrics(a, b, *, threshold_ang: float) -> tuple[float, int, float]:
    d = b.get_positions() - a.get_positions()
    d, _ = find_mic(d, cell=a.cell, pbc=a.pbc)
    r = np.linalg.norm(d, axis=1)
    return float(r.mean()), int((r > float(threshold_ang)).sum()), float(r.max())


def _parse_args(
    argv: list[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mlip-remap",
        description=(
            "Compute displacement metrics between POSCAR_i and POSCAR_f, and write a final structure ('remapped_POSCAR_f') "
            "that maximises overlap between POSCAR_i and POSCAR_f "
        ),
    )
    parser.add_argument(
        "--poscar-i",
        type=Path,
        default=None,
        help="Initial POSCAR path. If omitted, uses neb.defaults.poscar_i from <repo_root>/config.yml.",
    )
    parser.add_argument(
        "--poscar-f",
        type=Path,
        default=None,
        help="Final POSCAR path. If omitted, uses neb.defaults.poscar_f from <repo_root>/config.yml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    config_path = (repo_root / "config.yml").expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yml at repo root: {config_path}")
    config = load_yaml(config_path)
    run_root = config_path.parent

    neb_cfg = config.get("neb", {}) or {}
    neb_defaults_cfg = neb_cfg.get("defaults", {}) or {}

    cfg_poscar_i = resolve_path(run_root, neb_defaults_cfg.get("poscar_i"))
    cfg_poscar_f = resolve_path(run_root, neb_defaults_cfg.get("poscar_f"))
    if cfg_poscar_i is None or cfg_poscar_f is None:
        raise ValueError("config.yml must define neb.defaults.poscar_i and neb.defaults.poscar_f.")

    args = _parse_args(argv)

    using_defaults = (args.poscar_i is None) or (args.poscar_f is None)
    if using_defaults:
        print("USING DEFAULTS")

    poscar_i = args.poscar_i or cfg_poscar_i
    poscar_f = args.poscar_f or cfg_poscar_f
    out_poscar_f = poscar_f.parent / "remapped_POSCAR_f.vasp"

    a = read(poscar_i)
    b = read(poscar_f)

    b_mapped = map_final_to_initial_by_species(a, b)
    out_poscar_f.parent.mkdir(parents=True, exist_ok=True)
    write(out_poscar_f, b_mapped, format="vasp", direct=True)

    threshold_ang = 1.0
    avg_f, n_gt1_f, max_f = _metrics(a, b, threshold_ang=threshold_ang)
    avg_m, n_gt1_m, max_m = _metrics(a, b_mapped, threshold_ang=threshold_ang)

    print(f"POSCAR_f: avg={avg_f:.6f}A, >{threshold_ang:g}A={n_gt1_f}, max={max_f:.6f}A")
    print(f"modified: avg={avg_m:.6f}A, >{threshold_ang:g}A={n_gt1_m}, max={max_m:.6f}A")
    print(f"Wrote mapped POSCAR_f to {out_poscar_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
