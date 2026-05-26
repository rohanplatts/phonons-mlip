from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class NEBDefaults:
    n_images_fallback: int

    # to quickly converge to an initial rough path
    maxstep_mlip_guess: float # in angstroms, is the maximum atomic displacement per optimiser step
    fmax_mlip_guess: float
    steps_mlip_guess: int
    k_spring_mlip: float

    k_spring: float

    # to finely converge to the final path 
    maxstep_mlip_d3: float # to get the final path 
    fmax_mlip_d3: float
    steps_mlip_d3: int

    # to now shift the images up the path to get the maximum. 
    maxstep_ci: float
    fmax_ci: float
    steps_ci: int


@dataclass(frozen=True)
class NEBPaths:
    structures_dir: Path
    poscar_i: Path
    poscar_f: Path
    dft_neb_dat: Path | None
    models_root: Path


@dataclass(frozen=True)
class NEBInputs:
    model_name: str
    n_images: int | None
    poscar_i: Path
    poscar_f: Path
    dft_neb_dat: Path | None
    relax_endpoints: bool
    remap_f_i: bool
    include_vdw: bool
    compare: bool = False
    report_benchmark: bool = False
    overwrite: bool = False
    models_root: Path | None = None
    results_root: Path | None = None
    vasp_inputs_dir: Path | None = None
    device: str = "cuda"
    dtype: str = "float32"


@dataclass(frozen=True)
class NEBOutputDirs:
    out_raw: Path
    vasp_mlip_d3_dir: Path | None = None
    vasp_ci_dir: Path | None = None


@dataclass(frozen=True)
class NEBResults:
    s_mlip: np.ndarray
    e_mlip: np.ndarray
    barrier: float
    delta_e: float
