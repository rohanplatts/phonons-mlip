#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import argparse
import re
from dataclasses import dataclass, field
from typing import Any, Literal
import matplotlib
import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write
import torch

if __package__ in (None, ""):
    # Allow running as a script from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "mlip_phonons"

from .get_calc import get_calc_object
from .relax import relax
from .phonons import get_phonons, get_band_structure, get_dos, write_gamma_band_yaml_for_plumipy
from .plot import obj_plot_band, obj_plot_band_dos, obj_plot_dos
from .tools.plumipy_conversions import write_contcar_for_plumipy, write_minimal_outcar_for_plumipy
from .config_classes import ExecutiveCfg, ModelCfg, StructureCfg


REPO_ROOT = Path(__file__).resolve().parents[2]


matplotlib.use("Agg")  # no GUI, no .show()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available (check node).") # included for hpc use
torch.set_default_device("cuda") #TODO: make this configurable?

# ════════════════════════════════════
# private helpers (use _ to indicate these, not intended to be called globally
# ════════════════════════════════════


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        dict[str, Any]: Parsed YAML contents.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def _get_config_path(config: dict[str, Any], key: str) -> str | None:
    """Read a path-like config value from either top-level or a paths section."""
    paths = config.get("paths", {}) or {}
    value = paths.get(key, None)
    if value is None:
        value = config.get(key, None)
    return value

def _resolve_path(
    root: Path,
    value: str | Path | None,
    alt_root: Path | None = None,
) -> Path | None:
    """Resolve a path relative to alt_root if it exists there, otherwise root.

    Args:
        root (Path): Base directory for relative paths.
        value (str | Path | None): Path to resolve.
        alt_root (Path | None): Optional alternate root to prefer if the path exists there.

    Returns:
        Path | None: Absolute path or None if value is None.
    """
    if value is None:
        return None
    p = value if isinstance(value, Path) else Path(value)
    if p.is_absolute():
        return p
    if alt_root is not None:
        alt_candidate = alt_root / p
        if alt_candidate.exists():
            return alt_candidate
    return root / p

def _ints_from_any(x: Any) -> list[int]:
    """Extract integers from lists, tuples arrays or strings, to lists. 

    Args:
        x (Any): Input containing integers (str, list, tuple, or ndarray).

    Returns:
        list[int]: Extracted integers.

    Raises:
        ValueError: If input type is unsupported.
    """
    # Robustly extract ints from strings like "(3,3,3)", "[1, 1, 1]", "2 2 1", etc.
    if isinstance(x, str):
        return [int(v) for v in re.findall(r"-?\d+", x)]
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asanyarray(x).astype(int).flatten()
        return [int(v) for v in arr.tolist()]
    raise ValueError(f"Expected str/list/tuple/ndarray of ints, got {type(x)}")

# in config_classes
def _parse_supercell_matrix(sc_in: Any) -> tuple[int, int, int] | np.ndarray:
    """Parse a supercell matrix from a flexible input.

    Args:
        sc_in (Any): Input containing 3 or 9 integers.

    Returns:
        tuple[int, int, int] | np.ndarray: Parsed supercell matrix.

    Raises:
        ValueError: If the input does not contain 3 or 9 integers.
    """
    ints = _ints_from_any(sc_in)
    if len(ints) == 3:
        return (ints[0], ints[1], ints[2])
    if len(ints) == 9:
        return np.array(ints, dtype=int).reshape(3, 3)
    raise ValueError(f"Invalid supercell_matrix={sc_in!r}. Need 3 ints or 9 ints.")

# in config_classes
def _parse_kpts(kpts_in: Any) -> list[int]:
    """Parse a k-point mesh into three integers.

    Args:
        kpts_in (Any): Input containing 3 integers.

    Returns:
        list[int]: Three k-point mesh integers.

    Raises:
        ValueError: If the input does not contain exactly 3 integers.
    """
    ints = _ints_from_any(kpts_in)
    if len(ints) != 3:
        raise ValueError(f"Invalid kpts={kpts_in!r}. Need 3 ints.")
    return ints

def _is_identity_supercell(sc: tuple[int, int, int] | np.ndarray) -> bool:
    """Check whether a supercell matrix is the identity.

    Args:
        sc (tuple[int, int, int] | np.ndarray): Supercell matrix.

    Returns:
        bool: True if the matrix is identity.
    """
    if isinstance(sc, tuple):
        return sc == (1, 1, 1)
    return np.array_equal(sc, np.eye(3, dtype=int))


# ════════════════════════════════════
# Dataclasses for config
# ════════════════════════════════════


if False: 
    StructureGroup = Literal["pure", "defects"]

    @dataclass(frozen=True)
    class ExecutiveCfg:
        """Configuration for execution options and output naming.

        Attributes:
            plots (bool): Whether to generate plots.
            results_root (Path): Root output directory.
            raw_subdir (str): Subdirectory for raw outputs.
            plot_subdir (str): Subdirectory for plots.
            output_name_templates (dict[str, str]): Filename template overrides.
        """
        plots: bool = False
        results_root: Path = Path("results")
        raw_subdir: str = "raw"
        plot_subdir: str = "plot"
        output_name_templates: dict[str, str] = field(default_factory=dict)

        @staticmethod
        def from_config(d: dict[str, Any]) -> "ExecutiveCfg":
            """Build an ExecutiveCfg from a config dictionary.

            Args:
                d (dict[str, Any]): Parsed configuration dictionary.

            Returns:
                ExecutiveCfg: Constructed configuration instance.
            """
            ex = d.get("executive", {}) or {}
            out_names = ex.get("output_names", {}) or {}
            return ExecutiveCfg(
                plots=bool(ex.get("plots", False)),
                results_root=Path(ex.get("results_root", "results")),
                raw_subdir=str(ex.get("raw_subdir", "raw")),
                plot_subdir=str(ex.get("plot_subdir", "plot")),
                output_name_templates={str(k): str(v) for k, v in out_names.items()},
            ) 

    @dataclass(frozen=True)
    class ModelCfg:
        """Model configuration metadata.

        Attributes:
            name (str): Model name key.
            environment (str): Environment label or tag.
            model_path (Path): Path to the model file.
            default_structure (str): Default structure name.
        """
        name: str
        environment: str
        model_path: Path
        default_structure: str

        @staticmethod
        def from_config(config: dict[str, Any], model_name: str) -> "ModelCfg":
            """Build a ModelCfg from the main config dict.

            Args:
                config (dict[str, Any]): Parsed configuration dictionary.
                model_name (str): Model name key in config.

            Returns:
                ModelCfg: Constructed configuration instance.
            """
            m = config["models"][model_name]
            return ModelCfg(
                name=model_name,
                environment=str(m.get("environment", "")),
                model_path=Path(m.get("model_path", "")),
                default_structure=str(m.get("material", "")),
            )

    @dataclass(frozen=True)
    class StructureCfg:
        """Structure configuration for a run.

        Attributes:
            name (str): Structure name key.
            group (StructureGroup): "pure" or "defects".
            unitcell_path (Path): Path to the unit cell file.
            primitive_cell_path (Path | None): Path to primitive cell file.
            is_file_relaxed (bool): Whether the input structure is relaxed.
            supercell_matrix (tuple[int, int, int] | np.ndarray): Supercell matrix.
            delta (float): Displacement distance in angstrom.
            want_band_structure (bool): Whether to compute band structure.
            kpts (list[int]): k-point mesh for DOS.
            npts (int): Number of points along band path segments.
            width_ev (float): Smearing width in eV.
        """

        name: str
        group: StructureGroup
        unitcell_path: Path
        primitive_cell_path: Path | None
        is_file_relaxed: bool
        supercell_matrix: tuple[int, int, int] | np.ndarray
        delta: float
        want_band_structure: bool
        kpts: list[int]
        npts: int
        width_ev: float

        @staticmethod
        def from_config(config: dict[str, Any], structure_name: str) -> "StructureCfg":
            """Build a StructureCfg from the main config dict.

            Args:
                config (dict[str, Any]): Parsed configuration dictionary.
                structure_name (str): Structure name key in config.

            Returns:
                StructureCfg: Constructed configuration instance.
            """
            structures = config["structures"]
            group: StructureGroup | None = None
            entry: dict[str, Any] | None = None

            for g in ("pure", "defects"):
                if g in structures and isinstance(structures[g], dict) and structure_name in structures[g]:
                    group = g  
                    entry = structures[g][structure_name]
                    break

            if entry is None or group is None:
                raise ValueError(f"Structure {structure_name!r} not found under structures.pure/defects.")

            prim_rel = entry.get("primitive_cell_path", None)

            unit_rel = entry.get("unitcell_path", None)
            if unit_rel is None:
                raise ValueError("Missing required key: unitcell_path")

            return StructureCfg(
                name=structure_name,
                group=group,
                unitcell_path=Path(unit_rel),
                primitive_cell_path=Path(prim_rel) if prim_rel else None,
                is_file_relaxed=bool(entry.get("is_file_relaxed", False)),
                supercell_matrix=_parse_supercell_matrix(entry.get("supercell_matrix")),
                delta=float(entry.get("delta", 0.01)),
                want_band_structure=bool(entry.get("want_band_structure", True)),
                kpts=_parse_kpts(entry.get("kpts", [12, 12, 12])),
                npts=int(entry.get("npts", 400)),
                width_ev=float(entry.get("width_ev", 0.0)),
            )

# ═══════════════════════════════════
# default naming. mutable ofc. {base} is the filler for the model+material. 
# Since everything is unique via a) the Model, and b) the material, keep the model 
# names and material names unique and each run will have unique files PER model x material.
# for same runs it WILL rewrite. 
# ════════════════════════════════════



if False: # (added this to config_classes)
    # THE NAMES OF THE FILES THEMSELVES
    DEFAULT_NAME_TEMPLATES: dict[str, str] = {
        # raw
        "relax_traj": "{base}_relax.traj",
        "relaxed_poscar": "{base}_relaxed.poscar",
        "phonons_obj": "{base}_phonons.yaml",
        "force_constants": "{base}_force_constants.yaml",
        "phonon_dos_npz": "{base}_phonon_dos.npz",
        "phonon_band_yaml": "{base}_phonon_band.yaml",
        # plots (optional)
        "phonon_band_plot": "{base}_phonon_band_plot.png",
        "phonon_dispersion_dos_plot": "{base}_phonon_dispersion_dos.png",
        "phonon_dos_plot": "{base}_phonon_dos.png",
        # Plumipy
        "band_plumipy": "{base}_band.yaml",
        "contcar_gs_plumipy": "{base}_CONTCAR_GS",
        "outcar_gs_plumipy": "{base}_OUTCAR_GS"
    }
    # THE DIRECTORIES THE FILES ARE WRITTEN TO. 
    @dataclass(frozen=True)
    class OutputPlan:
        """Resolved output directories and filename mappings.

        Attributes:
            results_root (Path): Root output directory for this run.
            raw_dir (Path): Directory for raw outputs.
            plot_dir (Path): Directory for plot outputs.
            names (dict[str, str]): Output name templates mapped to keys.
        """
        results_root: Path
        raw_dir: Path
        plot_dir: Path
        names: dict[str, str]

        def raw(self, key: str) -> Path:
            """Resolve a raw output path by key.

            Args:
                key (str): Output name key.

            Returns:
                Path: Resolved raw output path.
            """
            if key not in self.names:
                raise KeyError(f"Unknown output key: {key}")
            p = _resolve_path(self.raw_dir, self.names[key])
            assert p is not None
            return p

        def plot(self, key: str) -> Path:
            """Resolve a plot output path by key.

            Args:
                key (str): Output name key.

            Returns:
                Path: Resolved plot output path.
            """
            if key not in self.names:
                raise KeyError(f"Unknown output key: {key}")
            p = _resolve_path(self.plot_dir, self.names[key])
            assert p is not None
            return p

        def plot_plumipy(self, key: str) -> Path: 
            """Resolve a plumipy plot output path by key.

            Args:
                key (str): Output name key.

            Returns:
                Path: Resolved plumipy output path.
            """
            if key not in self.names:
                raise KeyError(f"Unknown output key: {key}")
            p = _resolve_path(self.raw_dir / "Plumipy_Files", self.names[key])
            assert p is not None
            return p

    # ASSEMBLING THE NAMES AND OUTPUT DIRECTORIES INTO AN OUTPUT PLAN

    def build_output_plan(
        exec_cfg: ExecutiveCfg,
        model: ModelCfg,
        structure: StructureCfg,
        run_root: Path,
    ) -> OutputPlan:
        """Create an OutputPlan for a specific model/structure run.

        Args:
            exec_cfg (ExecutiveCfg): Execution configuration.
            model (ModelCfg): Model configuration.
            structure (StructureCfg): Structure configuration.
            run_root (Path): Root directory to resolve relative paths.

        Returns:
            OutputPlan: Resolved output plan.
        """

        # THE BASE IDENTIFIER FOR THE FILES
        base = f"{model.name}_{structure.name}"

        templates = dict(DEFAULT_NAME_TEMPLATES) # ensures its a dictionary.copy() 
        templates.update(exec_cfg.output_name_templates)

        names = {k: v.format(base=base, model=model.name, structure=structure.name) for k, v in templates.items()}

        results_root = (run_root / exec_cfg.results_root / model.name / structure.name).resolve()
        raw_dir = results_root / exec_cfg.raw_subdir
        plot_dir = results_root / exec_cfg.plot_subdir
        raw_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        return OutputPlan(results_root=results_root, raw_dir=raw_dir, plot_dir=plot_dir, names=names)


# ════════════════════════════════════
# RUNSTATE
# ════════════════════════════════════

@dataclass
class RunState:
    """Mutable run state for the pipeline.

    Attributes:
        unitcell_path (Path | None): Path to the unit cell file.
        primitive_path (Path | None): Path to the primitive cell file.
        unitcell_unrelaxed (Atoms | None): Loaded unit cell atoms.
        primitive_atoms (Atoms | None): Loaded primitive atoms.
        phonopy_unitcell_unrelaxed (Atoms | None): Chosen unit cell for phonopy.
        phonopy_unitcell_relaxed (Atoms | None): Relaxed unit cell for phonopy.
        bandpath_source (Atoms | None): Atoms used for band path generation.
        primitive_m_a (Any): Primitive matrix or atoms for phonopy.
        mode_label (str): Workflow label describing chosen path.
        phonon (Any): Phonopy object with computed data.
        warnings (list[str]): Collected warnings.
        written_files (dict[str, Path]): Outputs recorded by key.
    """
    # inputs (loaded)
    unitcell_path: Path | None = None
    primitive_path: Path | None = None
    unitcell_unrelaxed: Atoms | None = None
    primitive_atoms: Atoms | None = None

    # derived / chosen
    phonopy_unitcell_unrelaxed: Atoms | None = None
    phonopy_unitcell_relaxed: Atoms | None = None
    bandpath_source: Atoms | None = None
    primitive_m_a: Any = None  # either np.ndarray (3x3) or Atoms, as your get_phonons accepts
    mode_label: str = ""

    # results
    phonon: Any = None  # Phonopy object

    # bookkeeping
    warnings: list[str] = field(default_factory=list)
    written_files: dict[str, Path] = field(default_factory=dict)

    def note_file(self, key: str, path: Path) -> None: # to note later
        """Record a written file path in the run state.

        Args:
            key (str): Output name key.
            path (Path): Output path.

        Returns:
            None
        """
        self.written_files[key] = path

# ════════════════════════════════════
# CHOOSERS
# ════════════════════════════════════

def _supercell_det(sc: tuple[int, int, int] | np.ndarray) -> int:
    """Compute the determinant magnitude for a supercell matrix.

    Args:
        sc (tuple[int, int, int] | np.ndarray): Supercell matrix.

    Returns:
        int: Absolute determinant rounded to nearest int.
    """
    if isinstance(sc, tuple):
        return int(sc[0] * sc[1] * sc[2])
    det = float(np.linalg.det(sc.astype(float)))
    return int(round(abs(det)))

def choose_phonopy_unitcell_and_primitive(
    structure_cfg: StructureCfg,
    unitcell_atoms: Atoms,
    primitive_atoms: Atoms | None
):
    """Choose the phonopy unit cell and primitive matrix/atoms for a run.

    Args:
        structure_cfg (StructureCfg): Structure configuration.
        unitcell_atoms (Atoms): Loaded unit cell atoms.
        primitive_atoms (Atoms | None): Loaded primitive atoms, if provided.

    Returns:
        tuple[Atoms, Any, str]: Phonopy unit cell, primitive matrix/atoms, and mode label.
    """
    I = np.eye(3)

    if structure_cfg.group == "defects":
        return unitcell_atoms, I, "defect_supercell_as_primitive"

    # pure
    if primitive_atoms is None:
        return unitcell_atoms, I, "unitcell_as_primitive_assumed"

    det = _supercell_det(structure_cfg.supercell_matrix)
    expected_super_atoms = len(primitive_atoms) * det

    # if unitcell file already has the expected supercell atom count, treat it as already-built supercell
    # and use primitive cell as the phonopy unit cell for standard primitive-BZ band structure.
    if det > 1 and len(unitcell_atoms) == expected_super_atoms:
        return primitive_atoms, I, "primitive_unitcell_standard_workflow"

    return unitcell_atoms, primitive_atoms, "unitcell_with_provided_primitive"


# ════════════════════════=════════════
# STEPS FOR PIPELINE.
# ════════════════════════════════════=

def step_load_structures(
    state: RunState,
    structure_cfg: StructureCfg,
    run_root: Path,
    structures_root: Path | None,
) -> None:
    """Load unit cell and optional primitive cell from disk.

    Args:
        state (RunState): Current run state.
        structure_cfg (StructureCfg): Structure configuration.
        run_root (Path): Root directory to resolve relative paths.
        structures_root (Path | None): Optional root for structure files.

    Returns:
        None
    """
    unitcell_path = _resolve_path(run_root, structure_cfg.unitcell_path, structures_root)
    if unitcell_path is None:
        raise ValueError("Need unitcell_path")
    state.unitcell_path = unitcell_path
    state.unitcell_unrelaxed = read(unitcell_path)

    if structure_cfg.primitive_cell_path is not None:
        prim_path = _resolve_path(run_root, structure_cfg.primitive_cell_path, structures_root)
        if prim_path is not None:
            state.primitive_path = prim_path
            state.primitive_atoms = read(prim_path)

def step_relax_if_needed(
    state: RunState,
    structure_cfg: StructureCfg,
    out: OutputPlan,
    calc: Any,
) -> None:
    """Relax the structure if the input is not already relaxed.

    Args:
        state (RunState): Current run state.
        structure_cfg (StructureCfg): Structure configuration.
        out (OutputPlan): Output plan for file locations.
        calc (Any): ASE-compatible calculator.

    Returns:
        None
    """
    assert state.phonopy_unitcell_unrelaxed is not None

    if structure_cfg.is_file_relaxed:
        # We trust the input file is already relaxed
        state.phonopy_unitcell_relaxed = state.phonopy_unitcell_unrelaxed.copy()
        return

    # Relax positions only (your relax() does not change lattice vectors)
    atoms = state.phonopy_unitcell_unrelaxed.copy()
    atoms.calc = calc

    relax_traj_path = out.raw("relax_traj")
    atoms = relax(atoms, outdir=out.raw_dir, filename=out.names["relax_traj"])
    state.note_file("relax_traj", relax_traj_path)

    relaxed_poscar = out.raw("relaxed_poscar")
    write(relaxed_poscar, atoms, format="vasp")
    state.note_file("relaxed_poscar", relaxed_poscar)

    state.phonopy_unitcell_relaxed = atoms

def _get_supported_element_types(calc: Any) -> set[str]:
    element_types = None
    if hasattr(calc, "element_types"):
        element_types = getattr(calc, "element_types")

    if element_types is None:
        raise AttributeError("Calculator does not expose supported elements via `calc.element_types`.")

    try:
        values = list(element_types)
    except TypeError as exc:
        raise TypeError(f"`calc.element_types` is not iterable (type={type(element_types)}).") from exc

    supported: set[str] = set()
    for v in values:
        if isinstance(v, str):
            supported.add(v)
        elif isinstance(v, (int, np.integer)): # account for atomic numbers
            try:
                from ase.data import chemical_symbols
                z = int(v)
                if 0 < z < len(chemical_symbols):
                    supported.add(chemical_symbols[z])
                else:
                    supported.add(str(v))
            except Exception:
                supported.add(str(v))
        else:
            supported.add(str(v))
    return supported

def step_defect_safeguard(state: RunState, calc: Any, structure_cfg: StructureCfg) -> None:
    """Validate defect workflow assumptions.
    Check structure elements are supported by the model.

    Args:
        structure_cfg (StructureCfg): Structure configuration.

    Returns:
        None
    """
    if structure_cfg.group == "defects" and not _is_identity_supercell(structure_cfg.supercell_matrix):
        raise ValueError(
            "Defect workflow expects the defect supercell is already the full cell to analyse. "
            "Set supercell_matrix to [1,1,1] (identity)."
        )
    assert state.phonopy_unitcell_unrelaxed is not None
    structure_elements = set(state.phonopy_unitcell_unrelaxed.get_chemical_symbols())
    supported = _get_supported_element_types(calc)
    missing = sorted(structure_elements - supported)
    if missing:
        raise ValueError(
            "Structure contains element(s) not supported by model. "
            f"missing={missing} structure={sorted(structure_elements)} supported={sorted(supported)}"
        )

def step_phonons(
    state: RunState,
    structure_cfg: StructureCfg,
    out: OutputPlan,
    calc: Any,
) -> None:
    """Compute phonons and persist raw outputs.

    Args:
        state (RunState): Current run state.
        structure_cfg (StructureCfg): Structure configuration.
        out (OutputPlan): Output plan for file locations.
        calc (Any): ASE-compatible calculator.

    Returns:
        None
    """
    
    assert state.phonopy_unitcell_relaxed is not None

    phonons_obj_path = out.raw("phonons_obj")

    ph = get_phonons(
        unitcell_a=state.phonopy_unitcell_relaxed,
        mlip_calc=calc,
        supercell_m=structure_cfg.supercell_matrix if structure_cfg.group == "pure" else np.eye(3),
        primitive_m_a=state.primitive_m_a if structure_cfg.group == "pure" else np.eye(3),
        delta=structure_cfg.delta,
        outdir=phonons_obj_path,
    )
    state.phonon = ph
    state.note_file("phonons_obj", phonons_obj_path)

def step_band_yaml(
    state: RunState,
    structure_cfg: StructureCfg,
    out: OutputPlan,
) -> None:
    """Compute band structure and write band.yaml.

    Args:
        state (RunState): Current run state.
        structure_cfg (StructureCfg): Structure configuration.
        out (OutputPlan): Output plan for file locations.

    Returns:
        None
    """
    if not structure_cfg.want_band_structure:
        return

    if state.phonon is None:
        raise RuntimeError("Band structure requested but phonons were not computed.")

    assert state.bandpath_source is not None
    
    band_yaml_path = out.raw("phonon_band_yaml")
    ph = get_band_structure(
        unitcell_NOTrelaxed=state.bandpath_source,
        phonon=state.phonon,
        npts=structure_cfg.npts,
        outdir_bandyaml=band_yaml_path,
    )
    state.phonon = ph
    state.note_file("phonon_band_yaml", band_yaml_path)

def step_plots(
    state: RunState,
    exec_cfg: ExecutiveCfg,
    structure_cfg: StructureCfg,
    model_cfg: ModelCfg,
    out: OutputPlan,
) -> None:
    """Save phonon plots to disk.

    Args:
        state (RunState): Current run state.
        exec_cfg (ExecutiveCfg): Execution configuration.
        structure_cfg (StructureCfg): Structure configuration.
        model_cfg (ModelCfg): Model configuration.
        out (OutputPlan): Output plan for file locations.

    Returns:
        None
    """
    if not exec_cfg.plots:
        return

    if state.phonon is None:
        raise RuntimeError("Plots requested but phonons were not computed.")

    import matplotlib.pyplot as plt  # backend already set to Agg

    base_name = f"{model_cfg.name} {structure_cfg.name}"
    plots: list[tuple[str, Any, str]] = [
        ("phonon_dos_plot", obj_plot_dos, f"{base_name} DOS"),
    ]
    if structure_cfg.want_band_structure:
        plots.extend(
            [
                ("phonon_band_plot", obj_plot_band, f"{base_name} Band Structure"),
                ("phonon_dispersion_dos_plot", obj_plot_band_dos, f"{base_name} DOS and Band structure"),
            ]
        )

    for key, fn, title in plots:
        path = out.plot(key)
        fn(state.phonon, outdir=path, title=title)
        state.note_file(key, path)

    plt.close("all")

def step_plumipy_conversion(
        state: RunState,
        structure_cfg: StructureCfg,
        out: OutputPlan,
        mlip_calc: Any,
) -> None:
    """Write plumipy-compatible output files.

    Args:
        state (RunState): Current run state.
        structure_cfg (StructureCfg): Structure configuration.
        out (OutputPlan): Output plan for file locations.
        mlip_calc (Any): ASE-compatible calculator.

    Returns:
        None
    """

    atoms_unrelaxed = state.phonopy_unitcell_unrelaxed # ase atoms
    atoms = state.phonopy_unitcell_relaxed # ase atoms
    atoms.calc = mlip_calc #fix these None type lame ass redlines elegantly. 

    # Write plumipy inputs
    contcar_path = out.plot_plumipy("contcar_gs_plumipy")
    outcar_path = out.plot_plumipy("outcar_gs_plumipy")
    band_path = out.plot_plumipy("band_plumipy")
    
    if atoms:
        print("writing contcar and outcar")
        write_contcar_for_plumipy(atoms, contcar_path)
        write_minimal_outcar_for_plumipy(atoms, outcar_path)
        write_gamma_band_yaml_for_plumipy(state.phonon, band_path)
    """
    python scripts/tools/plumipy_conversions.py 'small-omat-0'
    """
    state.note_file("contcar_gs_plumipy", contcar_path)
    state.note_file("outcar_gs_plumipy", outcar_path)
    state.note_file("band_plumipy", band_path)


def main() -> int:
    """Entry point for the phonon pipeline command line .

    Returns:
        int: exit code.
    """
    parser = argparse.ArgumentParser(description="Compute phonons, DOS, and optional band structure.")
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Model name from config.yml models. If omitted, uses mlip_phonons.defaults.model_name.",
    )
    parser.add_argument(
        "--structure",
        help="Override which structure to run (key under structures.pure/defects). "
            "If omitted, uses models.<model>.material from config.",
        default=None,
    )
    parser.add_argument(
        "--config",
        help=f"Path to config.yml (default: {REPO_ROOT / 'config.yml'})",
        default=str(REPO_ROOT / "config.yml"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    run_root = config_path.parent

    config = _load_yaml(config_path)
    defaults_cfg = (config.get("mlip_phonons", {}) or {}).get("defaults", {}) or {}
    default_model = defaults_cfg.get("model_name")
    model_name = args.model_name or default_model
    if not model_name:
        models_cfg = config.get("models", {}) or {}
        model_name = next(iter(models_cfg.keys()), None)
    if not model_name:
        raise SystemExit("Missing model_name. Set mlip_phonons.defaults.model_name in config.yml.")
    model_name = str(model_name)
    structure_override = args.structure or defaults_cfg.get("structure") or None

    try:
        exec_cfg = ExecutiveCfg.from_config(config)
        model_cfg = ModelCfg.from_config(config, model_name)

        assets_root_cfg = _get_config_path(config, "assets_root")
        structures_root_cfg = _get_config_path(config, "structures_root")
        assets_root = _resolve_path(run_root, assets_root_cfg) if assets_root_cfg else (run_root / "assets")
        structures_root = (
            _resolve_path(run_root, structures_root_cfg)
            if structures_root_cfg
            else (run_root / "assets" / "structures")
        )

        structure_name = structure_override or model_cfg.default_structure
        structure_cfg = StructureCfg.from_config(config, structure_name)

        out = build_output_plan(exec_cfg, model_cfg, structure_cfg, run_root)
        calc = get_calc_object(model_cfg.name, models_root= assets_root / "models")

        state = RunState()

        # ze pipeline
        step_load_structures(state, structure_cfg, run_root, structures_root)
        assert state.unitcell_unrelaxed is not None
        phonopy_unitcell, primitive_m_a, mode_label = choose_phonopy_unitcell_and_primitive(
            structure_cfg, state.unitcell_unrelaxed, state.primitive_atoms
        )
        state.phonopy_unitcell_unrelaxed = phonopy_unitcell.copy()
        state.bandpath_source = phonopy_unitcell.copy()
        state.primitive_m_a = primitive_m_a
        state.mode_label = mode_label
        step_relax_if_needed(state, structure_cfg, out, calc)
        #step_defect_safeguard(state, calc,structure_cfg)
        step_phonons(state, structure_cfg, out, calc)
        step_band_yaml(state, structure_cfg, out)
        if state.phonon is None:
            raise RuntimeError("DOS requested but phonons were not computed.")
        state.phonon = get_dos(state.phonon, kpts_mesh=structure_cfg.kpts, outdir=None)
        step_plots(state, exec_cfg, structure_cfg, model_cfg, out)
        step_plumipy_conversion(state, structure_cfg,out, calc) 

        print(f"||SUCCESS|| mode={state.mode_label} results={out.results_root}")

        if state.written_files:
            print("||WRITTEN FILESS||")
            for k, p in state.written_files.items():
                print(f"  - {k}: {p}")

    except Exception as e:
        msg = f"{model_name} failed. error is {e}. Moving on to next model."
        print(msg)
        with open("test/errorlog.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
