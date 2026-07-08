from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import numpy as np 
import re

"""
This is where all of the classes defined from the config.yml are stored. 
Hence this script is dedicated to reading config.yml, and obtaining its values in a structured way.

Hence if you want to add a new calculation to perform for every model, 
then you would:
1. first identify how you want the inputs for that calculation structured 
in config.yml. for example: 

    (as you can see in config.yml, the current structure is to add 
    a new dictionary, for example: 
    structures:
    pure:
        diamond:
        unitcell_path: diamond_super.poscar
        is_file_relaxed: true # if it is relaxed then kpoint path accuracy decreases, 
        # for which it is then recomended to definitely also submit the primitive cell
        primitive_cell_path: diamond_primitive.poscar
        
        supercell_matrix: (3,3,3) 
        delta: 0.01 # displacement size
        want_band_structure:  false 

        kpts: [12, 12, 12]
        npts: 400
        width_ev: 1e-3
        
        new calc: 
            enabled = True 
            newcalc_var1 = ..
            newcalc_var2 = .. 
            etc) 

2. then just simply add the attributes that this calculation requires from config to its 
relevant data class. If there is some 'overuling' element to this calculation, like say you want 
the option to make every model run this calculation with the exact same parameters, then, keeping 
the newcalc example, you would update config.yml the section pertaining to 'executive': 

    executive:
    plots: True
    newcalc_set_all_var: 
        enabled = True
        newcalc_var1 = ..
        newcalc_var2 = .. 
        etc

then add that to executive config. 

if the data pertains to the structure, add it to structurecfg. if the data pertains 
to the model, add it to modelcfg. 

3. this last step would then be adding this functionality to main.py in the form of a step. 
#TODO: improve this doc string. step 3 is vague. update readme with it.


"""



StructureGroup = Literal["pure", "defects"] # structures are either pure or defects. 

DEFAULT_NAME_TEMPLATES: dict[str, str] = {
    "relax_traj": "{base}_relax.traj",
    "relaxed_poscar": "{base}_relaxed.poscar",
    "phonons_obj": "{base}_phonons.yaml",
    "force_constants": "{base}_force_constants.yaml",
    "phonon_dos_npz": "{base}_phonon_dos.npz",
    "phonon_band_yaml": "{base}_phonon_band.yaml",
    "phonon_band_plot": "{base}_phonon_band_plot.png",
    "phonon_dispersion_dos_plot": "{base}_phonon_dispersion_dos.png",
    "phonon_dos_plot": "{base}_phonon_dos.png",
    "band_plumipy": "{base}_band.yaml",
    "contcar_gs_plumipy": "{base}_CONTCAR_GS",
    "outcar_gs_plumipy": "{base}_OUTCAR_GS",
}

# some private parsers
def _ints_from_any(x: Any) -> list[int]:
    """Extract integers from lists, tuples arrays or strings, to lists.

    Args:
        x (Any): Input containing integers (str, list, tuple, or ndarray).

    Returns:
        list[int]: Extracted integers.

    Raises:
        ValueError: If input type is unsupported.
    """
    if isinstance(x, str):
        return [int(v) for v in re.findall(r"-?\d+", x)]
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asanyarray(x).astype(int).flatten()
        return [int(v) for v in arr.tolist()]
    raise ValueError(f"Expected str/list/tuple/ndarray of ints, got {type(x)}")

# some private parsers
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



@dataclass(frozen=True)
class ExecutiveCfg:
    """Configuration for execution options and output naming.

    Attributes:
        plots (bool): Whether to generate plots.
        plumipy (bool): Whether to generate Plumipy-ready exports.
        results_root (Path): Root output directory.
        raw_subdir (str): Subdirectory for raw outputs.
        plot_subdir (str): Subdirectory for plots.
        output_name_templates (dict[str, str]): Filename template overrides.
    """
    plots: bool = False
    plumipy: bool = False
    results_root: Path = Path("results")
    raw_subdir: str = "raw"
    plot_subdir: str = "plot"
    output_name_templates: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_config(d: dict[str, Any]) -> "ExecutiveCfg":
        """Build an ExecutiveCfg from main config dict

        Args:
            d (dict[str, Any]): Parsed configuration dictionary.

        Returns:
            ExecutiveCfg: Constructed configuration instance.
        """
        ex = d.get("executive", {}) or {}
        out_names = ex.get("output_names", {}) or {}
        return ExecutiveCfg(
            plots=bool(ex.get("plots", False)),
            plumipy=bool(ex.get("plumipy", False)),
            results_root=Path(ex.get("results_root", "results")),
            raw_subdir=str(ex.get("raw_subdir", "raw")),
            plot_subdir=str(ex.get("plot_subdir", "plot")),
            output_name_templates={str(k): str(v) for k, v in out_names.items()},
        ) 

@dataclass(frozen=True)
class ModelCfg:
    """Model configuration data.

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
    """Structure configuration data.

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
    @staticmethod 
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

        templates = dict(DEFAULT_NAME_TEMPLATES)
        templates.update(exec_cfg.output_name_templates)

        names = {k: v.format(base=base, model=model.name, structure=structure.name) for k, v in templates.items()}

        results_root = (run_root / exec_cfg.results_root / structure.name).resolve()
        raw_dir = results_root / exec_cfg.raw_subdir
        plot_dir = results_root / exec_cfg.plot_subdir
        raw_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        return OutputPlan(results_root=results_root, raw_dir=raw_dir, plot_dir=plot_dir, names=names)
