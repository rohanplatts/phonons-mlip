from __future__ import annotations

from pathlib import Path

from .config import parse_oxidation_states
from .io import case_root, import_snb_inputs


def generate_snb_inputs(
    *,
    bulk: str | Path,
    defect: str | Path,
    oxidation_states: dict[str, int | float] | list[str],
    results_root: str | Path,
    case_name: str,
    overwrite: bool = False,
) -> Path:
    try:
        from pymatgen.core import Structure
        from shakenbreak.input import Distortions
    except Exception as exc:
        raise RuntimeError(
            "ShakeNBreak generation requires pymatgen and shakenbreak. "
            "Install/activate the environment containing shakenbreak, or use --snb-dir to import existing SnB inputs."
        ) from exc

    states = parse_oxidation_states(oxidation_states) if isinstance(oxidation_states, list) else oxidation_states
    if not states:
        raise ValueError("Oxidation states are required for SnB generation, e.g. --oxidation-states Cs=1 Pb=2 I=-1")

    out_dir = case_root(results_root, case_name) / "snb_inputs"
    if out_dir.exists() and any(out_dir.rglob("POSCAR")) and not overwrite:
        return import_snb_inputs(snb_dir=out_dir, results_root=results_root, case_name=case_name, overwrite=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    bulk_structure = Structure.from_file(bulk)
    defect_structure = Structure.from_file(defect)

    distortions = Distortions.from_structures(
        defects=defect_structure,
        bulk=bulk_structure,
        oxidation_states=states,
    )
    distortions.write_vasp_files(output_path=str(out_dir), verbose=True)
    return import_snb_inputs(snb_dir=out_dir, results_root=results_root, case_name=case_name, overwrite=True)

