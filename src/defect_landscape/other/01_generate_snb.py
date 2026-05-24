from pathlib import Path
from pymatgen.core import Structure
from shakenbreak.input import Distortions


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

BULK_POSCAR = ROOT / "data" / "bulk" / "POSCAR"
DEFECT_POSCAR = ROOT / "data" / "initial_defect" / "POSCAR"

OUT_DIR = ROOT / "runs" / "snb_initials"


bulk = Structure.from_file(BULK_POSCAR)
defect = Structure.from_file(DEFECT_POSCAR)

print(f"Bulk sites: {len(bulk)}")
print(f"Defect sites: {len(defect)}")

dist = Distortions.from_structures(
    defects=defect,
    bulk=bulk,
    oxidation_states={"Cs": 1, "Pb": 2, "I": -1},
)

dist.write_vasp_files(
    output_path=str(OUT_DIR),
    verbose=True,
)

print(f"\nWrote ShakeNBreak distorted inputs to:\n{OUT_DIR}")