from pathlib import Path
import numpy as np

from ase import Atoms
from ase.io import write


# -------------------------
# Paths
# -------------------------

ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

# This file is only used for its lattice.
# It can be your existing 79-atom relaxed/initial vacancy POSCAR.
LATTICE_SOURCE = ROOT / "data" / "bulk" / "POSCAR_neutral_I_vac"

BULK_OUT = ROOT / "data" / "bulk" / "POSCAR"
DEFECT_OUT = ROOT / "data" / "initial_defect" / "POSCAR"

# Approximate iodine vacancy coordinate.
# The script removes the nearest ideal iodine site.
VACANCY_TARGET_FRAC = np.array([0.3, 0.4, 0.5])


# -------------------------
# Helpers
# -------------------------

def read_lattice_from_poscar(path: Path) -> np.ndarray:
    """Read only the lattice vectors from a POSCAR."""
    lines = path.read_text().splitlines()

    scale = float(lines[1].split()[0])

    cell = np.array(
        [
            [float(x) for x in lines[2].split()[:3]],
            [float(x) for x in lines[3].split()[:3]],
            [float(x) for x in lines[4].split()[:3]],
        ],
        dtype=float,
    )

    return scale * cell


def pbc_frac_dist(a, b) -> float:
    """Fractional distance with periodic wrapping."""
    d = np.asarray(a) - np.asarray(b)
    d -= np.round(d)
    return float(np.linalg.norm(d))


# -------------------------
# Build ideal 80-atom CsPbI3
# -------------------------

def build_bulk_80(cell: np.ndarray) -> Atoms:
    """
    Build ideal Cs16 Pb16 I48 = 80 atom CsPbI3 supercell.

    This matches the 80-atom analogue of the 160-atom ordering:
    same xy columns, half the number of z layers.
    """

    symbols = []
    scaled_positions = []

    z_odd = [0.25, 0.75]
    z_even = [0.0, 0.5]

    cs_xy = [
        (0.0, 0.25),
        (0.0, 0.75),
        (0.25, 0.0),
        (0.25, 0.5),
        (0.5, 0.25),
        (0.5, 0.75),
        (0.75, 0.0),
        (0.75, 0.5),
    ]

    pb_xy = [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.25, 0.25),
        (0.25, 0.75),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.75, 0.25),
        (0.75, 0.75),
    ]

    i1_xy = pb_xy

    i2_xy = [
        (0.125, 0.375),
        (0.125, 0.875),
        (0.375, 0.125),
        (0.375, 0.625),
        (0.625, 0.375),
        (0.625, 0.875),
        (0.875, 0.125),
        (0.875, 0.625),
    ]

    i3_xy = [
        (0.125, 0.125),
        (0.125, 0.625),
        (0.375, 0.375),
        (0.375, 0.875),
        (0.625, 0.125),
        (0.625, 0.625),
        (0.875, 0.375),
        (0.875, 0.875),
    ]

    # Cs: 8 columns x 2 z layers = 16
    for x, y in cs_xy:
        for z in z_odd:
            symbols.append("Cs")
            scaled_positions.append((x, y, z))

    # Pb: 8 columns x 2 z layers = 16
    for x, y in pb_xy:
        for z in z_even:
            symbols.append("Pb")
            scaled_positions.append((x, y, z))

    # I sublattice 1: 16
    for x, y in i1_xy:
        for z in z_odd:
            symbols.append("I")
            scaled_positions.append((x, y, z))

    # I sublattice 2: 16
    for x, y in i2_xy:
        for z in z_even:
            symbols.append("I")
            scaled_positions.append((x, y, z))

    # I sublattice 3: 16
    for x, y in i3_xy:
        for z in z_even:
            symbols.append("I")
            scaled_positions.append((x, y, z))

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True,
    )

    counts = {s: atoms.get_chemical_symbols().count(s) for s in ["Cs", "Pb", "I"]}

    if len(atoms) != 80 or counts != {"Cs": 16, "Pb": 16, "I": 48}:
        raise RuntimeError(f"Bad bulk: n={len(atoms)}, counts={counts}")

    return atoms


def make_i_vacancy(bulk: Atoms) -> tuple[Atoms, int, np.ndarray]:
    """Remove the iodine closest to VACANCY_TARGET_FRAC."""
    symbols = bulk.get_chemical_symbols()
    scaled = bulk.get_scaled_positions(wrap=True)

    iodine_indices = [i for i, s in enumerate(symbols) if s == "I"]

    remove_index = min(
        iodine_indices,
        key=lambda i: pbc_frac_dist(scaled[i], VACANCY_TARGET_FRAC),
    )

    removed_frac = scaled[remove_index].copy()

    defect = bulk.copy()
    del defect[remove_index]

    counts = {s: defect.get_chemical_symbols().count(s) for s in ["Cs", "Pb", "I"]}

    if len(defect) != 79 or counts != {"Cs": 16, "Pb": 16, "I": 47}:
        raise RuntimeError(f"Bad defect: n={len(defect)}, counts={counts}")

    return defect, remove_index, removed_frac


# -------------------------
# Main
# -------------------------

def main():
    if not LATTICE_SOURCE.exists():
        raise FileNotFoundError(f"Missing lattice source: {LATTICE_SOURCE}")

    cell = read_lattice_from_poscar(LATTICE_SOURCE)

    bulk = build_bulk_80(cell)
    defect, removed_index, removed_frac = make_i_vacancy(bulk)

    BULK_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFECT_OUT.parent.mkdir(parents=True, exist_ok=True)

    write(BULK_OUT, bulk, format="vasp", direct=True, sort=False, vasp5=True)
    write(DEFECT_OUT, defect, format="vasp", direct=True, sort=False, vasp5=True)

    print("Wrote matched 80/79 ShakeNBreak inputs:")
    print(f"  bulk:   {BULK_OUT}")
    print(f"  defect: {DEFECT_OUT}")

    print("\nComposition:")
    print("  bulk:   Cs16 Pb16 I48, 80 atoms")
    print("  defect: Cs16 Pb16 I47, 79 atoms")

    print("\nVacancy:")
    print(f"  target frac coord:  {VACANCY_TARGET_FRAC}")
    print(f"  removed I index:    {removed_index}")
    print(f"  removed frac coord: {removed_frac}")


if __name__ == "__main__":
    main()