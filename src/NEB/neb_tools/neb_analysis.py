from __future__ import annotations

import numpy as np
from ase.geometry import find_mic
from ase.optimize import FIRE


class LoopDetected(RuntimeError):
    pass


class LoopGuard:
    def __init__(
        self,
        opt: FIRE,
        *,
        window: int = 60,
        max_unique: int = 2,
        repeat: int = 30,
        rounding: int = 6,
        label: str = "",
    ):
        self.opt = opt
        self.window = int(window)
        self.max_unique = int(max_unique)
        self.repeat = int(repeat)
        self.rounding = int(rounding)
        self.label = str(label)
        self.history: list[tuple[float, float]] = []
        self.counts: dict[tuple[float, float], int] = {}
        self.last: tuple[float, float] | None = None
        self.last_count = 0

    def __call__(self) -> None:
        forces = self.opt.atoms.get_forces()
        fmax = float(np.sqrt((forces * forces).sum(axis=1)).max())
        energy = float(self.opt.atoms.get_potential_energy())

        key = (round(energy, self.rounding), round(fmax, self.rounding))

        if key == self.last:
            self.last_count += 1
        else:
            self.last = key
            self.last_count = 1

        if self.last_count >= self.repeat:
            label = f" ({self.label})" if self.label else ""
            raise LoopDetected(
                f"Loop guard{label}: same (energy,fmax) repeated {self.last_count} steps: {key}"
            )

        self.history.append(key)
        self.counts[key] = self.counts.get(key, 0) + 1
        if len(self.history) > self.window:
            old = self.history.pop(0)
            cnt = self.counts.get(old, 0) - 1
            if cnt <= 0:
                self.counts.pop(old, None)
            else:
                self.counts[old] = cnt

        if len(self.history) >= self.window and len(self.counts) <= self.max_unique:
            label = f" ({self.label})" if self.label else ""
            raise LoopDetected(
                f"Loop guard{label}: only {len(self.counts)} unique (energy,fmax) values over "
                f"{len(self.history)} steps"
            )


def attach_loop_guard(
    opt: FIRE,
    *,
    window: int = 60,
    max_unique: int = 2,
    repeat: int = 30,
    rounding: int = 6,
    label: str = "",
) -> None:
    guard = LoopGuard(
        opt,
        window=window,
        max_unique=max_unique,
        repeat=repeat,
        rounding=rounding,
        label=label,
    )
    opt.attach(guard, interval=1)


# below is not used anymmore. 
def hungarian_min(cost: np.ndarray) -> np.ndarray:
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")
    n = int(a.shape[0])

    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1
    return assign


def map_final_to_initial_by_species(a, b):

    # purpose is to ensure that we are comparing the atoms correctly such that 
    # we have a minimum number of movements. This means finding the projection of one 
    # coord to another, building a cost matrix for all such projections, and then using hungarian 
    # min to map the indexes of the coordinates to each other such that there is maximal overlap
    
    a2 = a.copy()
    b2 = b.copy()

    cell = a2.cell
    pbc = a2.pbc
    sym = np.array(a2.get_chemical_symbols())
    pos_a = a2.get_positions()
    pos_b = b2.get_positions()
    new_pos_b = pos_b.copy()

    # minimisation is done per species
    
    for el in np.unique(sym):
        idx = np.where(sym == el)[0]
        xa = pos_a[idx]
        xb = pos_b[idx]

        c = np.zeros((len(idx), len(idx)), dtype=float)
        for i in range(len(idx)):
            d = xb - xa[i]
            d, _ = find_mic(d, cell=cell, pbc=pbc)
            c[i, :] = np.linalg.norm(d, axis=1)

        assign = hungarian_min(c)
        new_pos_b[idx] = xb[assign]

    b2.set_positions(new_pos_b)
    return b2


def build_images(a, b, n_images: int):
    if n_images < 3:
        raise ValueError("n_images must be >= 3")
    return [a] + [a.copy() for _ in range(n_images - 2)] + [b]


def normalize_image_cells(images, *, tol: float = 1e-6) -> float:
    """
    Collapse tiny floating-point cell noise across a NEB band.

    ASE NEB requires identical periodic cells. Some archived VASP paths differ only
    at the 1e-7 to 1e-8 Angstrom level because of text serialization. If the cell
    mismatch is within tol, copy the first image cell onto every image and return
    the maximum absolute cell difference encountered.

    Raises:
        ValueError: if the mismatch exceeds tol.
    """
    if not images:
        return 0.0

    ref_cell = np.asarray(images[0].cell.array, dtype=float)
    max_diff = 0.0
    for img in images[1:]:
        diff = float(np.max(np.abs(np.asarray(img.cell.array, dtype=float) - ref_cell)))
        max_diff = max(max_diff, diff)

    if max_diff > tol:
        raise ValueError(
            f"Variable-cell NEB paths are not supported by ASE NEB; max periodic cell mismatch {max_diff:.3e} "
            f"exceeds tolerance {tol:.3e}"
        )

    for img in images[1:]:
        img.cell = images[0].cell.copy()
    return max_diff


def energies_relative(images) -> np.ndarray:
    e = np.array([img.get_potential_energy() for img in images], dtype=float)
    return e - e[0]


def reaction_coordinate(images) -> np.ndarray:
    s = [0.0] 
    cell = images[0].cell
    pbc = images[0].pbc
    for a, b in zip(images[:-1], images[1:]):
        # images[:-1] is image n-1 and images[1:] is for images n. 
        d = b.get_positions() - a.get_positions()
        d, _ = find_mic(d, cell=cell, pbc=pbc)
        s.append(s[-1] + float(np.linalg.norm(d)))
    return np.array(s, dtype=float)
