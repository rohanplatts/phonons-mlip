from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

from .phon_classes import Structure


 
def read_poscar(path: Union[str, Path]) -> Structure:
    """Read a POSCAR/CONTCAR file into a Structure.

    Args:
        path (Union[str, Path]): Path to POSCAR/CONTCAR file.

    Returns:
        Structure: Parsed structure data.
    """

    p = Path(path)
    # Remove blank lines for more robust parsing
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() != ""]
    if len(lines) < 8:
        raise ValueError(f"Too few lines to be a POSCAR/CONTCAR: {p}")

    # Scale factor and lattice vectors (rows)
    scale = float(lines[1].split()[0])
    if scale <= 0:
        raise ValueError(f"Unsupported scale factor (<=0) in {p}: {scale}")

    lat = np.array([_parse_floats(lines[i], 3) for i in range(2, 5)], dtype=float) * scale  # (3,3)
    idx = 5

    # i had heard that VASP may list either element symbols or counts first; so we detect which.
    tokens = lines[idx].split()
    if _all_int(tokens):
        elements: List[str] = []
        counts = [int(t) for t in tokens]
        idx += 1
    else:
        elements = tokens
        idx += 1
        counts = [int(t) for t in lines[idx].split()]
        idx += 1

    # Total atom count
    n = int(np.sum(counts))
    if n <= 0:
        raise ValueError(f"Invalid atom counts in {p}: {counts}")

    if idx >= len(lines):
        raise ValueError(f"Unexpected EOF in {p} after counts")

    # Optional selective dynamics line
    if lines[idx].split()[0].lower().startswith("s"):
        idx += 1

    if idx >= len(lines):
        raise ValueError(f"Unexpected EOF in {p} before coordinate type")

    # Coordinate type: Direct or Cartesian
    ctype = lines[idx].split()[0].lower()
    idx += 1
    is_direct = ctype.startswith("d")
    is_cart = ctype.startswith("c") or ctype.startswith("k")
    if not (is_direct or is_cart):
        raise ValueError(f"Unknown coordinate type '{lines[idx-1]}' in {p}")

    # Ensure enough coordinate lines remain
    if idx + n > len(lines):
        raise ValueError(f"Not enough coordinate lines in {p}: need {n}, have {len(lines)-idx}")

    coords = np.array([_parse_floats(lines[idx + i], 3) for i in range(n)], dtype=float)  # (N,3)

    # Convert to fractional if coordinates were Cartesian so we can later
    # apply minimum-image wrapping in fractional space.
    if is_direct:
        frac = coords
    else:
        inv_lat = np.linalg.inv(lat)
        frac = coords @ inv_lat  # (N,3) @ (3,3)

    # Return in a normalized structure container (lattice + fractional coords).
    return Structure(lattice=lat, frac=frac, elements=elements, counts=counts)


 
def _extract_masses(data: Mapping[str, Any], natom: int) -> Optional[List[float]]:
    """Extract atomic masses from a phonopy band.yaml file. they are stored per 
    'point' in the file, we need an array of them indexed with the indexed atoms.

    Args:
        data (Mapping[str, Any]): Parsed YAML data.
        natom (int): Number of atoms expected.

    Returns:
        Optional[List[float]]: Mass list if available, otherwise None.
    """

    # if there is a flat "mass" list
    m = data.get("mass", None)
    if isinstance(m, list) and len(m) == natom and all(isinstance(x, (int, float)) for x in m):
        return [float(x) for x in m]

    # is its a 'points' list.
    pts = data.get("points", None)
    if isinstance(pts, list) and len(pts) == natom:
        masses: List[float] = []
        for pt in pts:
            if not isinstance(pt, dict) or "mass" not in pt:
                return None
            masses.append(float(pt["mass"]))
        return masses

    # if its an 'atoms' list 
    atoms = data.get("atoms", None) or data.get("atom", None)
    if isinstance(atoms, list) and len(atoms) == natom:
        masses = []
        for a in atoms:
            if isinstance(a, dict) and "mass" in a:
                masses.append(float(a["mass"]))
            else:
                return None
        return masses

    return None


 
def _parse_eigenvector(ev: Any, natom: int) -> List[List[complex]]:
    """Parse a phonopy eigenvector entry into complex vectors.

    Args:
        ev (Any): Eigenvector entry from YAML.
        natom (int): Number of atoms.

    Returns:
        List[List[complex]]: Parsed eigenvector list (natom x 3).
    """

    # Expect one eigenvector per atom
    if not isinstance(ev, list) or len(ev) != natom:
        raise ValueError("Invalid eigenvector shape (expected list of length natom)")
    out: List[List[complex]] = []
    for a in ev:
        # Each atom must have 3 components
        if not isinstance(a, list) or len(a) != 3:
            raise ValueError("Invalid eigenvector atom entry (expected length-3 list)")
        comps: List[complex] = []
        for c in a:
            # each component is [real, imag]
            if not (isinstance(c, list) and len(c) == 2):
                raise ValueError("Invalid eigenvector component (expected [re, im])")
            comps.append(complex(float(c[0]), float(c[1])))
        out.append(comps)
    return out


 
def _parse_floats(line: str, n: int) -> List[float]:
    """Parse at least n floats from a whitespace-delimited line.
    read_poscar helper.
    returns parsed floats.
    """

    toks = line.split()
    if len(toks) < n:
        raise ValueError(f"Expected at least {n} floats in line: '{line}'")
    return [float(toks[i]) for i in range(n)]


 
def _all_int(things: Sequence[str]) -> bool:
    if not things:
        return False
    for t in things:
        try:
            int(t)
        except Exception:
            return False
    return True


 
def discover_ml_band_paths(results_root: Union[str, Path]) -> List[str]:
    """Discover ML model band.yaml paths under the results directory. Since for this project 
    the band.yamls all exist at Project_dir/results/*model*/raw/plumipy_files/band.yaml
    however, it does this generally by a simple search for key file name 'band.yaml'. 

    Args:
        results_root (Union[str, Path]): Root directory containing model results.

    Returns:
        List[str]: Sorted list of band.yaml paths.
    """

    root = Path(results_root)
    if not root.exists():
        return []
    # Find all band.yaml files under results (each should correspond to a model)
    paths = [p for p in root.rglob("band.yaml") if p.is_file()]
    return [str(p) for p in sorted(paths)]


_PHONON_COUPLING_PATH_KEYS = {"contcar_gs", "contcar_es", "band_dft_path"}

_PHONON_COUPLING_BASE_DEFAULTS: dict[str, Any] = {
    "contcar_gs": "test/CBVN/CONTCAR_GS",
    "contcar_es": "test/CBVN/CONTCAR_ES",
    "band_dft_path": "test/CBVN/band.yaml",
    "q_tol": 1e-4,
    "lattice_tol": 1e-5,
    "threshold": 0.9,
    "freq_cluster_tol": 0.5,
    "freq_window": 0.5,
    "remove_mass_weighted_com": True,
    "gamma_only": True,
    "alpha": 1.3,
    "weight_kind": "S",
}


def _resolve_repo_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = value if isinstance(value, Path) else Path(value)
    return path if path.is_absolute() else root / path


def _resolve_band_ml_paths(value: Any, root: Path) -> list[Path]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        entries = [value]
    resolved: list[Path] = []
    for entry in entries:
        if entry is None:
            continue
        entry_path = Path(entry)
        resolved.append(entry_path if entry_path.is_absolute() else root / entry_path)
    return resolved


def _load_config(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / "config.yml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_phonon_coupling_defaults(repo_root: Path) -> dict[str, Any]:
    config_data = _load_config(repo_root)
    section = config_data.get("phonon_coupling", {}) or {}

    defaults: dict[str, Any] = {}
    for key, base_value in _PHONON_COUPLING_BASE_DEFAULTS.items():
        raw_value = section.get(key, base_value)
        if key in _PHONON_COUPLING_PATH_KEYS:
            resolved = _resolve_repo_path(repo_root, raw_value)
            defaults[key] = str(resolved) if resolved is not None else ""
            continue

        if isinstance(base_value, float):
            try:
                coerced = float(raw_value)
            except (TypeError, ValueError):
                coerced = base_value
            defaults[key] = coerced
        else:
            defaults[key] = raw_value

    defaults["band_ml_paths"] = _resolve_band_ml_paths(section.get("band_ml_paths"), repo_root)
    return defaults


def build_phonon_coupling_argparser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--contcar_gs", default=defaults["contcar_gs"])
    parser.add_argument("--contcar_es", default=defaults["contcar_es"])
    parser.add_argument("--band_dft", default=defaults["band_dft_path"])
    parser.add_argument("--band_ml", action="append", default=None, help="Repeatable: --band_ml path/to/band.yaml")
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument("--freq_cluster_tol", type=float, default=defaults["freq_cluster_tol"])
    parser.add_argument("--freq_window", type=float, default=defaults["freq_window"])
    parser.add_argument("--gamma_only", action="store_true", default=defaults["gamma_only"])
    parser.add_argument("--alpha", type=float, default=defaults["alpha"])
    parser.add_argument(
        "--weight_kind",
        default=defaults["weight_kind"],
        choices=["p", "S", "lambda"],
        help="DFT per-mode weight: p=|a|^2, S=|ω||a|^2, lambda=|ω|^2|a|^2",
    )
    return parser
