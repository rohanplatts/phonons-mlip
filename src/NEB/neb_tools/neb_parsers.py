from __future__ import annotations

from pathlib import Path
import gzip
import re
import shutil
import numpy as np
import yaml

from ase.io import read, write


_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-0-9.]+)")
_E0_RE = re.compile(r"energy\s+without\s+entropy\s*=\s*([-0-9.]+)")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def resolve_config_path(config_path: Path | None, *, repo_root: Path | None = None) -> Path:
    if config_path is not None:
        resolved = config_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Missing config file: {resolved}")
        return resolved

    candidates = [(Path.cwd() / "config.yml").expanduser().resolve()]
    resolved_repo_root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    repo_candidate = (resolved_repo_root / "config.yml").resolve()
    if repo_candidate not in candidates:
        candidates.append(repo_candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing config.yml. Checked: {checked}. Use --config to specify one explicitly.")


def image_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()])


def read_text(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as handle:
        return handle.read()


def parse_outcar_energy(outcar: Path, *, key: str) -> float:
    if not outcar.exists():
        raise FileNotFoundError(outcar)
    text = read_text(outcar)
    lines = text.splitlines()
    if key == "toten":
        for line in reversed(lines):
            match = _TOTEN_RE.search(line)
            if match:
                return float(match.group(1))
    elif key == "e0":
        for line in reversed(lines):
            match = _E0_RE.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"Could not parse {key} energy from {outcar}")


def collect_outcar_energies(root: Path, *, key: str) -> list[float]:
    energies: list[float] = []
    for img_dir in image_dirs(root):
        outcar = img_dir / "OUTCAR"
        if not outcar.exists():
            outcar = img_dir / "OUTCAR.gz"
        energies.append(parse_outcar_energy(outcar, key=key))
    if not energies:
        raise ValueError(f"No image OUTCARs found under {root}")
    return energies


def parse_last_outcar_forces(outcar: Path, n_atoms: int) -> np.ndarray:
    """
    Parse the last 'POSITION  TOTAL-FORCE (eV/Angst)' table from an OUTCAR(/.gz).
    Returns forces as (n_atoms, 3) float array in eV/Ang.
    """
    hdr = "POSITION                                       TOTAL-FORCE (eV/Angst)"
    lines = read_text(outcar).splitlines()
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if hdr in lines[i]:
            idx = i
            break
    if idx is None:
        raise ValueError(f"Could not find force table header in {outcar}")

    j = idx + 2  # skip dashed line
    forces = np.empty((n_atoms, 3), dtype=float)
    for a in range(n_atoms):
        parts = lines[j + a].split()
        if len(parts) < 6:
            raise ValueError(f"Unexpected force table line in {outcar}: {lines[j+a]!r}")
        forces[a, :] = [float(parts[3]), float(parts[4]), float(parts[5])]
    return forces


def load_poscar_forces_from_dft_run(dft_root: Path) -> tuple[list[np.ndarray], list[Path]]:
    """
    Load per-image DFT forces from a VASP NEB run directory.
    Expects image folders 00,01,... each containing OUTCAR or OUTCAR.gz and a POSCAR/CONTCAR.
    """
    forces: list[np.ndarray] = []
    poscars: list[Path] = []
    for img_dir in image_dirs(dft_root):
        poscar = img_dir / "POSCAR"
        if not poscar.exists():
            contcar = img_dir / "CONTCAR"
            if contcar.exists():
                poscar = contcar
        if not poscar.exists():
            continue

        outcar = img_dir / "OUTCAR"
        if not outcar.exists():
            outcar = img_dir / "OUTCAR.gz"
        if not outcar.exists():
            continue

        atoms = read(poscar)
        f = parse_last_outcar_forces(outcar, n_atoms=len(atoms))
        forces.append(f)
        poscars.append(poscar)

    if not forces:
        raise ValueError(f"No DFT NEB images parsed under {dft_root}")
    return forces, poscars


def load_s_mlip(npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    return np.asarray(data["s_mlip"], dtype=float)


def write_neb_dat(out_path: Path, s_mlip: np.ndarray, energies: np.ndarray) -> None:
    if len(energies) != len(s_mlip):
        raise ValueError(
            f"Image count mismatch: s_mlip has {len(s_mlip)} entries, energies has {len(energies)}"
        )
    e_rel = energies - energies[0]
    data = np.column_stack([np.arange(len(s_mlip), dtype=int), s_mlip, e_rel])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, data, fmt=["%d", "%.8f", "%.8f"])


def read_endpoints(poscar_i: Path, poscar_f: Path):
    a = read(poscar_i)
    b = read(poscar_f)
    return a, b


def choose_n_images(path: Path | None, fallback: int, cli_n_images: int | None = None) -> int:
    if cli_n_images is not None:
        if int(cli_n_images) < 3:
            raise ValueError("n_images must be >= 3")
        return int(cli_n_images)

    if path is None:
        return int(fallback)

    try:
        data = np.loadtxt(path)
        n = int(np.atleast_2d(data).shape[0])
        return max(n, 3)
    except Exception:
        return int(fallback)


def write_vasp_neb_images(images, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img_dir = outdir / f"{i:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        write(img_dir / "POSCAR", img, format="vasp", direct=True, vasp5=True)
    return outdir


def copy_vasp_inputs(src_dir: Path, dest_dir: Path) -> None:
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)


def export_vasp_neb_paths(
    *,
    out_raw: Path,
    images_mlip_d3,
    images_ci,
    vasp_inputs_dir: Path | None = None,
) -> tuple[Path, Path]:
    mlip_d3_dir = write_vasp_neb_images(images_mlip_d3, out_raw / "vasp_mlip_d3")
    ci_dir = write_vasp_neb_images(images_ci, out_raw / "vasp_ci")
    if vasp_inputs_dir is not None:
        copy_vasp_inputs(vasp_inputs_dir, mlip_d3_dir)
        copy_vasp_inputs(vasp_inputs_dir, ci_dir)
    return mlip_d3_dir, ci_dir


def write_neb_npz(
    out_raw: Path,
    *,
    s_mlip: np.ndarray,
    e_mlip: np.ndarray,
    n_images: int,
    dft_neb_dat: Path | None,
    poscar_i: Path,
    poscar_f: Path,
    vasp_mlip_d3_dir: Path,
    vasp_ci_dir: Path,
) -> Path:
    out_path = out_raw / "neb_raw.npz"
    np.savez_compressed(
        out_path,
        s_mlip=s_mlip,
        e_mlip=e_mlip,
        n_images=np.array([n_images], dtype=int),
        dft_neb_dat="" if dft_neb_dat is None else str(dft_neb_dat),
        poscar_i=str(poscar_i),
        poscar_f=str(poscar_f),
        vasp_mlip_d3_dir=str(vasp_mlip_d3_dir),
        vasp_guess_dir=str(vasp_mlip_d3_dir),
        vasp_ci_dir=str(vasp_ci_dir),
    )
    return out_path


def write_neb_summary(
    out_raw: Path,
    *,
    model_name: str,
    n_images: int,
    barrier: float,
    delta_e: float,
    dft_neb_dat: Path | None,
    vasp_mlip_d3_dir: Path,
    vasp_ci_dir: Path,
) -> Path:
    summary_path = out_raw / "summary.txt"
    summary_path.write_text(
        f"model={model_name}\n"
        f"n_images={n_images}\n"
        f"barrier_eV={barrier:.6f}\n"
        f"deltaE_eV={delta_e:.6f}\n"
        f"dft_neb_dat={'None' if dft_neb_dat is None else str(dft_neb_dat)}\n"
        f"vasp_mlip_d3_dir={vasp_mlip_d3_dir}\n"
        f"vasp_guess_dir={vasp_mlip_d3_dir}\n"
        f"vasp_ci_dir={vasp_ci_dir}\n"
        f"out_raw={out_raw}\n",
        encoding="utf-8",
    )
    return summary_path
