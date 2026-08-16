"""Translate a conventional VASP NEB directory into the NEB config schema.

This module is deliberately limited to input discovery and validation.  It
does not read intermediate coordinates into the MLIP path or alter the NEB
engine; the existing runner remains responsible for interpolation and all
scientific stages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_DIRECTIVE_RE = re.compile(
    r"^\s*[#!]\s*MLIP_(WORKFLOW|MODEL)\s*=\s*(.*?)\s*$",
    re.IGNORECASE,
)
_DIRECTIVE_PREFIX_RE = re.compile(r"^\s*[#!]\s*MLIP_(WORKFLOW|MODEL)\b", re.IGNORECASE)
_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_NUMERIC_FOLDER_RE = re.compile(r"\d+")


def _parse_incar(path: Path) -> tuple[int, str | None]:
    """Read the small, explicitly supported subset of a VASP INCAR."""
    images: int | None = None
    model_name: str | None = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read VASP INCAR: {path}") from exc

    for line_number, line in enumerate(lines, start=1):
        directive_match = _DIRECTIVE_RE.match(line)
        if directive_match:
            key = directive_match.group(1).upper()
            value = directive_match.group(2).strip()
            if key == "WORKFLOW":
                if value.upper() != "NEB":
                    raise ValueError(
                        f"{path}:{line_number}: MLIP_WORKFLOW must be NEB, got {value!r}"
                    )
            elif key == "MODEL":
                if not value:
                    raise ValueError(f"{path}:{line_number}: MLIP_MODEL cannot be empty")
                if model_name is not None:
                    raise ValueError(f"{path}:{line_number}: repeated MLIP_MODEL directive")
                model_name = value
            continue

        stripped = line.strip()
        if stripped.startswith(("#", "!")) and _DIRECTIVE_PREFIX_RE.match(line):
            raise ValueError(f"{path}:{line_number}: malformed MLIP directive")
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue

        # VASP accepts semicolon-separated assignments.  A # or ! starts an
        # ordinary inline comment; directives are handled only as full lines.
        for assignment in re.split(r"[;]", re.split(r"[#!]", line, maxsplit=1)[0]):
            match = _ASSIGNMENT_RE.match(assignment)
            if not match:
                continue
            key, value = match.groups()
            if key.upper() != "IMAGES":
                continue
            value = value.strip()
            if images is not None:
                raise ValueError(f"{path}:{line_number}: repeated IMAGES assignment")
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: IMAGES must be an integer, got {value!r}"
                ) from exc
            if parsed < 1:
                raise ValueError(
                    f"{path}:{line_number}: IMAGES must be at least 1, got {parsed}"
                )
            images = parsed

    if images is None:
        raise ValueError(f"VASP INCAR is missing required IMAGES assignment: {path}")
    return images, model_name


def _validate_image_folders(vasp_dir: Path, intermediate_images: int) -> list[Path]:
    last_index = intermediate_images + 1
    expected_names = [f"{index:02d}" for index in range(last_index + 1)]
    observed_names = sorted(
        [
            entry.name
            for entry in vasp_dir.iterdir()
            if entry.is_dir() and _NUMERIC_FOLDER_RE.fullmatch(entry.name)
        ],
        key=int,
    )
    if observed_names != expected_names:
        raise ValueError(
            "VASP NEB image folders must be exactly contiguous; "
            f"expected {expected_names}, observed {observed_names}"
        )

    image_paths: list[Path] = []
    for name in expected_names:
        poscar = vasp_dir / name / "POSCAR"
        if not poscar.is_file():
            raise ValueError(f"Missing POSCAR for VASP NEB image {name}: {poscar}")
        image_paths.append(poscar.resolve())
    return image_paths


def translate_vasp_neb_directory(vasp_dir: str | Path) -> dict[str, Any]:
    """Return an in-memory MLIP-Workflows config for a VASP NEB directory."""
    root = Path(vasp_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"VASP NEB directory does not exist: {root}")
    incar = root / "INCAR"
    if not incar.is_file():
        raise ValueError(f"VASP NEB directory is missing INCAR: {incar}")

    intermediate_images, model_name = _parse_incar(incar)
    image_paths = _validate_image_folders(root, intermediate_images)

    workflow: dict[str, Any] = {
        "poscar_i": str(image_paths[0]),
        "poscar_f": str(image_paths[-1]),
        "n_images": intermediate_images + 2,
        "vasp_inputs_dir": str(root),
    }
    if model_name is not None:
        workflow["model_name"] = model_name
    return {"workflows": {"neb": workflow}}


def run_vasp_neb_directory(
    vasp_dir: str | Path,
    *,
    repo_root: Path | None = None,
) -> int:
    """Translate and run a VASP NEB directory without writing config.yml."""
    from NEB.run_neb_raw_v2 import run_neb_from_config

    root = Path(vasp_dir).expanduser().resolve()
    config = translate_vasp_neb_directory(root)
    return run_neb_from_config(config, run_root=root, repo_root=repo_root)
