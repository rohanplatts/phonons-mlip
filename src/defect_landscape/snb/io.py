from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


CANDIDATE_FIELDS = [
    "case_name",
    "candidate_id",
    "source_poscar",
    "staged_poscar",
    "relative_path",
    "source_snb_dir",
]


def safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+%/-]+", "_", str(label)).strip("_/")
    cleaned = re.sub(r"/+", "/", cleaned)
    return cleaned or "candidate"


def safe_model_label(model_name: str) -> str:
    return safe_label(model_name).replace("/", "_")


def case_root(results_root: str | Path, case_name: str) -> Path:
    return Path(results_root).resolve() / safe_label(case_name)


def analysis_dir(results_root: str | Path, case_name: str) -> Path:
    return case_root(results_root, case_name) / "analysis"


def manifest_path(results_root: str | Path, case_name: str) -> Path:
    return analysis_dir(results_root, case_name) / "candidate_manifest.csv"


def ensure_case_dirs(results_root: str | Path, case_name: str) -> Path:
    root = case_root(results_root, case_name)
    for rel in ["snb_inputs", "mlip_relaxed", "analysis", "selected_for_dft", "dft_validate"]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def copy_structure(src: str | Path, dst: str | Path, overwrite: bool = True) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def discover_poscars(snb_dir: str | Path) -> list[Path]:
    root = Path(snb_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        if root.name != "POSCAR":
            raise ValueError(f"Expected a POSCAR file or SnB directory, got {root}")
        return [root]
    return sorted(path for path in root.rglob("POSCAR") if path.is_file())


def candidate_id_from_poscar(snb_dir: Path, poscar: Path, index: int) -> str:
    try:
        rel_parent = poscar.parent.relative_to(snb_dir)
    except ValueError:
        rel_parent = Path(f"candidate_{index:04d}")
    if str(rel_parent) == ".":
        return f"candidate_{index:04d}"
    return safe_label(rel_parent.as_posix())


def import_snb_inputs(
    *,
    snb_dir: str | Path,
    results_root: str | Path,
    case_name: str,
    overwrite: bool = False,
) -> Path:
    root = ensure_case_dirs(results_root, case_name)
    source_root = Path(snb_dir).resolve()
    staged_root = root / "snb_inputs"
    poscars = discover_poscars(source_root)
    if not poscars:
        raise RuntimeError(f"No POSCAR files found under {source_root}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, poscar in enumerate(poscars):
        candidate_id = candidate_id_from_poscar(source_root if source_root.is_dir() else poscar.parent, poscar, index)
        base_id = candidate_id
        suffix = 1
        while candidate_id in seen:
            suffix += 1
            candidate_id = f"{base_id}_{suffix}"
        seen.add(candidate_id)

        staged = staged_root / candidate_id / "POSCAR"
        copy_structure(poscar, staged, overwrite=True)
        rows.append(
            {
                "case_name": case_name,
                "candidate_id": candidate_id,
                "source_poscar": str(poscar),
                "staged_poscar": str(staged),
                "relative_path": candidate_id,
                "source_snb_dir": str(source_root),
            }
        )

    write_csv_rows(manifest_path(results_root, case_name), rows, CANDIDATE_FIELDS)
    write_json(
        analysis_dir(results_root, case_name) / "case_metadata.json",
        {
            "case_name": case_name,
            "source_snb_dir": str(source_root),
            "n_candidates": len(rows),
        },
    )
    return manifest_path(results_root, case_name)


def load_candidates(results_root: str | Path, case_name: str) -> list[dict[str, str]]:
    rows = read_csv_rows(manifest_path(results_root, case_name))
    if not rows:
        raise FileNotFoundError(f"No candidate manifest found: {manifest_path(results_root, case_name)}")
    return rows

