from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .io import analysis_dir, case_root, copy_structure, read_csv_rows, write_csv_rows


DFT_MANIFEST_FIELDS = [
    "dft_id",
    "case_name",
    "selection_id",
    "model_name",
    "model_label",
    "cluster_id",
    "best_candidate_id",
    "best_mlip_energy_eV",
    "best_mlip_dE_eV",
    "selected_poscar",
    "representative_contcar",
    "dft_dir",
]


TEMPLATE_FILES = ["INCAR", "KPOINTS", "POTCAR", "submit.sh"]


def _link_or_copy(src: Path, dst: Path, *, copy: bool, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare_dft(
    *,
    results_root: str | Path,
    case_name: str,
    vasp_inputs_dir: str | Path | None = None,
    copy_potcar: bool = False,
    overwrite: bool = False,
) -> Path:
    selected_path = analysis_dir(results_root, case_name) / "selected_for_dft_manifest.csv"
    selected = read_csv_rows(selected_path)
    if not selected:
        raise FileNotFoundError(f"No selected clusters found: {selected_path}")

    template_dir = Path(vasp_inputs_dir).resolve() if vasp_inputs_dir else None
    root = case_root(results_root, case_name)
    rows: list[dict[str, Any]] = []
    for row in selected:
        dft_id = row["selection_id"]
        dft_dir = root / "dft_validate" / dft_id
        dft_dir.mkdir(parents=True, exist_ok=True)
        copy_structure(row["selected_poscar"], dft_dir / "POSCAR", overwrite=overwrite or not (dft_dir / "POSCAR").exists())

        if template_dir:
            for name in TEMPLATE_FILES:
                src = template_dir / name
                if not src.exists():
                    continue
                if name == "POTCAR":
                    _link_or_copy(src, dft_dir / name, copy=copy_potcar, overwrite=overwrite)
                else:
                    if not (dft_dir / name).exists() or overwrite:
                        shutil.copy2(src, dft_dir / name)

        rows.append(
            {
                "dft_id": dft_id,
                "case_name": case_name,
                "selection_id": row["selection_id"],
                "model_name": row["model_name"],
                "model_label": row["model_label"],
                "cluster_id": row["cluster_id"],
                "best_candidate_id": row["best_candidate_id"],
                "best_mlip_energy_eV": row["best_mlip_energy_eV"],
                "best_mlip_dE_eV": row["best_mlip_dE_eV"],
                "selected_poscar": row["selected_poscar"],
                "representative_contcar": row["representative_contcar"],
                "dft_dir": str(dft_dir),
            }
        )

    out = analysis_dir(results_root, case_name) / "dft_validation_manifest.csv"
    write_csv_rows(out, rows, DFT_MANIFEST_FIELDS)
    return out
