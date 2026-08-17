"""Small, scheduler-agnostic handoff for generated VASP NEB inputs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


KNOWN_SUBMISSION_SCRIPT = "vasp_bunya.sh"


def find_known_submission_script(directory: Path) -> Path | None:
    """Return the explicitly supported submission script, if present."""
    candidate = Path(directory).expanduser().resolve() / KNOWN_SUBMISSION_SCRIPT
    return candidate if candidate.is_file() else None


def copy_known_submission_script(source_dir: Path, destination_dir: Path) -> Path | None:
    source = find_known_submission_script(source_dir)
    if source is None:
        return None
    destination = Path(destination_dir).expanduser().resolve() / source.name
    if source != destination:
        shutil.copy2(source, destination)
    return destination


def submit_vasp_ci(vasp_ci_dir: Path) -> None:
    """Submit a copied known script, or explain why manual submission is needed."""
    ci_dir = Path(vasp_ci_dir).expanduser().resolve()
    script = find_known_submission_script(ci_dir)
    print(f"VASP-ready inputs: {ci_dir}")
    if script is None:
        print(f"No unambiguous submission script found in {ci_dir}; submit manually if needed.")
        return

    sbatch = shutil.which("sbatch")
    if sbatch is None:
        print(f"sbatch is unavailable. Submission script: {script}")
        return

    command = [sbatch, f"--chdir={ci_dir}", str(script)]
    try:
        completed = subprocess.run(
            command,
            cwd=ci_dir,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        print(f"Could not submit {script}: {exc}")
        return

    if completed.returncode == 0:
        output = (completed.stdout or completed.stderr or "").strip()
        print(f"Submitted VASP refinement job from {ci_dir}. {output}".rstrip())
    else:
        detail = (completed.stderr or completed.stdout or "").strip()
        print(f"sbatch rejected {script} (exit {completed.returncode}). {detail}".rstrip())
