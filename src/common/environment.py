"""Shared model-to-Conda environment dispatch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_MODELS_PATH = REPO_ROOT / "SUPPORTED_MODELS.yml"
DISPATCH_GUARD = "MLIP_WORKFLOWS_ENV_DISPATCHED"


def load_supported_models(path: Path = SUPPORTED_MODELS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing supported model registry: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    models = raw.get("models", raw) or {}
    if not isinstance(models, dict):
        raise TypeError(f"Unsupported SUPPORTED_MODELS.yml format: {path}")
    return models


def required_environment(
    model_name: str,
    supported_models: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    registry_path: Path = SUPPORTED_MODELS_PATH,
) -> str:
    models = load_supported_models(registry_path) if supported_models is None else supported_models
    entry = models.get(model_name)
    if entry is None:
        raise KeyError(f"Model {model_name!r} is missing from {registry_path.name}")
    environment = entry.get("environment")
    if not environment:
        raise ValueError(f"Model {model_name!r} has no environment in {registry_path.name}")
    return str(environment)


def current_environment(env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    name = values.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    prefix = values.get("CONDA_PREFIX")
    return Path(prefix).name if prefix else None


def conda_command(environment: str, command: list[str]) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", environment, *command]


def run_in_environment(
    environment: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    model_name: str | None = None,
) -> int:
    full_command = conda_command(environment, command)
    child_env = os.environ.copy()
    child_env[DISPATCH_GUARD] = "1"
    src_root = REPO_ROOT / "src"
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = str(src_root) if not existing_pythonpath else f"{src_root}:{existing_pythonpath}"
    try:
        completed = subprocess.run(full_command, cwd=cwd, env=child_env, check=False)
    except FileNotFoundError as exc:
        name = f" for model {model_name!r}" if model_name else ""
        raise RuntimeError(
            f"Could not launch required Conda environment {environment!r}{name}: conda was not found"
        ) from exc
    if completed.returncode != 0:
        name = f" for model {model_name!r}" if model_name else ""
        print(
            f"Required Conda environment {environment!r}{name} failed with exit code "
            f"{completed.returncode}."
        )
    return int(completed.returncode)


def dispatch_if_needed(
    model_name: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    registry_path: Path = SUPPORTED_MODELS_PATH,
) -> int | None:
    environment = required_environment(model_name, registry_path=registry_path)
    values = os.environ if env is None else env
    current = current_environment(values)
    guarded = values.get(DISPATCH_GUARD) == "1"
    if current == environment:
        return None
    if guarded:
        raise RuntimeError(
            f"Environment dispatch guard is set, but model {model_name!r} requires "
            f"{environment!r}; current environment is {current or 'unknown'!r}."
        )
    return run_in_environment(environment, command, cwd=cwd, model_name=model_name)
