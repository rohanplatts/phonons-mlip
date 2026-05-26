from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_MODELS_PATH = REPO_ROOT / "SUPPORTED_MODELS.yml"


@dataclass(frozen=True)
class WorkflowSpec:
    module: str
    config_flag: str


WORKFLOWS: dict[str, WorkflowSpec] = {
    "phonons": WorkflowSpec(module="mlip_phonons.main", config_flag="inputs"),
    "neb": WorkflowSpec(module="NEB.run_neb_raw_v2", config_flag="inputs"),
    "snb": WorkflowSpec(module="defect_landscape.snb.cli", config_flag="config"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def resolve_inputs_path(inputs: Path) -> Path:
    path = Path(inputs).expanduser().resolve()
    if path.is_dir():
        path = path / "config.yml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.yml: {path}")
    return path


def load_supported_models() -> dict[str, dict[str, Any]]:
    if not SUPPORTED_MODELS_PATH.exists():
        raise FileNotFoundError(f"Missing supported model registry: {SUPPORTED_MODELS_PATH}")
    raw = load_yaml(SUPPORTED_MODELS_PATH)
    models = raw.get("models", raw) or {}
    if not isinstance(models, dict):
        raise TypeError(f"Unsupported SUPPORTED_MODELS.yml format: {SUPPORTED_MODELS_PATH}")
    return models


def model_environment(model_name: str, supported_models: dict[str, dict[str, Any]]) -> str:
    entry = supported_models.get(model_name)
    if entry is None:
        raise KeyError(f"Model {model_name!r} is missing from {SUPPORTED_MODELS_PATH.name}")
    if not isinstance(entry, dict):
        raise TypeError(f"Model entry for {model_name!r} must be a mapping")
    environment = entry.get("environment")
    if not environment:
        raise ValueError(f"Model {model_name!r} has no environment in {SUPPORTED_MODELS_PATH.name}")
    return str(environment)


def model_names(config: dict[str, Any]) -> list[str]:
    defaults = config.get("defaults", {}) or {}
    value = defaults.get("model_name")
    if value is None:
        raise ValueError("defaults.model_name is required")
    if isinstance(value, list):
        names = [str(item) for item in value if str(item)]
    else:
        names = [str(value)]
    if not names:
        raise ValueError("defaults.model_name must contain at least one model")
    return names


def benchmark_root(config_path: Path, config: dict[str, Any]) -> Path:
    defaults = config.get("defaults", {}) or {}
    root_value = defaults.get("outputs_root", defaults.get("results_root", "results"))
    root = Path(root_value)
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def rewrite_config_for_model(config: dict[str, Any], model_name: str, *, config_path: Path) -> dict[str, Any]:
    rewritten = copy.deepcopy(config)
    defaults = dict(rewritten.get("defaults", {}) or {})
    defaults["model_name"] = model_name
    defaults["outputs_root"] = "."
    models_root = defaults.get("models_root")
    if models_root is not None:
        models_root_path = Path(models_root)
        if not models_root_path.is_absolute():
            models_root_path = (config_path.parent / models_root_path).resolve()
        defaults["models_root"] = str(models_root_path)
    rewritten["defaults"] = defaults
    return rewritten


def build_command(spec: WorkflowSpec, *, config_path: Path, environment: str) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "python",
        "-m",
        spec.module,
    ]
    if spec.config_flag == "inputs":
        command.extend(["--inputs", str(config_path.parent)])
    elif spec.config_flag == "config":
        command.extend(["--config", str(config_path)])
    else:
        raise ValueError(f"Unknown workflow config flag: {spec.config_flag!r}")
    return command


def run_workflow(spec: WorkflowSpec, *, config_path: Path, environment: str) -> int:
    command = build_command(spec, config_path=config_path, environment=environment)
    print(shlex.join(command))
    env = os.environ.copy()
    src_root = REPO_ROOT / "src"
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_root) if not pythonpath else f"{src_root}:{pythonpath}"
    completed = subprocess.run(command, cwd=config_path.parent, check=False, env=env)
    return int(completed.returncode)


def maybe_fan_out(workflow: str, *, config_path: Path, config: dict[str, Any]) -> int | None:
    names = model_names(config)
    if len(names) == 1:
        return None

    spec = WORKFLOWS[workflow]
    supported_models = load_supported_models()
    environments = {name: model_environment(name, supported_models) for name in names}

    root = benchmark_root(config_path, config)
    root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for name in names:
        model_dir = root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_config = model_dir / "config.yml"
        dump_yaml(model_config, rewrite_config_for_model(config, name, config_path=config_path))
        rc = run_workflow(spec, config_path=model_config, environment=environments[name])
        if rc != 0:
            failures.append((name, rc))

    if failures:
        for name, rc in failures:
            print(f"{name} failed with exit code {rc}")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="workflow-fanout", description="Launch a workflow for one or more models.")
    parser.add_argument("workflow", choices=sorted(WORKFLOWS))
    parser.add_argument("--inputs", type=Path, required=True, help="Path to the benchmark config.yml or its containing directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_inputs_path(args.inputs)
    config = load_yaml(config_path)
    rc = maybe_fan_out(args.workflow, config_path=config_path, config=config)
    return 0 if rc is None else rc


if __name__ == "__main__":
    raise SystemExit(main())
