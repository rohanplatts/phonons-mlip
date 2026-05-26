from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WorkflowContext:
    config_path: Path
    config: dict[str, Any]
    defaults: dict[str, Any]
    workflow: dict[str, Any]


def _resolve_config_path(config: str | Path | None, inputs: str | Path | None) -> Path:
    if config is not None:
        return Path(config).expanduser().resolve()
    if inputs is not None:
        root = Path(inputs).expanduser().resolve()
        if root.is_file():
            return root
        candidate = root / "config.yml"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Missing config.yml in input directory: {root}")
    raise FileNotFoundError("Pass --config PATH or --inputs PATH.")


def workflow_context(workflow_name: str, *, config: str | Path | None = None, inputs: str | Path | None = None) -> WorkflowContext:
    config_path = _resolve_config_path(config, inputs)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = config_data.get("defaults", {}) or {}
    workflows = config_data.get("workflows", {}) or {}
    workflow = workflows.get(workflow_name, {}) or {}
    return WorkflowContext(
        config_path=config_path,
        config=config_data,
        defaults=defaults,
        workflow=workflow,
    )

