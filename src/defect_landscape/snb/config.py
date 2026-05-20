from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SnbDefaults:
    model_name: str = "mace-mpa-0-medium"
    results_root: Path = Path("resultsSNB")
    models_root: Path = Path("assets/models")
    bulk: Path | None = None
    defect: Path | None = None
    snb_dir: Path | None = None
    oxidation_states: dict[str, int | float] = field(default_factory=dict)
    vasp_inputs_dir: Path | None = None
    prepare_dft: bool = False
    copy_potcar: bool = False
    include_vdw: bool = True
    overwrite: bool = False
    device: str = "cuda"
    dtype: str = "float32"


@dataclass(frozen=True)
class SnbSettings:
    fmax: float = 0.03
    max_steps: int = 600
    energy_window_eV: float = 0.50
    max_clusters_per_model: int = 10
    matcher_ltol: float = 0.2
    matcher_stol: float = 0.3
    matcher_angle_tol: float = 5.0


@dataclass(frozen=True)
class SnbConfig:
    config_path: Path
    run_root: Path
    defaults: SnbDefaults
    settings: SnbSettings


def resolve_config_path(config: str | Path | None, repo_root: Path = REPO_ROOT) -> Path:
    if config is not None:
        path = Path(config)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    cwd_config = Path.cwd() / "config.yml"
    if cwd_config.exists():
        return cwd_config.resolve()

    return (repo_root / "config.yml").resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(run_root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else (run_root / path).resolve()


def load_config(config_path: str | Path | None = None, repo_root: Path = REPO_ROOT) -> SnbConfig:
    path = resolve_config_path(config_path, repo_root=repo_root)
    raw = load_yaml(path)
    run_root = path.parent

    snb = raw.get("snb", {}) or {}
    defaults_raw = snb.get("defaults", {}) or {}
    settings_raw = snb.get("settings", {}) or {}

    defaults = SnbDefaults(
        model_name=str(defaults_raw.get("model_name") or "mace-mpa-0-medium"),
        results_root=resolve_path(run_root, defaults_raw.get("results_root")) or (run_root / "resultsSNB"),
        models_root=resolve_path(run_root, defaults_raw.get("models_root")) or (run_root / "assets/models"),
        bulk=resolve_path(run_root, defaults_raw.get("bulk")),
        defect=resolve_path(run_root, defaults_raw.get("defect")),
        snb_dir=resolve_path(run_root, defaults_raw.get("snb_dir")),
        oxidation_states=parse_oxidation_states(defaults_raw.get("oxidation_states")),
        vasp_inputs_dir=resolve_path(run_root, defaults_raw.get("vasp_inputs_dir")),
        prepare_dft=bool(defaults_raw.get("prepare_dft", False)),
        copy_potcar=bool(defaults_raw.get("copy_potcar", False)),
        include_vdw=bool(defaults_raw.get("include_vdw", True)),
        overwrite=bool(defaults_raw.get("overwrite", False)),
        device=str(defaults_raw.get("device") or "cuda"),
        dtype=str(defaults_raw.get("dtype") or "float32"),
    )

    settings = SnbSettings(
        fmax=float(settings_raw.get("fmax", 0.03)),
        max_steps=int(settings_raw.get("max_steps", 600)),
        energy_window_eV=float(settings_raw.get("energy_window_eV", 0.50)),
        max_clusters_per_model=int(settings_raw.get("max_clusters_per_model", 10)),
        matcher_ltol=float(settings_raw.get("matcher_ltol", 0.2)),
        matcher_stol=float(settings_raw.get("matcher_stol", 0.3)),
        matcher_angle_tol=float(settings_raw.get("matcher_angle_tol", 5.0)),
    )

    return SnbConfig(config_path=path, run_root=run_root, defaults=defaults, settings=settings)


def _parse_number(value: Any) -> int | float:
    val = float(value)
    return int(val) if val.is_integer() else val


def parse_oxidation_states(items: list[str] | dict[str, Any] | None) -> dict[str, int | float]:
    if isinstance(items, dict):
        return {str(element): _parse_number(value) for element, value in items.items() if str(element)}

    states: dict[str, int | float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected oxidation state as Element=value, got {item!r}")
        element, value = item.split("=", 1)
        element = element.strip()
        if not element:
            raise ValueError(f"Missing element in oxidation state {item!r}")
        states[element] = _parse_number(value)
    return states
