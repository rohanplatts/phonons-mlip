from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing config file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {path}")
    return payload


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        raise SystemExit(f"Missing or invalid '{key}' section in config.yml")
    return section


def _require(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section or section[key] in (None, ""):
        raise SystemExit(f"Missing {section_name}.{key} in config.yml")
    return section[key]


def _resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute() or base is None:
        return path
    return (base / path).resolve()


def _ensure_mace_family(config: dict[str, Any]) -> None:
    family = config.get("family")
    if family not in (None, "mace"):
        raise SystemExit(f"Unsupported family for this workflow: {family}")


def _load_mace_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    resolved = config_path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    config = _load_yaml(resolved)
    _ensure_mace_family(config)
    return config, resolved


def _neb_rules_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("neb_data_set_synth")
    if not isinstance(section, dict):
        raise SystemExit("Missing or invalid 'neb_data_set_synth' section in config.yml")
    return section


def add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Path to the MACE workflow config.yml.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print the planned conversion, but do not write files.")


def run_convert(args) -> int:
    config, config_file = _load_mace_config(Path(args.config))
    neb_rules = _neb_rules_section(config)

    rules_path = _resolve_path(_require(neb_rules, "neb_data_set_synth", "rules"), base=config_file.parent)
    if rules_path is None:
        raise SystemExit("Could not resolve neb_data_set_synth.rules")

    repo_root = Path(__file__).resolve().parents[2]

    forwarded = [
        sys.executable,
        "-m",
        "fine_tuning.cli",
        "mace",
        "--curate-neb",
        "--inputs",
        str(rules_path),
    ]

    if args.dry_run:
        print(f"config: {config_file}")
        print("command:")
        print(" ".join(forwarded))
        print(f"cwd: {repo_root}")
        return 0

    completed = subprocess.run(forwarded, check=False, cwd=repo_root)
    return int(completed.returncode)
