from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mlip_workflows.cli import _looks_like_help, _print_help


def main(argv: list[str] | None = None) -> int:
    if _looks_like_help(argv):
        _print_help("mlip-phonons")
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inputs")
    parser.add_argument("--outputs")
    parser.add_argument("--config")
    args, remaining = parser.parse_known_args(argv)

    input_root = Path(args.inputs).expanduser().resolve() if args.inputs else None
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if config_path is None:
        if input_root is None:
            raise FileNotFoundError("Pass --inputs PATH or --config PATH.")
        config_path = input_root / "config.yml"
    if input_root is None:
        input_root = config_path.parent

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    workflow = (config.get("workflows", {}) or {}).get("phonons", {}) or {}
    model_name = workflow.get("model_name") or (config.get("defaults", {}) or {}).get("model_name") or "model"
    structure_name = workflow.get("structure") or workflow.get("structure_name") or "structure"
    unitcell_path = workflow.get("unitcell_path") or "POSCAR"
    primitive_path = workflow.get("primitive_cell_path") or workflow.get("primitive_path") or "primitive.poscar"
    output_root = Path(args.outputs).expanduser().resolve() if args.outputs else (input_root / "results").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "models": {model_name: {}},
        "structures": {
            "pure": {
                structure_name: {
                    "unitcell_path": str(unitcell_path),
                    "primitive_cell_path": str(primitive_path),
                }
            }
        },
        "workflows": {
            "phonons": {
                "structure": structure_name,
            }
        },
    }
    (output_root / "resolved-config.yml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")

    from mlip_phonons.main import main as _main

    forwarded = ["--config", str(output_root / "resolved-config.yml"), "--structure", structure_name, *remaining]
    return _main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
