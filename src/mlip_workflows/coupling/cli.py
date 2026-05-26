from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mlip_workflows.cli import _looks_like_help, _print_help


def main(argv: list[str] | None = None) -> int:
    if _looks_like_help(argv):
        _print_help("mlip-coup")
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inputs")
    parser.add_argument("--config")
    parser.add_argument("--outputs")
    args, remaining = parser.parse_known_args(argv)

    input_root = Path(args.inputs).expanduser().resolve() if args.inputs else None
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if config_path is None:
        if input_root is None:
            raise FileNotFoundError("Pass --inputs PATH or --config PATH.")
        config_path = input_root / "config.yml"
    if input_root is None:
        input_root = config_path.parent
    output_root = Path(args.outputs).expanduser().resolve() if args.outputs else (input_root / "results").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    (output_root / "resolved-config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from coupling_modes.phonon_coupling import main as _main

    forwarded = ["--inputs", str(input_root), "--outputs", str(output_root), *remaining]
    return _main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
