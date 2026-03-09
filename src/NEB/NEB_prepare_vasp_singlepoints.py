#!/usr/bin/env python3
"""
Prepare VASP single-point inputs for MLIP path images (vasp_ci or vasp_guess).

Copies INCAR/KPOINTS/POTCAR into each image folder and optionally writes a
simple run script to execute VASP in each image directory.
"""

from __future__ import annotations

from pathlib import Path

import argparse

from NEB.neb_tools.neb_parsers import copy_vasp_inputs, image_dirs, load_yaml, resolve_config_path, resolve_path


def _model_dirs(results_root: Path, model_name: str | None) -> list[Path]:
    if model_name is None:
        return sorted([p for p in results_root.iterdir() if p.is_dir()])
    return [results_root / model_name]


def _resolve_inputs_dir(dft_neb_dat: Path | None, vasp_inputs_dir: Path | None) -> Path | None:
    if vasp_inputs_dir is not None:
        return vasp_inputs_dir
    if dft_neb_dat is None:
        return None
    if dft_neb_dat.is_dir():
        return dft_neb_dat
    return dft_neb_dat.parent


def _resolve_vasp_dir(raw_dir: Path, path_choice: str) -> Path:
    if path_choice in ("vasp_guess", "vasp_mlip_d3"):
        return raw_dir / "vasp_mlip_d3"
    if path_choice == "vasp_ci":
        return raw_dir / "vasp_ci"
    return raw_dir / path_choice


def _write_run_script(vasp_dir: Path, run_cmd: str) -> Path:
    script_path = vasp_dir / "run_vasp_singlepoints.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "\n"
        "for d in 0*/; do\n"
        "  (cd \"$d\" && " + run_cmd + ")\n"
        "done\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return script_path


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_path = resolve_config_path(pre_args.config)
    config = load_yaml(config_path)
    run_root = config_path.parent

    neb_cfg = config.get("neb", {}) or {}
    neb_defaults_cfg = neb_cfg.get("defaults", {}) or {}

    default_results_root = resolve_path(run_root, neb_defaults_cfg.get("results_root")) or (
        run_root / "resultsNEB"
    )
    default_dft_neb_dat = resolve_path(run_root, neb_defaults_cfg.get("dft_neb_dat"))
    if default_dft_neb_dat is None:
        structures_dir = resolve_path(run_root, neb_defaults_cfg.get("structures_dir")) or (
            run_root / "assets" / "structures" / "NEB"
        )
        candidate = structures_dir / "neb.dat"
        default_dft_neb_dat = candidate if candidate.exists() else None

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--results-root", type=Path, default=default_results_root)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--path",
        type=str,
        default="vasp_ci",
        help="Which path to prepare: vasp_ci or vasp_guess (alias: vasp_mlip_d3).",
    )
    parser.add_argument(
        "--vasp-inputs-dir",
        type=Path,
        default=None,
        help="Directory containing INCAR/KPOINTS/POTCAR to copy into each image folder.",
    )
    parser.add_argument(
        "--dft-neb-dat",
        type=Path,
        default=default_dft_neb_dat,
        help="Used as a fallback to locate VASP inputs (uses parent directory).",
    )
    parser.add_argument(
        "--run-cmd",
        type=str,
        default="vasp_std",
        help="Command to run VASP in each image directory (written into run script).",
    )
    parser.add_argument(
        "--write-run-script",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a run_vasp_singlepoints.sh helper into the prepared folder.",
    )
    args = parser.parse_args(argv)

    results_root = args.results_root
    if not results_root.exists():
        raise SystemExit(f"Missing results root: {results_root}")

    inputs_dir = _resolve_inputs_dir(args.dft_neb_dat, args.vasp_inputs_dir)
    if inputs_dir is None:
        raise SystemExit("Missing VASP inputs dir; provide --vasp-inputs-dir or --dft-neb-dat.")
    inputs_dir = inputs_dir.expanduser().resolve()
    if not inputs_dir.exists():
        raise SystemExit(f"Missing VASP inputs dir: {inputs_dir}")

    for model_dir in _model_dirs(results_root, args.model):
        model = model_dir.name
        raw_dir = model_dir / "raw"
        vasp_dir = _resolve_vasp_dir(raw_dir, args.path)
        if not vasp_dir.exists():
            print(f"[{model}] skip (missing vasp dir): {vasp_dir}")
            continue

        img_dirs = image_dirs(vasp_dir)
        for img_dir in img_dirs:
            copy_vasp_inputs(inputs_dir, img_dir)

        if args.write_run_script:
            script_path = _write_run_script(vasp_dir, args.run_cmd)
            print(f"[{model}] wrote {script_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
