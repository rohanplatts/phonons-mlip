from __future__ import annotations

import argparse
from pathlib import Path

from .neb_data_set_synth.siv_data import run_neb_curation
from .fine_tuning_scripts.orb.extxyz_to_orb_ase_db import convert_extxyz_to_db


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlip-ft")
    subparsers = parser.add_subparsers(dest="family", required=True)

    for family in ("mace", "orb", "petmad"):
        family_parser = subparsers.add_parser(family, help=f"{family.upper()} fine-tuning workflow helpers")
        family_parser.add_argument(
            "--curate-neb",
            action="store_true",
            help="Run the NEB curation pipeline before any family-specific post-processing.",
        )
        family_parser.add_argument(
            "--inputs",
            type=Path,
            required=True,
            help="Path to the siv_rules.yml file that defines the NEB curation inputs.",
        )

    return parser


def _orb_db_paths(out_dir: Path, prefix: str) -> list[tuple[Path, Path]]:
    return [
        (out_dir / f"{prefix}_train.extxyz", out_dir / f"{prefix}_train.db"),
        (out_dir / f"{prefix}_val.extxyz", out_dir / f"{prefix}_val.db"),
        (out_dir / f"{prefix}_test.extxyz", out_dir / f"{prefix}_test.db"),
    ]


def _convert_orb_outputs(config: dict[str, object]) -> Path:
    out_dir = Path(config["outputs"]["out_dir"])
    prefix = str(config["outputs"]["prefix"])
    pairs = _orb_db_paths(out_dir, prefix)

    for input_path, output_path in pairs:
        if not input_path.exists():
            raise SystemExit(f"Missing curated extxyz file: {input_path}")
        convert_extxyz_to_db(
            input_path,
            output_path,
            energy_key="REF_energy",
            forces_key="REF_forces",
            overwrite=True,
            charge=None,
            spin=None,
        )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.curate_neb:
        parser.error("--curate-neb is required")

    config = run_neb_curation(Path(args.inputs))
    out_dir = Path(config["outputs"]["out_dir"])
    prefix = str(config["outputs"]["prefix"])

    if args.family == "orb":
        _convert_orb_outputs(config)
        print(f"ORB databases written to {out_dir}")
        return 0

    if args.family in ("mace", "petmad"):
        print(f"Curated extxyz written to {out_dir} with prefix {prefix}")
        return 0

    parser.error(f"Unsupported family: {args.family}")
    return 2


if __name__ == "__main__":
    raise SystemExit("Use the `mlip-ft` console command instead of invoking this module directly.")
