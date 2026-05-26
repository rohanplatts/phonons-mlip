from __future__ import annotations

import argparse
from pathlib import Path

from coupling_modes.coup_tools.phon_analysis import run
from coupling_modes.coup_tools.phon_parsers import (
    build_phonon_coupling_argparser,
    discover_ml_band_paths,
    load_phonon_coupling_defaults,
)
from coupling_modes.coup_tools.phon_plot import render_report


def _resolve_input_root(proj_root: Path, inputs: Path | None, config: Path | None) -> Path:
    if config is not None:
        config_path = Path(config).expanduser().resolve()
        return config_path.parent if config_path.is_file() else config_path
    if inputs is not None:
        input_path = Path(inputs).expanduser().resolve()
        return input_path.parent if input_path.is_file() else input_path
    return proj_root


def main(argv: list[str] | None = None) -> int:
    proj_root = Path(__file__).resolve().parent.parent.parent
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--inputs", type=Path, default=None)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_parser.add_argument("--outputs", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)

    input_root = _resolve_input_root(proj_root, pre_args.inputs, pre_args.config)
    results_root = (Path(pre_args.outputs).expanduser().resolve() if pre_args.outputs is not None else (input_root / "results").resolve())
    defaults = load_phonon_coupling_defaults(input_root)
    auto_ml_paths = discover_ml_band_paths(results_root)
    if not auto_ml_paths and results_root != input_root:
        auto_ml_paths = discover_ml_band_paths(input_root)
    if not defaults["band_ml_paths"]:
        defaults["band_ml_paths"] = auto_ml_paths

    parser = build_phonon_coupling_argparser(defaults)
    parser.set_defaults(outputs=results_root)
    args = parser.parse_args(remaining)
    band_ml_paths = args.band_ml if args.band_ml is not None else defaults["band_ml_paths"]
    if not band_ml_paths:
        raise ValueError(f"No ML band.yaml files found under {results_root}")

    out = run(
        contcar_gs=args.contcar_gs,
        contcar_es=args.contcar_es,
        band_dft_path=args.band_dft,
        band_ml_paths=band_ml_paths,
        q_tol=defaults["q_tol"],
        lattice_tol=defaults["lattice_tol"],
        threshold=float(args.threshold),
        freq_cluster_tol=float(args.freq_cluster_tol),
        freq_window=float(args.freq_window),
        remove_mass_weighted_com=defaults["remove_mass_weighted_com"],
        gamma_only=bool(args.gamma_only),
        alpha=float(args.alpha),
        weight_kind=str(args.weight_kind),
    )

    report = render_report(
        out,
        threshold=float(args.threshold),
        freq_cluster_tol=float(args.freq_cluster_tol),
        freq_window=float(args.freq_window),
        alpha=float(args.alpha),
        weight_kind=str(args.weight_kind),
    )
    print(report)

    # Also save the report to disk
    out_dir = Path(args.outputs).expanduser().resolve() if args.outputs is not None else (proj_root / "resultsPhonCoupling")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = "phonon_coupling_report"
    i = 0
    while True:
        out_path = out_dir / f"{base}_{i}.txt"
        if not out_path.exists():
            break
        i += 1
    out_path.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
