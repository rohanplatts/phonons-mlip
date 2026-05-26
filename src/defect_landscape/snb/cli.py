from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .cluster import available_models, cluster_model
from .compare import compare_dft
from .config import SnbConfig, load_config, load_yaml, parse_oxidation_states
from .dft_parse import check_dft
from .generate import generate_snb_inputs
from .io import import_snb_inputs, manifest_path
from .relax import relax_model
from .report import write_report
from .select import select_clusters
from .vasp import prepare_dft
from common.benchmarking import maybe_fan_out


KNOWN_COMMANDS = {
    "generate",
    "run",
    "relax",
    "cluster",
    "select",
    "prepare-dft",
    "check-dft",
    "compare-dft",
    "report",
}

GLOBAL_OPTIONS_WITH_VALUES = {"--config", "--models-root", "--results-root", "--device", "--dtype"}
GLOBAL_BOOLEAN_OPTIONS = {"--include-vdw", "--no-include-vdw", "--overwrite", "--no-overwrite"}


def _normalise_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["run"]
    insert_at = 0
    while insert_at < len(argv):
        token = argv[insert_at]
        if token in {"-h", "--help"}:
            return argv
        if any(token.startswith(f"{option}=") for option in GLOBAL_OPTIONS_WITH_VALUES):
            insert_at += 1
            continue
        if token in GLOBAL_OPTIONS_WITH_VALUES:
            insert_at += 2
            continue
        if token in GLOBAL_BOOLEAN_OPTIONS:
            insert_at += 1
            continue
        break
    if insert_at >= len(argv):
        return [*argv, "run"]
    if insert_at < len(argv):
        token = argv[insert_at]
        if not token.startswith("-") and token not in KNOWN_COMMANDS:
            return [*argv[:insert_at], "run", *argv[insert_at:]]
    return argv


normalise_argv = _normalise_argv


def _preload_config(argv: list[str]) -> SnbConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--inputs")
    args, _ = parser.parse_known_args(argv)
    return load_config(args.config, inputs=args.inputs)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=argparse.SUPPRESS, help="Path to config.yml.")
    parser.add_argument("--inputs", default=argparse.SUPPRESS, help="Path to an input directory containing config.yml.")
    parser.add_argument("--models-root", type=Path, default=argparse.SUPPRESS, help="Root folder containing MLIP models.")
    parser.add_argument("--results-root", type=Path, default=argparse.SUPPRESS, help="Root folder for resultsSNB cases.")
    parser.add_argument("--device", default=argparse.SUPPRESS, help="Calculator device, for example cuda or cpu.")
    parser.add_argument("--dtype", default=argparse.SUPPRESS, help="Calculator dtype, for example float32.")
    parser.add_argument(
        "--include-vdw",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Use dispersion-corrected calculator path when the model supports it.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Overwrite existing stage outputs instead of resuming.",
    )


def _add_case_flags(parser: argparse.ArgumentParser, *, need_model: bool = False) -> None:
    if need_model:
        parser.add_argument("model_name", nargs="?", help="Model name accepted by common.get_calc.get_calc_object().")
    else:
        parser.add_argument("model_name", nargs="?", help="Optional model name.")
    parser.add_argument("--case-name", help="Case label. Defaults to the SnB or defect folder name where possible.")


def _add_input_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snb-dir", type=Path, help="Existing ShakeNBreak output/input directory containing candidate POSCARs.")
    parser.add_argument("--bulk", type=Path, help="Bulk POSCAR used by ShakeNBreak generation.")
    parser.add_argument("--defect", type=Path, help="Defect POSCAR used by ShakeNBreak generation.")
    parser.add_argument("--oxidation-states", nargs="*", default=None, help="Oxidation states as Element=value, for example Cs=1 Pb=2 I=-1.")


def build_parser(config: SnbConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlip-snb",
        description="MLIP-accelerated ShakeNBreak relaxation, clustering, DFT export, and validation.",
    )
    _add_shared_flags(parser)
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Generate ShakeNBreak distorted structures.")
    _add_shared_flags(generate)
    generate.add_argument("--bulk", type=Path, required=True)
    generate.add_argument("--defect", type=Path, required=True)
    generate.add_argument("--case-name", help="Optional output label. Defaults to the defect POSCAR parent folder.")
    generate.add_argument("--oxidation-states", nargs="+", help="Oxidation states as Element=value. Defaults to snb.defaults.oxidation_states.")

    run = subparsers.add_parser("run", help="Import/generate SnB inputs, relax, cluster, select, and optionally export DFT folders.")
    _add_shared_flags(run)
    run.add_argument("model_name", nargs="?", help="Model name accepted by common.get_calc.get_calc_object().")
    run.add_argument("--case-name")
    _add_input_flags(run)
    run.add_argument("--fmax", type=float)
    run.add_argument("--max-steps", type=int)
    run.add_argument("--threads", type=int, default=16)
    run.add_argument("--energy-window", type=float)
    run.add_argument("--max-clusters", type=int)
    run.add_argument("--top-k-clusters", type=int)
    run.add_argument("--always-include-ground", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--union-across-models", action="store_true")
    run.add_argument("--prepare-dft", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    run.add_argument("--vasp-inputs-dir", type=Path)
    run.add_argument("--copy-potcar", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    relax = subparsers.add_parser("relax", help="Relax imported SnB candidates with one MLIP.")
    _add_shared_flags(relax)
    _add_case_flags(relax, need_model=True)
    relax.add_argument("--fmax", type=float)
    relax.add_argument("--max-steps", type=int)
    relax.add_argument("--threads", type=int, default=16)

    cluster = subparsers.add_parser("cluster", help="Cluster MLIP-relaxed structures.")
    _add_shared_flags(cluster)
    _add_case_flags(cluster)
    cluster.add_argument("--matcher-ltol", type=float)
    cluster.add_argument("--matcher-stol", type=float)
    cluster.add_argument("--matcher-angle-tol", type=float)

    select = subparsers.add_parser("select", help="Select one representative per MLIP cluster for DFT validation.")
    _add_shared_flags(select)
    _add_case_flags(select)
    select.add_argument("--energy-window", type=float)
    select.add_argument("--max-clusters", type=int)
    select.add_argument("--top-k-clusters", type=int)
    select.add_argument("--always-include-ground", action=argparse.BooleanOptionalAction, default=True)
    select.add_argument("--union-across-models", action="store_true")

    dft = subparsers.add_parser("prepare-dft", help="Create VASP-ready DFT refinement folders from selected clusters.")
    _add_shared_flags(dft)
    dft.add_argument("--case-name", help="Optional output label. If omitted, inferred from config or a single existing results case.")
    dft.add_argument("--vasp-inputs-dir", type=Path)
    dft.add_argument("--copy-potcar", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    check = subparsers.add_parser("check-dft", help="Parse completed or partial DFT validation folders.")
    _add_shared_flags(check)
    check.add_argument("--case-name", help="Optional output label. If omitted, inferred from config or a single existing results case.")

    compare = subparsers.add_parser("compare-dft", help="Compare completed DFT refinements to MLIP-selected clusters.")
    _add_shared_flags(compare)
    compare.add_argument("--case-name", help="Optional output label. If omitted, inferred from config or a single existing results case.")
    compare.add_argument("--matcher-ltol", type=float)
    compare.add_argument("--matcher-stol", type=float)
    compare.add_argument("--matcher-angle-tol", type=float)

    report = subparsers.add_parser("report", help="Write a human-readable report for a case.")
    _add_shared_flags(report)
    report.add_argument("--case-name", help="Optional output label. If omitted, inferred from config or a single existing results case.")

    return parser


def _value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _case_label_from_path(path: str | Path) -> str:
    path = Path(path)
    if path.name.upper() in {"POSCAR", "CONTCAR"}:
        parent = path.parent.name
        return parent if parent and parent != "." else path.stem
    return path.name or path.stem


def _single_existing_case(results_root: str | Path | None) -> str | None:
    if results_root is None:
        return None
    root = Path(results_root)
    if not root.exists():
        return None
    cases = sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "analysis").exists())
    return cases[0] if len(cases) == 1 else None


def _case_name(args: argparse.Namespace, config: SnbConfig, results_root: str | Path | None = None) -> str:
    if getattr(args, "case_name", None):
        return args.case_name
    if getattr(args, "snb_dir", None):
        return _case_label_from_path(args.snb_dir)
    if getattr(args, "defect", None):
        return _case_label_from_path(args.defect)
    if config.defaults.snb_dir:
        return _case_label_from_path(config.defaults.snb_dir)
    if config.defaults.defect:
        return _case_label_from_path(config.defaults.defect)
    existing = _single_existing_case(results_root)
    if existing:
        return existing
    return "snb_case"


def _model_name(args: argparse.Namespace, config: SnbConfig) -> str:
    return getattr(args, "model_name", None) or config.defaults.model_name


def _results_root(args: argparse.Namespace, config: SnbConfig) -> Path:
    return Path(_value(args, "results_root", config.defaults.results_root)).resolve()


def _models_root(args: argparse.Namespace, config: SnbConfig) -> Path:
    return Path(_value(args, "models_root", config.defaults.models_root)).resolve()


def _ensure_inputs(args: argparse.Namespace, config: SnbConfig, *, case_name: str, results_root: Path) -> None:
    bulk = getattr(args, "bulk", None) or config.defaults.bulk
    defect = getattr(args, "defect", None) or config.defaults.defect
    snb_dir = getattr(args, "snb_dir", None) or config.defaults.snb_dir
    overwrite = bool(_value(args, "overwrite", config.defaults.overwrite))

    if bulk and defect:
        states = parse_oxidation_states(getattr(args, "oxidation_states", None) or config.defaults.oxidation_states)
        if not states:
            raise ValueError(
                "Oxidation states are required when generating SnB inputs from --bulk and --defect. "
                "Pass --oxidation-states Cs=1 Pb=2 I=-1 or set snb.defaults.oxidation_states in config.yml."
            )
        path = generate_snb_inputs(
            bulk=bulk,
            defect=defect,
            oxidation_states=states,
            results_root=results_root,
            case_name=case_name,
            overwrite=overwrite,
        )
        print(f"Wrote candidate manifest: {path}")
        return

    if snb_dir:
        path = import_snb_inputs(snb_dir=snb_dir, results_root=results_root, case_name=case_name, overwrite=overwrite)
        print(f"Wrote candidate manifest: {path}")
        return

    existing = manifest_path(results_root, case_name)
    if not existing.exists():
        raise FileNotFoundError("Provide --snb-dir, or provide --bulk/--defect/--oxidation-states, or run generate first.")


def dispatch(args: argparse.Namespace, config: SnbConfig) -> Path | list[Path] | None:
    results_root = _results_root(args, config)
    overwrite = bool(_value(args, "overwrite", config.defaults.overwrite))

    if args.command == "generate":
        case_name = _case_name(args, config, results_root)
        states = parse_oxidation_states(args.oxidation_states or config.defaults.oxidation_states)
        if not states:
            raise ValueError(
                "Oxidation states are required for SnB generation. "
                "Pass --oxidation-states Cs=1 Pb=2 I=-1 or set snb.defaults.oxidation_states in config.yml."
            )
        path = generate_snb_inputs(
            bulk=args.bulk,
            defect=args.defect,
            oxidation_states=states,
            results_root=results_root,
            case_name=case_name,
            overwrite=overwrite,
        )
        print(f"Wrote candidate manifest: {path}")
        return path

    if args.command == "run":
        case_name = _case_name(args, config, results_root)
        model_name = _model_name(args, config)
        _ensure_inputs(args, config, case_name=case_name, results_root=results_root)
        relaxation = relax_model(
            model_name=model_name,
            results_root=results_root,
            case_name=case_name,
            models_root=_models_root(args, config),
            device=_value(args, "device", config.defaults.device),
            dtype=_value(args, "dtype", config.defaults.dtype),
            include_vdw=bool(_value(args, "include_vdw", config.defaults.include_vdw)),
            fmax=float(args.fmax if args.fmax is not None else config.settings.fmax),
            max_steps=int(args.max_steps if args.max_steps is not None else config.settings.max_steps),
            overwrite=overwrite,
            threads=args.threads,
        )
        print(f"Wrote relaxation summary: {relaxation}")
        clustered = cluster_model(
            model_name=model_name,
            results_root=results_root,
            case_name=case_name,
            matcher_ltol=config.settings.matcher_ltol,
            matcher_stol=config.settings.matcher_stol,
            matcher_angle_tol=config.settings.matcher_angle_tol,
        )
        print(f"Wrote clusters: {clustered}")
        selected = select_clusters(
            results_root=results_root,
            case_name=case_name,
            model_name=model_name,
            energy_window_eV=float(args.energy_window if args.energy_window is not None else config.settings.energy_window_eV),
            max_clusters=int(args.max_clusters if args.max_clusters is not None else config.settings.max_clusters_per_model),
            top_k_clusters=args.top_k_clusters,
            always_include_ground=args.always_include_ground,
            union_across_models=args.union_across_models,
            overwrite=overwrite,
        )
        print(f"Wrote DFT selection manifest: {selected}")
        if bool(_value(args, "prepare_dft", config.defaults.prepare_dft)):
            dft = prepare_dft(
                results_root=results_root,
                case_name=case_name,
                vasp_inputs_dir=args.vasp_inputs_dir or config.defaults.vasp_inputs_dir,
                copy_potcar=bool(_value(args, "copy_potcar", config.defaults.copy_potcar)),
                overwrite=overwrite,
            )
            print(f"Wrote DFT validation manifest: {dft}")
        report = write_report(results_root=results_root, case_name=case_name)
        print(f"Wrote report: {report}")
        return report

    if args.command == "relax":
        case_name = _case_name(args, config, results_root)
        model_name = _model_name(args, config)
        path = relax_model(
            model_name=model_name,
            results_root=results_root,
            case_name=case_name,
            models_root=_models_root(args, config),
            device=_value(args, "device", config.defaults.device),
            dtype=_value(args, "dtype", config.defaults.dtype),
            include_vdw=bool(_value(args, "include_vdw", config.defaults.include_vdw)),
            fmax=float(args.fmax if args.fmax is not None else config.settings.fmax),
            max_steps=int(args.max_steps if args.max_steps is not None else config.settings.max_steps),
            overwrite=overwrite,
            threads=args.threads,
        )
        print(f"Wrote relaxation summary: {path}")
        return path

    if args.command == "cluster":
        case_name = _case_name(args, config, results_root)
        models = [_model_name(args, config)] if args.model_name else available_models(results_root, case_name)
        paths = [
            cluster_model(
                model_name=model,
                results_root=results_root,
                case_name=case_name,
                matcher_ltol=float(args.matcher_ltol if args.matcher_ltol is not None else config.settings.matcher_ltol),
                matcher_stol=float(args.matcher_stol if args.matcher_stol is not None else config.settings.matcher_stol),
                matcher_angle_tol=float(args.matcher_angle_tol if args.matcher_angle_tol is not None else config.settings.matcher_angle_tol),
            )
            for model in models
        ]
        for path in paths:
            print(f"Wrote clusters: {path}")
        return paths

    if args.command == "select":
        case_name = _case_name(args, config, results_root)
        path = select_clusters(
            results_root=results_root,
            case_name=case_name,
            model_name=args.model_name,
            energy_window_eV=float(args.energy_window if args.energy_window is not None else config.settings.energy_window_eV),
            max_clusters=int(args.max_clusters if args.max_clusters is not None else config.settings.max_clusters_per_model),
            top_k_clusters=args.top_k_clusters,
            always_include_ground=args.always_include_ground,
            union_across_models=args.union_across_models,
            overwrite=overwrite,
        )
        print(f"Wrote DFT selection manifest: {path}")
        return path

    if args.command == "prepare-dft":
        case_name = _case_name(args, config, results_root)
        path = prepare_dft(
            results_root=results_root,
            case_name=case_name,
            vasp_inputs_dir=args.vasp_inputs_dir or config.defaults.vasp_inputs_dir,
            copy_potcar=bool(_value(args, "copy_potcar", config.defaults.copy_potcar)),
            overwrite=overwrite,
        )
        print(f"Wrote DFT validation manifest: {path}")
        return path

    if args.command == "check-dft":
        case_name = _case_name(args, config, results_root)
        path = check_dft(results_root=results_root, case_name=case_name)
        print(f"Wrote DFT status: {path}")
        return path

    if args.command == "compare-dft":
        case_name = _case_name(args, config, results_root)
        path = compare_dft(
            results_root=results_root,
            case_name=case_name,
            matcher_ltol=float(args.matcher_ltol if args.matcher_ltol is not None else config.settings.matcher_ltol),
            matcher_stol=float(args.matcher_stol if args.matcher_stol is not None else config.settings.matcher_stol),
            matcher_angle_tol=float(args.matcher_angle_tol if args.matcher_angle_tol is not None else config.settings.matcher_angle_tol),
        )
        report = write_report(results_root=results_root, case_name=case_name)
        print(f"Wrote comparison: {path}")
        print(f"Wrote report: {report}")
        return path

    if args.command == "report":
        case_name = _case_name(args, config, results_root)
        path = write_report(results_root=results_root, case_name=case_name)
        print(f"Wrote report: {path}")
        return path

    return None


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config = _preload_config(raw_argv)
    fanout_rc = maybe_fan_out("snb", config_path=config.config_path, config=load_yaml(config.config_path))
    if fanout_rc is not None:
        return fanout_rc
    argv2 = _normalise_argv(raw_argv)
    parser = build_parser(config)
    args = parser.parse_args(argv2)
    if args.command is None:
        parser.print_help()
        return 0
    dispatch(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
