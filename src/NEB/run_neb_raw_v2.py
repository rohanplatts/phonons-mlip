#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml

from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE
import os 

# this file lives in <repo>/src/NEB/run_neb_raw.py
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from NEB.neb_tools.neb_analysis import (
    LoopDetected,
    attach_loop_guard,
    build_images,
    energies_relative,
    map_final_to_initial_by_species,
    normalize_image_cells,
    reaction_coordinate,
)
from NEB.neb_tools.neb_classes import NEBDefaults, NEBInputs, NEBOutputDirs, NEBResults
from NEB.neb_tools.neb_parsers import (
    choose_n_images,
    export_vasp_neb_paths,
    read_endpoints,
    write_neb_npz,
    write_neb_summary,
    resolve_config_path,
)
from common.benchmarking import benchmark_root, maybe_fan_out, model_names
from NEB.neb_tools.benchmark_report import benchmark_results_ready, generate_family_benchmark_report


def _parse_args(
    argv: list[str] | None,
    *,
    default_config_path: Path,
    default_inputs: NEBInputs,
) -> NEBInputs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config_path)
    parser.add_argument("--inputs", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("model_name", nargs="?", default=default_inputs.model_name)
    parser.add_argument("--n-images", type=int, default=default_inputs.n_images)
    parser.add_argument("--poscar-i", type=Path, default=default_inputs.poscar_i)
    parser.add_argument("--poscar-f", type=Path, default=default_inputs.poscar_f)
    parser.add_argument("--dft-neb-dat", type=Path, default=default_inputs.dft_neb_dat)
    parser.add_argument("--models-root", type=Path, default=default_inputs.models_root)
    parser.add_argument("--results-root", type=Path, default=default_inputs.results_root)
    parser.add_argument(
        "--vasp-inputs-dir",
        type=Path,
        default=default_inputs.vasp_inputs_dir,
        help="Optional directory containing INCAR/KPOINTS/POTCAR to copy into VASP exports.",
    )
    parser.add_argument("--device", type=str, default=default_inputs.device)
    parser.add_argument("--dtype", type=str, default=default_inputs.dtype)
    parser.add_argument(
        "--relax-endpoints",
        action=argparse.BooleanOptionalAction,
        default=default_inputs.relax_endpoints,
        help="Relax the initial and final structures with MLIP before NEB.",
    )
    parser.add_argument(
        "--remap-f-i",
        action=argparse.BooleanOptionalAction,
        default=default_inputs.remap_f_i,
        help="Apply within-species Hungarian remapping to the final endpoint before interpolation. This will also save the remapped poscar_f in the same folder as poscar_f.",
    )
    parser.add_argument(
        "--include-vdw",
        action=argparse.BooleanOptionalAction,
        default=default_inputs.include_vdw,
        help="Include D3 (vdW) error corrections when evaluating MLIP forces.",
    )
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run NEB_compare_all for the results root instead of a NEB run.",
    )
    parser.add_argument(
        "--report-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate benchmark plots and README content after the benchmark results exist.",
    )
    
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=default_inputs.overwrite,
        help="Reuse (WARNING: if true it will overwrite previous results) the existing results directory instead of creating a new suffixed one.",
    )

    args = parser.parse_args(argv)

    return NEBInputs(
        model_name=str(args.model_name),
        n_images=args.n_images,
        poscar_i=args.poscar_i,
        poscar_f=args.poscar_f,
        dft_neb_dat=args.dft_neb_dat,
        relax_endpoints=args.relax_endpoints,
        remap_f_i=args.remap_f_i,
        include_vdw=args.include_vdw,
        compare=args.compare,
        report_benchmark=args.report_benchmark,
        overwrite=args.overwrite,
        models_root=args.models_root,
        results_root=args.results_root,
        vasp_inputs_dir=args.vasp_inputs_dir,
        device=str(args.device),
        dtype=str(args.dtype),
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config.yml: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def _build_neb_inputs_and_defaults(
    config: dict,
    *,
    run_root: Path,
    repo_root: Path,
) -> tuple[NEBInputs, NEBDefaults]:
    """Lower the existing raw NEB config schema into execution inputs/defaults.

    This is intentionally the boundary shared by YAML-backed and future
    non-YAML frontends. The merge order and fallback values mirror the legacy
    ``main`` implementation.
    """
    global_defaults = config.get("defaults", {}) or {}
    workflow_cfg = ((config.get("workflows", {}) or {}).get("neb", {}) or {})
    workflow_settings_cfg = workflow_cfg.get("settings", {}) or {}
    neb_cfg = config.get("neb", {}) or {}
    neb_defaults_cfg = {
        **global_defaults,
        **(neb_cfg.get("defaults", {}) or {}),
        **{key: value for key, value in workflow_cfg.items() if key != "settings"},
    }
    neb_settings_cfg = {
        **(neb_cfg.get("settings", {}) or {}),
        **(workflow_settings_cfg if isinstance(workflow_settings_cfg, dict) else {}),
    }

    structures_dir = _resolve_path(run_root, neb_defaults_cfg.get("structures_dir")) or (
        repo_root / "assets" / "structures" / "NEB"
    )
    default_poscar_i = _resolve_path(run_root, neb_defaults_cfg.get("poscar_i")) or (structures_dir / "POSCAR_i")
    default_poscar_f = _resolve_path(run_root, neb_defaults_cfg.get("poscar_f")) or (structures_dir / "POSCAR_f")
    default_dft_neb_dat = _resolve_path(run_root, neb_defaults_cfg.get("dft_neb_dat"))
    if default_dft_neb_dat is None:
        maybe = structures_dir / "neb.dat"
        default_dft_neb_dat = maybe if maybe.exists() else None

    default_models_root = _resolve_path(run_root, neb_defaults_cfg.get("models_root")) or (
        repo_root / "assets" / "models"
    )
    default_results_root = _resolve_path(
        run_root,
        neb_defaults_cfg.get("results_root") or neb_defaults_cfg.get("outputs_root"),
    ) or (run_root / "resultsNEB")
    configured_n_images = neb_defaults_cfg.get("n_images")
    default_inputs = NEBInputs(
        model_name=str(neb_defaults_cfg.get("model_name") or "ivac0_neb_ft"),
        n_images=None if configured_n_images is None else int(configured_n_images),
        poscar_i=default_poscar_i,
        poscar_f=default_poscar_f,
        dft_neb_dat=default_dft_neb_dat,
        relax_endpoints=bool(neb_defaults_cfg.get("relax_endpoints", True)),
        remap_f_i=bool(neb_defaults_cfg.get("remap_f_i", False)),
        include_vdw=bool(neb_defaults_cfg.get("include_vdw", True)),
        overwrite=bool(neb_defaults_cfg.get("overwrite", False)),
        models_root=default_models_root,
        results_root=default_results_root,
        vasp_inputs_dir=_resolve_path(run_root, neb_defaults_cfg.get("vasp_inputs_dir")),
        device=str(neb_defaults_cfg.get("device") or "cuda"),
        dtype=str(neb_defaults_cfg.get("dtype") or "float32"),
    )
    defaults = NEBDefaults(
        n_images_fallback=int(neb_settings_cfg.get("n_images_fallback", 9)),
        maxstep_mlip_guess=float(neb_settings_cfg.get("maxstep_mlip_guess", 0.05)),
        fmax_mlip_guess=float(neb_settings_cfg.get("fmax_mlip_guess", 0.03)),
        steps_mlip_guess=int(neb_settings_cfg.get("steps_mlip_guess", 3000)),
        k_spring_mlip=float(neb_settings_cfg.get("k_spring_mlip", 0.6)),
        k_spring=float(neb_settings_cfg.get("k_spring", 0.6)),
        maxstep_mlip_d3=float(neb_settings_cfg.get("maxstep_mlip_d3", 0.03)),
        fmax_mlip_d3=float(neb_settings_cfg.get("fmax_mlip_d3", 0.03)),
        steps_mlip_d3=int(neb_settings_cfg.get("steps_mlip_d3", 1400)),
        maxstep_ci=float(neb_settings_cfg.get("maxstep_ci", 0.03)),
        fmax_ci=float(neb_settings_cfg.get("fmax_ci", 0.03)),
        steps_ci=int(neb_settings_cfg.get("steps_ci", 1000)),
    )
    return default_inputs, defaults


def run_neb_from_config(
    config: dict,
    *,
    run_root: Path,
    repo_root: Path | None = None,
    inputs: NEBInputs | None = None,
) -> int:
    """Run the standard NEB engine from an already loaded config dictionary."""
    resolved_repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    default_inputs, defaults = _build_neb_inputs_and_defaults(
        config,
        run_root=Path(run_root),
        repo_root=resolved_repo_root,
    )
    return _run_neb(inputs if inputs is not None else default_inputs, defaults)


def _run_neb(args: NEBInputs, defaults: NEBDefaults) -> int:
    """Execute the unchanged scientific NEB workflow from resolved inputs."""
    from common.get_calc import get_calc_object
    from common.relax import relax

    model_name = str(args.model_name)
    results_root = args.results_root
    if results_root is None:
        raise ValueError("NEB results_root must be resolved before execution.")
    out_raw = (results_root / "raw").resolve()
    out_raw.mkdir(parents=True, exist_ok=True)

    a, b = read_endpoints(args.poscar_i, args.poscar_f)

    a_mlip = a.copy()
    b_mlip = b.copy()

    if len(a) != len(b):
        raise ValueError("Different atom counts between POSCAR_i and POSCAR_f.")
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        raise ValueError("Species/order differs between POSCAR_i and POSCAR_f.")
    if not np.allclose(a.cell.array, b.cell.array, atol=1e-8):
        raise ValueError("Cells differ between POSCAR_i and POSCAR_f.")

    device = args.device
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError:
            print("CUDA requested but torch is not installed; falling back to CPU.")
            device = "cpu"
        else:
            if not torch.cuda.is_available():
                print("CUDA requested but not available; falling back to CPU.")
                device = "cpu"
    dtype = args.dtype

    calc_mlip = get_calc_object(
        model_name,
        models_root=args.models_root,
        device=device,
        dtype=dtype,
        include_vdw=False,
    )

    calc_vdw = None
    if args.include_vdw:
        calc_vdw = get_calc_object(
            model_name,
            models_root=args.models_root,
            device=device,
            dtype=dtype,
            include_vdw=True,
        )

    if args.remap_f_i:
        b = map_final_to_initial_by_species(a, b)

    calc_relax = calc_vdw if args.include_vdw else calc_mlip
    a.calc = calc_relax
    b.calc = calc_relax
    a_relaxed = None
    b_relaxed = None

    if args.relax_endpoints:
        enable_cache = True
        if enable_cache:
            if os.path.exists(out_raw / "relaxed_traj_i.traj"):
                a_relaxed = read(out_raw / "relaxed_traj_i.traj", index=-1)
            if os.path.exists(out_raw / "relaxed_traj_f.traj"):
                b_relaxed = read(out_raw / "relaxed_traj_f.traj", index=-1)

        if not a_relaxed:
            a_relaxed = relax(a, outdir=out_raw, filename="relaxed_traj_i.traj")
        if not b_relaxed:
            b_relaxed = relax(b, outdir=out_raw, filename="relaxed_traj_f.traj")
    else:
        a_relaxed = a
        b_relaxed = b

    a = a_relaxed
    b = b_relaxed

    print("Choosing number of images and building the images")
    n_images = choose_n_images(args.dft_neb_dat, defaults.n_images_fallback, args.n_images)

    images = build_images(a, b, n_images)
    normalize_image_cells(images)

    for img in images:
        img.calc = calc_mlip

    neb = NEB(
        images,
        k=defaults.k_spring_mlip,
        climb=False,
        method="improvedtangent",
        allow_shared_calculator=True,
    )

    print("Interpolating middle images")
    neb.interpolate(method="idpp", mic=True)

    print("assigning opts1 (MLIP)")
    opts1 = FIRE(
        neb,
        trajectory=str(out_raw / "neb_mlip.traj"),
        logfile=str(out_raw / "neb_mlip.log"),
        maxstep=defaults.maxstep_mlip_guess,
    )
    print("running opts1")
    attach_loop_guard(opts1, label="opts1")
    try:
        opts1.run(fmax=defaults.fmax_mlip_guess, steps=defaults.steps_mlip_guess)
    except LoopDetected as exc:
        print(str(exc))

    if args.include_vdw:
        for img in images:
            img.calc = calc_vdw

        neb = NEB(
            images,
            k=defaults.k_spring,
            climb=False,
            method="improvedtangent",
            allow_shared_calculator=True,
        )

        print("assigning opts2 (Refining with D3)")
        opts2 = FIRE(
            neb,
            trajectory=str(out_raw / "neb_mlip_d3.traj"),
            logfile=str(out_raw / "neb_mlip_d3.log"),
            maxstep=defaults.maxstep_mlip_d3,
        )
        print("running opts2")
        attach_loop_guard(opts2, label="opts2")
        try:
            opts2.run(fmax=defaults.fmax_mlip_d3, steps=defaults.steps_mlip_d3)
        except LoopDetected as exc:
            print(str(exc))

    images_pre_ci = [img.copy() for img in images]

    neb_ci = NEB(
        images,
        k=defaults.k_spring,
        climb=True,
        method="improvedtangent",
        allow_shared_calculator=True,
    )
    print("assigning opt_ci")
    opt_ci = FIRE(
        neb_ci,
        trajectory=str(out_raw / "neb_ci.traj"),
        logfile=str(out_raw / "neb_ci.log"),
        maxstep=defaults.maxstep_ci,
    )

    print("running opt_ci")
    attach_loop_guard(opt_ci, label="opt_ci")
    try:
        opt_ci.run(fmax=defaults.fmax_ci, steps=defaults.steps_ci)
    except LoopDetected as exc:
        print(str(exc))

    vasp_mlip_d3_dir, vasp_ci_dir = export_vasp_neb_paths(
        out_raw=out_raw,
        images_mlip_d3=images_pre_ci,
        images_ci=[img.copy() for img in images],
        vasp_inputs_dir=args.vasp_inputs_dir,
    )
    output_dirs = NEBOutputDirs(
        out_raw=out_raw,
        vasp_mlip_d3_dir=vasp_mlip_d3_dir,
        vasp_ci_dir=vasp_ci_dir,
    )

    s_mlip = reaction_coordinate(images)
    e_mlip = energies_relative(images)

    results = NEBResults(
        s_mlip=s_mlip,
        e_mlip=e_mlip,
        barrier=float(np.max(e_mlip)),
        delta_e=float(e_mlip[-1]),
    )

    write_neb_npz(
        output_dirs.out_raw,
        s_mlip=results.s_mlip,
        e_mlip=results.e_mlip,
        n_images=n_images,
        dft_neb_dat=args.dft_neb_dat,
        poscar_i=args.poscar_i,
        poscar_f=args.poscar_f,
        vasp_mlip_d3_dir=output_dirs.vasp_mlip_d3_dir,
        vasp_ci_dir=output_dirs.vasp_ci_dir,
    )

    write_neb_summary(
        output_dirs.out_raw,
        model_name=model_name,
        n_images=n_images,
        barrier=results.barrier,
        delta_e=results.delta_e,
        dft_neb_dat=args.dft_neb_dat,
        vasp_mlip_d3_dir=output_dirs.vasp_mlip_d3_dir,
        vasp_ci_dir=output_dirs.vasp_ci_dir,
    )

    print(f"SUCCESS wrote raw NEB to {output_dirs.out_raw}")
    print(f"VASP pre-CI path written to {output_dirs.vasp_mlip_d3_dir}")
    print(f"VASP post-CI path written to {output_dirs.vasp_ci_dir}")
    return 0


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_parser.add_argument("--inputs", type=Path, default=None)
    pre_parser.add_argument("--vasp", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)

    if pre_args.vasp is not None:
        if pre_args.config is not None or pre_args.inputs is not None:
            pre_parser.error("--vasp cannot be combined with --config or --inputs")
        if remaining:
            pre_parser.error(
                "--vasp accepts no other run options; use the VASP directory "
                "INCAR directives and MLIP defaults"
            )
        from NEB.vasp_frontend import run_vasp_neb_directory

        return run_vasp_neb_directory(pre_args.vasp, repo_root=repo_root)

    config_path = resolve_config_path(pre_args.config, repo_root=repo_root, inputs=pre_args.inputs)

    config = _load_yaml(config_path)

    run_root = config_path.parent

    default_inputs, _ = _build_neb_inputs_and_defaults(
        config,
        run_root=run_root,
        repo_root=repo_root,
    )

    args = _parse_args(
        argv,
        default_config_path=config_path,
        default_inputs=default_inputs,
    )

    if args.report_benchmark:
        names = model_names(config)
        if len(names) < 2:
            raise ValueError(
                f"--report-benchmark requires at least two models in defaults.model_name, got {len(names)}"
            )
        results_root = benchmark_root(config_path, config)
        if not benchmark_results_ready(results_root, names):
            fanout_rc = maybe_fan_out("neb", config_path=config_path, config=config)
            if fanout_rc is not None and fanout_rc != 0:
                return fanout_rc
        generate_family_benchmark_report(config_path, config)
        return 0

    fanout_rc = maybe_fan_out("neb", config_path=config_path, config=config)
    if fanout_rc is not None:
        return fanout_rc

    if args.compare:
        from NEB.NEB_compare_all import main as compare_main

        compare_argv: list[str] = ["--config", str(config_path)]
        if args.results_root is not None:
            compare_argv.extend(["--results-root", str(args.results_root)])
        if args.models_root is not None:
            compare_argv.extend(["--models-root", str(args.models_root)])
        if args.dft_neb_dat is not None:
            compare_argv.extend(["--dft-neb-dat", str(args.dft_neb_dat)])
        compare_argv.append("--include-vdw" if args.include_vdw else "--no-include-vdw")
        return compare_main(compare_argv, repo_root=repo_root)
    return run_neb_from_config(
        config,
        run_root=run_root,
        repo_root=repo_root,
        inputs=args,
    )


if __name__ == "__main__":

    raise SystemExit(main())
