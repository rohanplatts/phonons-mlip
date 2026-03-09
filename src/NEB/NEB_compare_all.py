#!/usr/bin/env python3
"""
collecrt MLIP NEB raw results + compare to DFT neb.dat.

Reads:
  - DFT: assets/structures/NEB/neb.dat
  - MLIP: NEBresults/<model>/raw/neb_raw.npz

Writes:
  - NEBresults/<model>/plot/mep_compare.png
  - NEBresults/<model>/plot/report.md + report.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import numpy as np
import sys
import os
import yaml

from NEB.neb_tools.neb_parsers import (
    load_poscar_forces_from_dft_run,
    resolve_config_path,
    load_yaml,
    resolve_path,
)

def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path(repo_root: Path) -> None:
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _format_hms(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def log_timing_stats(log_path: Path) -> tuple[float, float | None, int]:
    """
    Parse an ASE optimizer log and return (total_seconds, avg_dt_seconds, n_steps_parsed).

    Handles a "Time" token formatted as:
      - time-of-day HH:MM:SS (optionally fractional seconds), or
      - elapsed seconds (float/int).
    """
    times: list[float] = []
    has_tod = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        step_i = None
        for i, p in enumerate(parts):
            if p.isdigit():
                step_i = i
                break
        if step_i is None or step_i + 1 >= len(parts):
            continue
        tok = parts[step_i + 1]
        if ":" in tok:
            has_tod = True
            hms = tok.split(":")
            if len(hms) != 3:
                continue
            h, mm, ss = hms
            try:
                t = int(h) * 3600 + int(mm) * 60 + float(ss)
            except ValueError:
                continue
        else:
            try:
                t = float(tok)
            except ValueError:
                continue
        times.append(t)

    n = len(times)
    if n < 2:
        return 0.0, None, n

    unwrapped = times
    if has_tod:
        # If these are time-of-day stamps, unwrap across midnight.
        unwrapped = [times[0]]
        for t in times[1:]:
            if t < unwrapped[-1]:
                # Handle tiny backward jumps from log timestamp granularity; treat as "no time passed".
                # Only assume a real day-wrap if the jump is large (e.g. 23:59 -> 00:01).
                if (unwrapped[-1] - t) > 12 * 3600:
                    t += 86400.0
                else:
                    t = unwrapped[-1]
            unwrapped.append(t)

    total = float(unwrapped[-1] - unwrapped[0])
    avg_dt = total / float(n - 1)
    return total, avg_dt, n


def load_dft_neb_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Assume neb.dat columns are [index, s, e]
    Returns (dft_s, dft_e).
    """
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        raise ValueError(f"neb.dat needs >=3 columns [index,s,e], got shape={data.shape}")
    return data[:, 1].astype(float), data[:, 2].astype(float)


def _force_error_metrics(
    *,
    model_name: str,
    models_root: Path,
    dft_root: Path,
    include_vdw: bool,
) -> dict[str, float | None]:
    """
    Compute force error metrics by evaluating the MLIP forces on the DFT NEB image geometries
    and comparing to DFT OUTCAR forces.
    """
    from ase.io import read
    from mlip_phonons.get_calc import get_calc_object

    dft_forces, poscars = load_poscar_forces_from_dft_run(dft_root)

    device = os.environ.get("MLIP_DEVICE", "cpu")
    dtype = os.environ.get("MLIP_DTYPE", "float32")
    calc = get_calc_object(model_name, models_root=models_root, device=device, dtype=dtype, include_vdw=include_vdw)

    sq_sum = 0.0
    n_comp = 0
    max_err = 0.0
    for f_dft, poscar in zip(dft_forces, poscars):
        atoms = read(poscar)
        atoms.calc = calc
        f_mlip = atoms.get_forces()
        diff = f_mlip - f_dft
        sq_sum += float(np.sum(diff * diff))
        n_comp += diff.size
        max_err = max(max_err, float(np.linalg.norm(diff, axis=1).max()))

    rmse = float(np.sqrt(sq_sum / n_comp)) if n_comp else None
    return {
        "force_RMSE_eV_per_A": rmse,
        "max_force_err_eV_per_A": float(max_err) if n_comp else None,
    }


def _max_f_perp_from_mlip_path(
    *,
    model_name: str,
    models_root: Path,
    vasp_ci_dir: Path,
    include_vdw: bool,
) -> dict[str, float | None]:
    """
    Compute max perpendicular force along the final MLIP path, using a simple 3N-space tangent:
      t_i = R_{i+1} - R_{i-1}
      F_perp = F - proj_t(F)
    Returns max over interior images of max atom ||F_perp||.
    """
    from ase.io import read
    from mlip_phonons.get_calc import get_calc_object

    img_dirs = sorted([p for p in vasp_ci_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if len(img_dirs) < 3:
        return {"max_F_perp_eV_per_A": None}

    images = [read(p / "POSCAR") for p in img_dirs]
    n_atoms = len(images[0])
    if any(len(a) != n_atoms for a in images):
        return {"max_F_perp_eV_per_A": None}

    device = os.environ.get("MLIP_DEVICE", "cpu")
    dtype = os.environ.get("MLIP_DTYPE", "float32")
    calc = get_calc_object(model_name, models_root=models_root, device=device, dtype=dtype, include_vdw=include_vdw)

    R = np.stack([a.get_positions() for a in images], axis=0)  # (nimg, nat, 3)
    F = []
    for a in images:
        a.calc = calc
        F.append(a.get_forces())
    F = np.stack(F, axis=0)  # (nimg, nat, 3)

    max_f_perp = 0.0
    for i in range(1, len(images) - 1):
        t = (R[i + 1] - R[i - 1]).reshape(-1)
        tt = float(np.dot(t, t))
        if tt == 0.0:
            continue
        f = F[i].reshape(-1)
        alpha = float(np.dot(f, t) / tt)
        f_perp = (f - alpha * t).reshape(n_atoms, 3)
        max_f_perp = max(max_f_perp, float(np.linalg.norm(f_perp, axis=1).max()))

    return {"max_F_perp_eV_per_A": float(max_f_perp)}


speed_logs = {
    "neb_mlip.log": "Mlip dt",
    "neb_mlip_d3.log": "Mlip_d3 dt",
    "neb_ci.log": "mlip_d3 climb dt",
}


def collect_speed_metrics(raw_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    total_seconds = 0
    found = False

    for log_name, avg_key in speed_logs.items():
        log_path = raw_dir / log_name
        if not log_path.exists():
            metrics[avg_key] = None
            continue
        total, avg_dt, _ = log_timing_stats(log_path)
        metrics[avg_key] = avg_dt
        total_seconds += total
        found = True

    metrics["Total NEB time (s)"] = total_seconds if found else None
    return metrics


def render_report(outdir: Path, title: str, plot_path: Path, metrics: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    md = [f"# {title}\n\n", "## Metrics\n\n"]
    for k, v in metrics.items():
        md.append(f"- **{k}**: {v}\n")
    md.append("\n## Plot\n\n")
    md.append(f"![mep_compare]({plot_path.name})\n")
    (outdir / "report.md").write_text("".join(md), encoding="utf-8")

    html = []
    html.append("<html><head><meta charset='utf-8'>")
    html.append(f"<title>{title}</title></head><body>")
    html.append(f"<h1>{title}</h1>")
    html.append("<h2>Metrics</h2><ul>")
    for k, v in metrics.items():
        html.append(f"<li><b>{k}</b>: {v}</li>")
    html.append("</ul>")
    html.append("<h2>Plot</h2>")
    html.append(f"<img src='{plot_path.name}' style='max-width:100%;'>")
    html.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(html), encoding="utf-8")


def plot_compare(
    *,
    dft_s: np.ndarray,
    dft_e: np.ndarray,
    mlip_s: np.ndarray,
    mlip_e: np.ndarray,
    out_png: Path,
    title: str,
) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # DFT assumed already relative to initial 
    # MLIP e is relative to initial 
    plt.figure()
    plt.plot(mlip_s, mlip_e, "o-", label="MLIP (CI-NEB)")
    plt.plot(dft_s, dft_e, "o-", label="DFT (neb.dat)")
    plt.xlabel("Reaction coordinate [Å]")
    plt.ylabel("Energy [eV] relative to initial")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return {
        "mlip_barrier_eV": float(np.max(mlip_e)),
        "mlip_deltaE_eV": float(mlip_e[-1]),
        "dft_barrier_eV": float(np.max(dft_e)),
        "dft_deltaE_eV": float(dft_e[-1]),
    }


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    repo_root = _resolve_repo_root(repo_root)
    _ensure_src_on_path(repo_root)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_path = resolve_config_path(pre_args.config, repo_root=repo_root)
    config = load_yaml(config_path)
    run_root = config_path.parent

    neb_cfg = config.get("neb", {}) or {}
    neb_defaults_cfg = neb_cfg.get("defaults", {}) or {}

    default_results_root = resolve_path(
        run_root,
        neb_defaults_cfg.get("results_root") or os.environ.get("NEB_RESULTS_ROOT"),
    ) or (run_root / "resultsNEB")
    default_models_root = resolve_path(
        run_root,
        neb_defaults_cfg.get("models_root") or os.environ.get("MLIP_MODELS_ROOT"),
    ) or (run_root / "assets" / "models")
    default_dft_neb_dat = resolve_path(
        run_root,
        neb_defaults_cfg.get("dft_neb_dat") or os.environ.get("NEB_DFT_NEB_DAT"),
    )
    if default_dft_neb_dat is None:
        structures_dir = resolve_path(run_root, neb_defaults_cfg.get("structures_dir")) or (
            run_root / "assets" / "structures" / "NEB"
        )
        candidate = structures_dir / "neb.dat"
        default_dft_neb_dat = candidate if candidate.exists() else None
    default_include_vdw = bool(neb_defaults_cfg.get("include_vdw", True))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--results-root", type=Path, default=default_results_root)
    parser.add_argument("--models-root", type=Path, default=default_models_root)
    parser.add_argument("--dft-neb-dat", type=Path, default=default_dft_neb_dat)
    parser.add_argument(
        "--include-vdw",
        action=argparse.BooleanOptionalAction,
        default=default_include_vdw,
        help="Whether to include D3 corrections when evaluating MLIP forces.",
    )
    args = parser.parse_args(argv)

    nebresults_root = args.results_root
    dft_neb_dat = args.dft_neb_dat
    models_root = args.models_root
    if not nebresults_root.exists():
        print(f"No NEBresults directory found at: {nebresults_root}")
        return 0

    if dft_neb_dat is None or not dft_neb_dat.exists():
        raise SystemExit(f"Missing DFT reference file: {dft_neb_dat}")

    dft_s, dft_e = load_dft_neb_dat(dft_neb_dat)

    model_dirs = sorted([p for p in nebresults_root.iterdir() if p.is_dir()])
    if not model_dirs:
        print(f"No model folders found under {nebresults_root}")
        return 0

    all_metrics: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        model = model_dir.name
        raw_dir = model_dir / "raw"
        npz_path = raw_dir / "neb_raw.npz"
        if not npz_path.exists():
            print(f"[{model}] skip (no raw npz): {npz_path}")
            continue

        data = dict(np.load(npz_path, allow_pickle=True))
        mlip_s = np.asarray(data["s_mlip"], dtype=float)
        mlip_e = np.asarray(data["e_mlip"], dtype=float)

        plot_dir = model_dir / "plot"
        out_png = plot_dir / "mep_compare.png"

        title = f"DFT vs {model} NEB MEP"
        metrics = plot_compare(
            dft_s=dft_s,
            dft_e=dft_e,
            mlip_s=mlip_s,
            mlip_e=mlip_e,
            out_png=out_png,
            title=title,
        )
        metrics.update(collect_speed_metrics(raw_dir))

        # Optional path/force quality metrics (may require ase + model backend installed).
        # - max_F_perp_eV_per_A: from the saved MLIP final path (vasp_ci) if available.
        # - force_RMSE_eV_per_A/max_force_err_eV_per_A: MLIP vs DFT forces on DFT geometries, if dft_neb_dat points
        #   to a VASP NEB directory structure with image OUTCARs.
        include_vdw = bool(args.include_vdw)
        try:
            vasp_ci_dir = raw_dir / "vasp_ci"
            if vasp_ci_dir.exists():
                metrics.update(
                    _max_f_perp_from_mlip_path(
                        model_name=model,
                        models_root=models_root,
                        vasp_ci_dir=vasp_ci_dir,
                        include_vdw=include_vdw,
                    )
                )
            else:
                metrics["max_F_perp_eV_per_A"] = None
        except Exception:
            metrics["max_F_perp_eV_per_A"] = None

        try:
            dft_neb_dat_from_npz = str(data.get("dft_neb_dat", "")).strip()
            dft_neb_dat_path = args.dft_neb_dat
            if dft_neb_dat_path is None:
                dft_neb_dat_path = Path(dft_neb_dat_from_npz) if dft_neb_dat_from_npz else None
            if dft_neb_dat_path and dft_neb_dat_path.exists():
                metrics.update(
                    _force_error_metrics(
                        model_name=model,
                        models_root=models_root,
                        dft_root=dft_neb_dat_path.parent,
                        include_vdw=include_vdw,
                    )
                )
            else:
                metrics["force_RMSE_eV_per_A"] = None
                metrics["max_force_err_eV_per_A"] = None
        except Exception:
            metrics["force_RMSE_eV_per_A"] = None
            metrics["max_force_err_eV_per_A"] = None

        # include a couple of important fields
        metrics["model"] = model
        metrics["raw_dir"] = str(raw_dir)
        metrics["dft_neb_dat"] = str(dft_neb_dat)

        metrics_path = plot_dir / "metrics.yaml"
        metrics_path.write_text(
            yaml.safe_dump(metrics, sort_keys=False),
            encoding="utf-8",
        )

        render_report(plot_dir, title=title, plot_path=out_png, metrics=metrics)

        print(f"[{model}] wrote {out_png} + report.md/report.html in {plot_dir}")

    # Rank models after metrics are generated, using metrics.yaml files on disk
    for model_dir in model_dirs:
        metrics_path = model_dir / "plot" / "metrics.yaml"
        if metrics_path.exists():
            all_metrics.append(yaml.safe_load(metrics_path.read_text(encoding="utf-8")))

    if all_metrics:
        for m in all_metrics:
            m["barrier_abs_err_eV"] = abs(m["mlip_barrier_eV"] - m["dft_barrier_eV"])
            m["deltaE_abs_err_eV"] = abs(m["mlip_deltaE_eV"] - m["dft_deltaE_eV"])

        ranked = sorted(
            all_metrics,
            key=lambda m: (m["barrier_abs_err_eV"], m["deltaE_abs_err_eV"]),
        )

        rankings_dir = nebresults_root / "rankings"
        rankings_dir.mkdir(parents=True, exist_ok=True)

        def _fmt_opt(value: Any, *, digits: int = 6) -> str:
            if value is None:
                return "NA"
            if isinstance(value, (int, float)):
                return f"{value:.{digits}f}"
            return str(value)

        lines: list[str] = []
        lines.append("FINAL RANKING (lower barrier_abs_err_eV is better)\n")
        lines.append("Sorted by barrier_abs_err_eV, then deltaE_abs_err_eV.\n\n")
        columns = [
            ("rank", "right"),
            ("model", "left"),
            ("barrier_abs_err_eV", "right"),
            ("deltaE_abs_err_eV", "right"),
            ("mlip_barrier_eV", "right"),
            ("dft_barrier_eV", "right"),
            ("mlip_deltaE_eV", "right"),
            ("dft_deltaE_eV", "right"),
            ("force_RMSE_eV_per_A", "right"),
            ("max_force_err_eV_per_A", "right"),
            ("max_F_perp_eV_per_A", "right"),
            ("Total NEB time (s)", "right"),
            ("Mlip dt", "right"),
            ("Mlip_d3 dt", "right"),
            ("mlip_d3 climb dt", "right"),
        ]

        rows: list[list[str]] = []
        for i, m in enumerate(ranked, start=1):
            rows.append(
                [
                    str(i),
                    str(m["model"]),
                    f"{m['barrier_abs_err_eV']:.6f}",
                    f"{m['deltaE_abs_err_eV']:.6f}",
                    f"{m['mlip_barrier_eV']:.6f}",
                    f"{m['dft_barrier_eV']:.6f}",
                    f"{m['mlip_deltaE_eV']:.6f}",
                    f"{m['dft_deltaE_eV']:.6f}",
                    _fmt_opt(m.get("force_RMSE_eV_per_A")),
                    _fmt_opt(m.get("max_force_err_eV_per_A")),
                    _fmt_opt(m.get("max_F_perp_eV_per_A")),
                    _fmt_opt(_format_hms(m.get("Total NEB time (s)"))),
                    _fmt_opt(m.get("Mlip dt")),
                    _fmt_opt(m.get("Mlip_d3 dt")),
                    _fmt_opt(m.get("mlip_d3 climb dt")),
                ]
            )

        widths = [len(name) for name, _ in columns]
        for row in rows:
            for idx, value in enumerate(row):
                if len(value) > widths[idx]:
                    widths[idx] = len(value)

        header_cells = []
        for (name, align), width in zip(columns, widths):
            header_cells.append(f"{name:>{width}}" if align == "right" else f"{name:<{width}}")
        lines.append("  ".join(header_cells) + "\n")

        for row in rows:
            cells = []
            for (name, align), width, value in zip(columns, widths, row):
                cells.append(f"{value:>{width}}" if align == "right" else f"{value:<{width}}")
            lines.append("  ".join(cells) + "\n")
        (rankings_dir / "rankings.txt").write_text("".join(lines), encoding="utf-8")

        print(f"Wrote rankings to {rankings_dir / 'rankings.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
