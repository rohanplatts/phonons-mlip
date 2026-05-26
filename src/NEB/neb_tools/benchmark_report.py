from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.benchmarking import benchmark_root, model_names

from . import ces2, cps2


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_START = "<!-- MLIP_BENCHMARK_START -->"
BENCHMARK_END = "<!-- MLIP_BENCHMARK_END -->"


@dataclass(frozen=True)
class BenchmarkModelReport:
    name: str
    model_dir: Path
    plot_dir: Path
    energy_error_png: Path
    energy_profiles_png: Path
    path_fidelity_png: Path
    path_fidelity_json: Path
    report_json: Path
    report_md: Path


@dataclass(frozen=True)
class BenchmarkFamilyReport:
    plot_dir: Path
    energy_error_png: Path
    energy_profiles_png: Path
    path_fidelity_png: Path
    path_fidelity_json: Path
    report_json: Path
    report_md: Path


def benchmark_results_ready(results_root: Path, names: list[str]) -> bool:
    if len(names) < 2:
        return False
    for name in names:
        model_dir = results_root / name
        if not model_dir.is_dir():
            return False
        if not (model_dir / "raw" / "neb_raw.npz").exists():
            return False
        if not (model_dir / "raw" / "vasp_ci").exists():
            return False
    return True


def _relative_to_repo(path: Path) -> str:
    return Path(os.path.relpath(path, REPO_ROOT)).as_posix()


def _family_root(config_path: Path) -> Path:
    return config_path.parent.parent


def _raw_output_root(config_path: Path) -> Path:
    return _family_root(config_path) / "0_raw_inputs" / "output1"


def _readme_path(config_path: Path) -> Path:
    return _family_root(config_path) / "README.md"


def _model_plot_relpath(readme_path: Path, plot_path: Path) -> str:
    return Path(os.path.relpath(plot_path, readme_path.parent)).as_posix()


def _family_plot_dir(config_path: Path) -> Path:
    return config_path.parent / "plot"


def _render_metric_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("Energy barrier abs err [eV]", metrics["energy"]["barrier_abs_err_eV"]),
        ("Delta E abs err [eV]", metrics["energy"]["deltaE_abs_err_eV"]),
        ("Energy profile RMSE [eV]", metrics["energy"]["profile_rmse_eV"]),
        ("Mean RMS displacement [A]", metrics["path"]["mean_rms_disp_A"]),
        ("Max RMS displacement [A]", metrics["path"]["max_rms_disp_A"]),
        ("AUC RMS displacement [A]", metrics["path"]["auc_rms_disp_A"]),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for label, value in rows:
        lines.append(f"| {label} | {float(value):.6f} |")
    return "\n".join(lines)


def _render_family_metric_table(metrics_by_model: dict[str, dict[str, Any]], model_order: list[str]) -> str:
    lines = [
        "| Model | Energy barrier abs err [eV] | Delta E abs err [eV] | Energy profile RMSE [eV] | Mean RMS displacement [A] | Max RMS displacement [A] | AUC RMS displacement [A] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in model_order:
        metrics = metrics_by_model[name]
        lines.append(
            "| {name} | {barrier:.6f} | {delta:.6f} | {profile:.6f} | {mean:.6f} | {maxv:.6f} | {auc:.6f} |".format(
                name=name,
                barrier=float(metrics["energy"]["barrier_abs_err_eV"]),
                delta=float(metrics["energy"]["deltaE_abs_err_eV"]),
                profile=float(metrics["energy"]["profile_rmse_eV"]),
                mean=float(metrics["path"]["mean_rms_disp_A"]),
                maxv=float(metrics["path"]["max_rms_disp_A"]),
                auc=float(metrics["path"]["auc_rms_disp_A"]),
            )
        )
    return "\n".join(lines)


def _render_model_report(report: BenchmarkModelReport, metrics: dict[str, Any]) -> str:
    rel_energy = report.energy_error_png.name
    rel_profiles = report.energy_profiles_png.name
    rel_path = report.path_fidelity_png.name
    return "\n".join(
        [
            f"# {report.name}",
            "",
            _render_metric_table(metrics),
            "",
            f"![Energy error]({rel_energy})",
            "",
            f"![Energy profiles]({rel_profiles})",
            "",
            f"![Path fidelity]({rel_path})",
            "",
        ]
    )


def _write_report_markdown(report: BenchmarkModelReport, metrics: dict[str, Any]) -> None:
    report.report_md.write_text(_render_model_report(report, metrics), encoding="utf-8")


def _render_family_section(
    *,
    config_path: Path,
    family_report: BenchmarkFamilyReport,
    metrics_by_model: dict[str, dict[str, Any]],
    model_order: list[str],
) -> str:
    command = f"mlip-neb --inputs {_relative_to_repo(config_path.parent)} --report-benchmark"
    readme_path = _readme_path(config_path)
    lines = [
        BENCHMARK_START,
        "",
        "The benchmark compares the baseline and fine-tuned models on the raw NEB input from `0_raw_inputs/output1`.",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "Compared models:",
        "",
    ]
    for idx, name in enumerate(model_order):
        role = "baseline" if idx == 0 else "fine-tuned"
        lines.append(f"- {role}: `{name}`")
    lines.extend(
        [
            "",
            "Metrics:",
            "",
            _render_family_metric_table(metrics_by_model, model_order),
            "",
            "Plots:",
            "",
            "### Energy",
            "",
            "![Energy profiles](" + _model_plot_relpath(readme_path, family_report.energy_profiles_png) + ")",
            "",
            "### Path fidelity",
            "",
            "![Path fidelity]("
            + _model_plot_relpath(readme_path, family_report.path_fidelity_png)
            + ")",
            "",
            f"[Report]({_model_plot_relpath(readme_path, family_report.report_md)})",
            "",
        ]
    )

    lines.append(BENCHMARK_END)
    return "\n".join(lines).rstrip() + "\n"


def _upsert_readme_section(readme_path: Path, section: str) -> None:
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README for benchmark report: {readme_path}")
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"\n?{re.escape(BENCHMARK_START)}.*?{re.escape(BENCHMARK_END)}\n?",
        re.DOTALL,
    )
    if pattern.search(text):
        updated = pattern.sub("\n" + section, text)
    else:
        updated = text.rstrip() + "\n\n" + section
    readme_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def generate_model_report(
    *,
    model_name: str,
    model_dir: Path,
    dft_root: Path,
    dft_neb_dat: Path,
) -> BenchmarkModelReport:
    plot_dir = model_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    npz_path = model_dir / "raw" / "neb_raw.npz"
    vasp_ci_dir = model_dir / "raw" / "vasp_ci"
    summary_path = model_dir / "raw" / "summary.txt"

    ref_s, ref_e = ces2.load_neb_dat(dft_neb_dat)
    model_s, model_e = ces2.load_mlip_npz(npz_path)
    energy_error_png = plot_dir / "energy_error.png"
    energy_profiles_png = plot_dir / "energy_profiles.png"
    energy_metrics = ces2.plot_error(
        ref_s=ref_s,
        ref_e=ref_e,
        models={model_name: (model_s, model_e)},
        out_png=energy_error_png,
        title=f"{model_name} NEB Energy Error",
    )
    ces2.plot_energy_profiles(
        ref_s=ref_s,
        ref_e=ref_e,
        models={model_name: (model_s, model_e)},
        out_png=energy_profiles_png,
        title=f"{model_name} NEB Energy Profiles",
    )

    ref_images = cps2.load_path_any(dft_root, prefer_contcar=False)
    model_images = cps2.load_path_any(vasp_ci_dir, expected_n_images=len(ref_images))
    path_fidelity_png = plot_dir / "path_fidelity.png"
    path_fidelity_json = plot_dir / "path_fidelity.json"
    cps2.plot(
        ref_images=ref_images,
        models={model_name: model_images},
        linear_images=[ref_images[0], ref_images[-1]],
        out_png=path_fidelity_png,
        out_json=path_fidelity_json,
        dft_neb_dat=dft_neb_dat if dft_neb_dat.exists() else None,
        barrier_sources={model_name: (summary_path if summary_path.exists() else None, npz_path)},
        title=f"{model_name} NEB Path Fidelity",
    )
    path_metrics = json.loads(path_fidelity_json.read_text(encoding="utf-8"))

    report_json = plot_dir / "report.json"
    report_md = plot_dir / "report.md"
    payload = {
        "model_name": model_name,
        "model_dir": str(model_dir.resolve()),
        "dft_root": str(dft_root.resolve()),
        "dft_neb_dat": str(dft_neb_dat.resolve()),
        "energy": energy_metrics[model_name],
        "path": path_metrics["models"][model_name],
        "supplementary_barriers_eV": path_metrics.get("supplementary_barriers_eV", {}),
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = BenchmarkModelReport(
        name=model_name,
        model_dir=model_dir,
        plot_dir=plot_dir,
        energy_error_png=energy_error_png,
        energy_profiles_png=energy_profiles_png,
        path_fidelity_png=path_fidelity_png,
        path_fidelity_json=path_fidelity_json,
        report_json=report_json,
        report_md=report_md,
    )
    _write_report_markdown(report, payload)
    return report


def generate_family_benchmark_report(config_path: Path, config: dict[str, Any]) -> list[BenchmarkModelReport]:
    names = model_names(config)
    if len(names) != 2:
        raise ValueError(
            f"Benchmark reports require exactly two models in defaults.model_name, got {len(names)}"
        )

    results_root = benchmark_root(config_path, config)
    dft_root = _raw_output_root(config_path)
    dft_neb_dat = dft_root / "neb.dat"
    if not dft_neb_dat.exists():
        raise FileNotFoundError(f"Missing DFT reference file for benchmark report: {dft_neb_dat}")

    reports: list[BenchmarkModelReport] = []
    for name in names:
        model_dir = results_root / name
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing benchmark results directory: {model_dir}")
        reports.append(
            generate_model_report(
                model_name=name,
                model_dir=model_dir,
                dft_root=dft_root,
                dft_neb_dat=dft_neb_dat,
            )
        )

    family_plot_dir = _family_plot_dir(config_path)
    family_plot_dir.mkdir(parents=True, exist_ok=True)

    ref_s, ref_e = ces2.load_neb_dat(dft_neb_dat)
    ref_images = cps2.load_path_any(dft_root, prefer_contcar=False)
    family_models_energy: dict[str, tuple[Any, Any]] = {}
    family_models_path: dict[str, list] = {}
    barrier_sources: dict[str, tuple[Path | None, Path | None]] = {}
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for report in reports:
        npz_path = report.model_dir / "raw" / "neb_raw.npz"
        vasp_ci_dir = report.model_dir / "raw" / "vasp_ci"
        summary_path = report.model_dir / "raw" / "summary.txt"
        family_models_energy[report.name] = ces2.load_mlip_npz(npz_path)
        family_models_path[report.name] = cps2.load_path_any(
            vasp_ci_dir,
            expected_n_images=len(ref_images),
        )
        barrier_sources[report.name] = (
            summary_path if summary_path.exists() else None,
            npz_path,
        )
        metrics_by_model[report.name] = json.loads(report.report_json.read_text(encoding="utf-8"))

    family_energy_error_png = family_plot_dir / "energy_error.png"
    family_energy_profiles_png = family_plot_dir / "energy_profiles.png"
    family_path_fidelity_png = family_plot_dir / "path_fidelity.png"
    family_path_fidelity_json = family_plot_dir / "path_fidelity.json"

    ces2.plot_error(
        ref_s=ref_s,
        ref_e=ref_e,
        models=family_models_energy,
        out_png=family_energy_error_png,
        title="NEB Energy Error: Baseline vs Fine-tuned",
    )
    ces2.plot_energy_profiles(
        ref_s=ref_s,
        ref_e=ref_e,
        models=family_models_energy,
        out_png=family_energy_profiles_png,
        title="NEB Energy Profiles: DFT, Baseline, and Fine-tuned",
    )
    cps2.plot(
        ref_images=ref_images,
        models=family_models_path,
        linear_images=[ref_images[0], ref_images[-1]],
        out_png=family_path_fidelity_png,
        out_json=family_path_fidelity_json,
        dft_neb_dat=dft_neb_dat if dft_neb_dat.exists() else None,
        barrier_sources=barrier_sources,
        title="NEB Path Fidelity: Baseline vs Fine-tuned",
    )

    family_report_json = family_plot_dir / "report.json"
    family_report_md = family_plot_dir / "report.md"
    family_payload = {
        "model_order": names,
        "dft_root": str(dft_root.resolve()),
        "dft_neb_dat": str(dft_neb_dat.resolve()),
        "models": metrics_by_model,
        "path_fidelity": json.loads(family_path_fidelity_json.read_text(encoding="utf-8")),
    }
    family_report_json.write_text(json.dumps(family_payload, indent=2), encoding="utf-8")
    family_report_md.write_text(
        "\n".join(
            [
                "# Benchmark Report",
                "",
                _render_family_metric_table(metrics_by_model, names),
                "",
                f"![Energy profiles]({_model_plot_relpath(_readme_path(config_path), family_energy_profiles_png)})",
                "",
                f"![Path fidelity]({_model_plot_relpath(_readme_path(config_path), family_path_fidelity_png)})",
                "",
            ]
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )

    family_report = BenchmarkFamilyReport(
        plot_dir=family_plot_dir,
        energy_error_png=family_energy_error_png,
        energy_profiles_png=family_energy_profiles_png,
        path_fidelity_png=family_path_fidelity_png,
        path_fidelity_json=family_path_fidelity_json,
        report_json=family_report_json,
        report_md=family_report_md,
    )
    readme_path = _readme_path(config_path)
    section = _render_family_section(
        config_path=config_path,
        family_report=family_report,
        metrics_by_model=metrics_by_model,
        model_order=names,
    )
    _upsert_readme_section(readme_path, section)
    return reports
