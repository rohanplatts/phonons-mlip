from __future__ import annotations

from pathlib import Path
from typing import Any

from .cluster import available_models
from .io import analysis_dir, manifest_path, read_csv_rows


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
        if abs(number) >= 100:
            return f"{number:.3f}"
        return f"{number:.5f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 50) -> str:
    if not rows:
        return "_No rows available._"
    visible = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |" for row in visible]
    extra = [f"\n_Only showing first {max_rows} of {len(rows)} rows._"] if len(rows) > max_rows else []
    return "\n".join([header, sep, *body, *extra])


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def _read(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def write_report(*, results_root: str | Path, case_name: str) -> Path:
    out_dir = analysis_dir(results_root, case_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _read(manifest_path(results_root, case_name))
    relaxations = _read(out_dir / "relaxation_results.csv")
    selected = _read(out_dir / "selected_for_dft_manifest.csv")
    dft_manifest = _read(out_dir / "dft_validation_manifest.csv")
    dft_status = _read(out_dir / "dft_status.csv")
    summary = _read(out_dir / "mlip_vs_dft.csv")
    geometry = _read(out_dir / "geometry_comparisons.csv")

    try:
        models = available_models(results_root, case_name)
    except Exception:
        models = sorted({row.get("model_name", "") for row in relaxations if row.get("model_name")})

    lines = [
        f"# MLIP-SnB Report: {case_name}",
        "",
        _section(
            "Method",
            "\n".join(
                [
                    f"- Imported/generated {len(candidates)} ShakeNBreak candidate POSCARs.",
                    f"- Relaxed candidates with MLIP model(s): {', '.join(models) if models else 'not run yet'}.",
                    "- Clustered MLIP-relaxed structures with pymatgen StructureMatcher using ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=False, attempt_supercell=False unless overridden.",
                    "- Ranked one representative per MLIP cluster by MLIP energy.",
                    "- Exported selected representatives as VASP-ready DFT validation folders when requested.",
                    "- When DFT results are present, compared MLIP and DFT cluster ranking, relative energies, StructureMatcher fits, and species-resolved Hungarian MIC geometry errors.",
                ]
            ),
        ),
    ]

    if candidates:
        lines.append(
            _section(
                "Exact Input Structures",
                markdown_table(candidates, ["candidate_id", "source_poscar", "staged_poscar"], max_rows=200),
            )
        )

    if relaxations:
        lines.append(
            _section(
                "MLIP Relaxation Summary",
                markdown_table(
                    relaxations,
                    ["model_name", "candidate_id", "energy_eV", "dE_mlip_eV", "max_force_eVA", "converged", "relaxed_contcar"],
                    max_rows=200,
                ),
            )
        )

    if selected:
        lines.append(
            _section(
                "Selected For DFT",
                markdown_table(
                    selected,
                    ["selection_id", "model_name", "cluster_id", "best_candidate_id", "best_mlip_dE_eV", "selected_poscar"],
                    max_rows=200,
                ),
            )
        )

    if dft_manifest:
        lines.append(
            _section(
                "DFT Folders",
                markdown_table(dft_manifest, ["selection_id", "model_name", "cluster_id", "dft_dir"], max_rows=200),
            )
        )

    if dft_status:
        lines.append(
            _section(
                "DFT Status",
                markdown_table(
                    dft_status,
                    ["selection_id", "model_name", "final_energy_eV", "max_final_force_eVA", "converged", "restart_needed", "restart_reason"],
                    max_rows=200,
                ),
            )
        )

    if summary:
        lines.append(
            _section(
                "Ground Cluster Results",
                markdown_table(
                    summary,
                    [
                        "model_name",
                        "top1_correct",
                        "top3_contains_dft_ground",
                        "dft_energy_penalty_eV",
                        "relative_energy_mae_eV",
                        "ground_geometry_rms_A",
                        "ground_geometry_max_A",
                    ],
                ),
            )
        )

    if geometry:
        lines.append(
            _section(
                "Geometry Comparisons",
                markdown_table(
                    geometry,
                    [
                        "selection_id",
                        "model_name",
                        "best_mlip_dE_eV",
                        "dft_dE_eV",
                        "dft_cluster_id",
                        "structure_matcher_fit",
                        "mlip_to_dft_rms_A",
                        "mlip_to_dft_max_A",
                    ],
                    max_rows=200,
                ),
            )
        )

    supervisor_lines = []
    if summary:
        n_top1 = sum(str(row.get("top1_correct", "")).lower() in {"true", "1"} for row in summary)
        supervisor_lines.append(
            f"Across {len(summary)} model comparison(s), {n_top1} had the MLIP-selected ground cluster relax to the same DFT cluster as the DFT ground cluster among the validated representatives."
        )
        penalties = [float(row["dft_energy_penalty_eV"]) for row in summary if row.get("dft_energy_penalty_eV") not in ("", None)]
        if penalties:
            supervisor_lines.append(f"The largest DFT penalty for using the MLIP-selected ground representative was {max(penalties):.5f} eV.")
    elif selected and not dft_status:
        supervisor_lines.append("MLIP-selected representatives are prepared for DFT validation, but completed DFT outputs have not been parsed yet.")
    else:
        supervisor_lines.append("Run the remaining stages before drawing a ground-cluster conclusion.")

    lines.append(_section("Supervisor Summary", "\n".join(f"- {line}" for line in supervisor_lines)))

    out = out_dir / "report.md"
    out.write_text("\n".join(lines).rstrip() + "\n")
    return out
