from pathlib import Path
import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")

ANALYSIS_DIR = ROOT / "runs" / "analysis"
DFT_VALIDATE_DIR = ROOT / "runs" / "dft_validate"
MLIP_RELAXED_DIR = ROOT / "runs" / "mlip_relaxed"

DFT_MANIFEST = ANALYSIS_DIR / "dft_validation_manifest.csv"
OUT_TXT = ANALYSIS_DIR / "human_read_performance_metrics.txt"


# -------------------------
# Formatting helpers
# -------------------------

def seconds_to_hms(seconds):
    if seconds is None or not np.isfinite(seconds):
        return ""

    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_float(x):
    try:
        return float(str(x).replace("D", "E"))
    except Exception:
        return np.nan


def parse_mem_to_gb(s):
    """
    Converts SLURM memory strings like 700M, 1G, 24G, 32000M to GB.
    Uses decimal-ish convention because this is just reporting.
    """
    if s is None:
        return np.nan

    s = str(s).strip().upper()
    m = re.match(r"([0-9.]+)\s*([KMGTP]?)", s)

    if not m:
        return np.nan

    val = float(m.group(1))
    unit = m.group(2)

    factors = {
        "": 1 / 1000,   # SLURM bare number is usually MB
        "K": 1 / 1e6,
        "M": 1 / 1000,
        "G": 1,
        "T": 1000,
        "P": 1e6,
    }

    return val * factors.get(unit, np.nan)


# -------------------------
# VASP parsing
# -------------------------

def parse_outcar_timing(outcar):
    outcar = Path(outcar)

    out = {
        "elapsed_sec": np.nan,
        "cpu_sec": np.nan,
        "converged_ionic": False,
        "final_energy_eV": np.nan,
        "final_max_force_eVA": np.nan,
    }

    if not outcar.exists():
        return out

    text = outcar.read_text(errors="ignore")

    elapsed = re.findall(r"Elapsed time \(sec\):\s*([-+0-9.EeDd]+)", text)
    cpu = re.findall(r"Total CPU time used \(sec\):\s*([-+0-9.EeDd]+)", text)
    energies = re.findall(r"free\s+energy\s+TOTEN\s+=\s*([-+0-9.EeDd]+)", text)

    if elapsed:
        out["elapsed_sec"] = parse_float(elapsed[-1])
    if cpu:
        out["cpu_sec"] = parse_float(cpu[-1])
    if energies:
        out["final_energy_eV"] = parse_float(energies[-1])

    out["converged_ionic"] = (
        "reached required accuracy - stopping structural energy minimisation"
        in text
    )

    out["final_max_force_eVA"] = parse_final_max_force(outcar)

    return out


def parse_final_max_force(outcar):
    outcar = Path(outcar)

    if not outcar.exists():
        return np.nan

    lines = outcar.read_text(errors="ignore").splitlines()
    starts = [i for i, line in enumerate(lines) if "TOTAL-FORCE (eV/Angst)" in line]

    if not starts:
        return np.nan

    idx = starts[-1]
    forces = []

    for line in lines[idx + 2:]:
        parts = line.split()

        if len(parts) < 6:
            if forces:
                break
            continue

        try:
            fx, fy, fz = map(float, parts[3:6])
            forces.append(math.sqrt(fx * fx + fy * fy + fz * fz))
        except Exception:
            if forces:
                break

    if not forces:
        return np.nan

    return max(forces)


def count_ionic_steps(oszicar):
    oszicar = Path(oszicar)

    if not oszicar.exists():
        return 0

    n = 0
    for line in oszicar.read_text(errors="ignore").splitlines():
        if re.match(r"^\s*\d+\s+F=", line):
            n += 1

    return n


# -------------------------
# SLURM submit parsing
# -------------------------

def read_submit_settings(submit):
    submit = Path(submit)

    out = {
        "ntasks": np.nan,
        "nodes": np.nan,
        "mem_gb_total": np.nan,
        "time_limit": "",
    }

    if not submit.exists():
        return out

    text = submit.read_text(errors="ignore")

    def grab(pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else None

    ntasks = grab(r"#SBATCH\s+--ntasks[=\s]+([^\s]+)")
    nodes = grab(r"#SBATCH\s+--nodes[=\s]+([^\s]+)")
    mem = grab(r"#SBATCH\s+--mem[=\s]+([^\s]+)")
    mem_per_cpu = grab(r"#SBATCH\s+--mem-per-cpu[=\s]+([^\s]+)")
    time_limit = grab(r"#SBATCH\s+--time[=\s]+([^\s]+)")

    if ntasks is not None:
        out["ntasks"] = int(float(ntasks))

    if nodes is not None:
        out["nodes"] = int(float(nodes))

    if mem is not None:
        out["mem_gb_total"] = parse_mem_to_gb(mem)

    elif mem_per_cpu is not None and np.isfinite(out["ntasks"]):
        out["mem_gb_total"] = parse_mem_to_gb(mem_per_cpu) * out["ntasks"]

    if time_limit is not None:
        out["time_limit"] = time_limit

    return out


# -------------------------
# Segment discovery
# -------------------------

def get_vasp_attempts(dft_dir):
    """
    One final DFT relaxation may have multiple VASP attempts:
      pre_restart_*/OUTCAR
      current OUTCAR

    We count all attempts with OUTCARs because each consumed compute.
    """
    dft_dir = Path(dft_dir)

    attempts = []

    for pre in sorted(dft_dir.glob("pre_restart_*")):
        if (pre / "OUTCAR").exists():
            attempts.append(
                {
                    "label": pre.name,
                    "dir": pre,
                    "outcar": pre / "OUTCAR",
                    "oszicar": pre / "OSZICAR",
                    "submit": pre / "submit.before_restart.sh",
                    "is_final": False,
                }
            )

    if (dft_dir / "OUTCAR").exists():
        attempts.append(
            {
                "label": "final",
                "dir": dft_dir,
                "outcar": dft_dir / "OUTCAR",
                "oszicar": dft_dir / "OSZICAR",
                "submit": dft_dir / "submit.sh",
                "is_final": True,
            }
        )

    return attempts


def summarize_dft_job(row):
    dft_dir = Path(row["dft_dir"])
    attempts = get_vasp_attempts(dft_dir)

    wall_sec_total = 0.0
    cpu_sec_total = 0.0
    nominal_core_hours = 0.0
    ionic_steps_total = 0

    ntasks_seen = []
    mem_seen = []
    time_limits_seen = []

    final_energy = np.nan
    final_force = np.nan
    final_converged = False

    for attempt in attempts:
        timing = parse_outcar_timing(attempt["outcar"])
        settings = read_submit_settings(attempt["submit"])
        ionic_steps = count_ionic_steps(attempt["oszicar"])

        elapsed = timing["elapsed_sec"]
        cpu_sec = timing["cpu_sec"]

        if np.isfinite(elapsed):
            wall_sec_total += elapsed

        if np.isfinite(cpu_sec):
            cpu_sec_total += cpu_sec

        if np.isfinite(elapsed) and np.isfinite(settings["ntasks"]):
            nominal_core_hours += settings["ntasks"] * elapsed / 3600.0

        ionic_steps_total += ionic_steps

        if np.isfinite(settings["ntasks"]):
            ntasks_seen.append(int(settings["ntasks"]))

        if np.isfinite(settings["mem_gb_total"]):
            mem_seen.append(float(settings["mem_gb_total"]))

        if settings["time_limit"]:
            time_limits_seen.append(settings["time_limit"])

        if attempt["is_final"]:
            final_energy = timing["final_energy_eV"]
            final_force = timing["final_max_force_eVA"]
            final_converged = timing["converged_ionic"]

    cpu_hours_total = cpu_sec_total / 3600.0 if cpu_sec_total > 0 else np.nan

    return {
        "model": row["model_label"],
        "cluster": int(row["cluster_id"]),
        "folder": dft_dir.name,
        "attempts": len(attempts),
        "ionic_steps_total": ionic_steps_total,
        "req_cpus_seen": ",".join(map(str, sorted(set(ntasks_seen)))) if ntasks_seen else "",
        "req_mem_GB_seen": ",".join(f"{x:.1f}" for x in sorted(set(mem_seen))) if mem_seen else "",
        "time_limits_seen": ",".join(sorted(set(time_limits_seen))) if time_limits_seen else "",
        "vasp_wall_time": seconds_to_hms(wall_sec_total),
        "vasp_wall_hours": wall_sec_total / 3600.0 if wall_sec_total > 0 else np.nan,
        "vasp_cpu_hours": cpu_hours_total,
        "nominal_core_hours": nominal_core_hours if nominal_core_hours > 0 else np.nan,
        "final_energy_eV": final_energy,
        "final_max_force_eVA": final_force,
        "converged": bool(final_converged),
    }


# -------------------------
# Optional MLIP timing/counts
# -------------------------

def find_mlip_runtime_seconds(result):
    """
    Tries common names. If your result.json does not store timing,
    this returns NaN.
    """
    keys = [
        "runtime_s",
        "runtime_sec",
        "elapsed_s",
        "elapsed_sec",
        "walltime_s",
        "walltime_sec",
        "relax_time_s",
        "relax_time_sec",
        "time_s",
        "time_sec",
        "seconds",
    ]

    for key in keys:
        if key in result:
            val = parse_float(result[key])
            if np.isfinite(val):
                return val

    return np.nan


def summarize_mlip_relaxations():
    rows = []

    if not MLIP_RELAXED_DIR.exists():
        return pd.DataFrame(rows)

    for model_dir in sorted(MLIP_RELAXED_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        runtimes = []
        n_results = 0

        for f in model_dir.rglob("result.json"):
            n_results += 1
            try:
                data = json.loads(f.read_text())
                runtimes.append(find_mlip_runtime_seconds(data))
            except Exception:
                runtimes.append(np.nan)

        finite = [x for x in runtimes if np.isfinite(x)]

        rows.append(
            {
                "model": model_dir.name,
                "n_mlip_relaxations": n_results,
                "mlip_wall_time": seconds_to_hms(sum(finite)) if finite else "not recorded",
                "mlip_wall_hours": sum(finite) / 3600.0 if finite else np.nan,
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# Main
# -------------------------

def main():
    if not DFT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing DFT manifest: {DFT_MANIFEST}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(DFT_MANIFEST)
    dft = pd.DataFrame([summarize_dft_job(row) for _, row in manifest.iterrows()])

    dft = dft.sort_values(["model", "cluster"]).reset_index(drop=True)

    show_cols = [
        "model",
        "cluster",
        "attempts",
        "ionic_steps_total",
        "req_cpus_seen",
        "req_mem_GB_seen",
        "vasp_wall_time",
        "vasp_cpu_hours",
        "nominal_core_hours",
        "final_max_force_eVA",
        "converged",
    ]

    dft_show = dft[show_cols].copy()

    for col in ["vasp_cpu_hours", "nominal_core_hours", "final_max_force_eVA"]:
        dft_show[col] = dft_show[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.3f}")

    dft_table = dft_show.to_string(index=False, max_colwidth=40)

    totals = {
        "n_dft_results": len(dft),
        "n_vasp_attempts": int(dft["attempts"].sum()),
        "total_ionic_steps": int(dft["ionic_steps_total"].sum()),
        "total_vasp_wall_time": seconds_to_hms(float(dft["vasp_wall_hours"].sum()) * 3600.0),
        "total_vasp_wall_hours": float(dft["vasp_wall_hours"].sum()),
        "total_vasp_cpu_hours": float(dft["vasp_cpu_hours"].dropna().sum()),
        "total_nominal_core_hours": float(dft["nominal_core_hours"].dropna().sum()),
    }

    mlip = summarize_mlip_relaxations()

    if not mlip.empty:
        mlip_show = mlip.copy()
        mlip_show["mlip_wall_hours"] = mlip_show["mlip_wall_hours"].map(
            lambda x: "" if not np.isfinite(x) else f"{x:.3f}"
        )
        mlip_table = mlip_show.to_string(index=False, max_colwidth=40)
    else:
        mlip_table = "No MLIP result.json files found."

    text = []
    text.append("ShakeNBreak / MLIP / DFT performance metrics")
    text.append("=" * 55)
    text.append("")
    text.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    text.append("")
    text.append("Definitions")
    text.append("-" * 55)
    text.append("attempts              = number of VASP OUTCARs counted, including pre_restart_* folders")
    text.append("ionic_steps_total     = sum of ionic steps across all counted VASP attempts")
    text.append("req_cpus_seen         = unique #SBATCH --ntasks values found in submit scripts")
    text.append("req_mem_GB_seen       = unique total requested memory values inferred from submit scripts")
    text.append("vasp_wall_time        = sum of OUTCAR 'Elapsed time (sec)' across attempts")
    text.append("vasp_cpu_hours        = sum of OUTCAR 'Total CPU time used (sec)' / 3600")
    text.append("nominal_core_hours    = sum of requested ntasks × VASP walltime")
    text.append("final_max_force_eVA   = max force from final OUTCAR force block")
    text.append("")
    text.append("Per DFT-validated structure")
    text.append("-" * 55)
    text.append(dft_table)
    text.append("")
    text.append("DFT totals")
    text.append("-" * 55)
    for k, v in totals.items():
        if isinstance(v, float):
            text.append(f"{k:28s}: {v:.3f}")
        else:
            text.append(f"{k:28s}: {v}")
    text.append("")
    text.append("MLIP pre-screening")
    text.append("-" * 55)
    text.append(mlip_table)
    text.append("")
    text.append("Notes")
    text.append("-" * 55)
    text.append("This report is based on files present locally under runs/dft_validate and runs/mlip_relaxed.")
    text.append("If SLURM allocated more cores/nodes than requested, nominal_core_hours may undercount true allocation.")
    text.append("For exact allocation efficiency, use seff/sacct on Bunya job IDs.")

    OUT_TXT.write_text("\n".join(text) + "\n", encoding="utf-8")

    print(f"Wrote:\n  {OUT_TXT}")
    print("")
    print(dft_table)


if __name__ == "__main__":
    main()