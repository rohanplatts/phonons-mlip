from pathlib import Path
import math
import re
import shutil
import stat
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")
ANALYSIS_DIR = ROOT / "runs" / "analysis"
RESTART_CSV = ANALYSIS_DIR / "dft_jobs_to_restart.csv"

HUMAN_ESTIMATE_TXT = ANALYSIS_DIR / "human_read_dft_restart_estimates.txt"


ESTIMATE_CSV = ANALYSIS_DIR / "dft_restart_estimates.csv"

LOCAL_DFT_VALIDATE_DIR = ROOT / "runs" / "dft_validate"

BUNYA_REMOTE = "s4802880@bunya.rcc.uq.edu.au"
BUNYA_DFT_VALIDATE_DIR = (
    "/scratch/user/s4802880/mlip_phonons/defect_landscape/dft_validate/dft_validate"
)

SUBMIT_HELPER = ANALYSIS_DIR / "restart_sbatch_commands.sh"

OUTPUT_FILES_TO_BACKUP = [
    "OUTCAR",
    "OSZICAR",
    "vasprun.xml",
    "vasp.out",
    "slurm.out",
    "slurm.err",
    "REPORT",
    "XDATCAR",
    "DOSCAR",
    "EIGENVAL",
    "IBZKPT",
    "PCDAT",
    "vaspout.h5",
]



def strip_comment(line: str) -> str:
    return re.split(r"[#!]", line, maxsplit=1)[0].strip()

def usable_file(path: Path, min_bytes: int = 1024):
    return path.exists() and path.stat().st_size >= min_bytes

def read_incar_value(incar: Path, key: str):
    if not incar.exists():
        return None

    key = key.upper()

    for line in incar.read_text(errors="ignore").splitlines():
        clean = strip_comment(line)
        m = re.match(r"^\s*([A-Za-z_]+)\s*=\s*(.+?)\s*$", clean)
        if not m:
            continue
        if m.group(1).upper() == key:
            return m.group(2).strip()

    return None


def parse_float(x):
    try:
        return float(str(x).replace("D", "E"))
    except Exception:
        return np.nan


def parse_int(x):
    try:
        return int(float(str(x)))
    except Exception:
        return np.nan


def parse_oszicar(oszicar: Path):
    out = {
        "steps_done": 0,
        "last_F_eV": np.nan,
        "last_E0_eV": np.nan,
        "last_dE_eV": np.nan,
    }

    if not oszicar.exists():
        return out

    ionic_lines = []

    for line in oszicar.read_text(errors="ignore").splitlines():
        m = re.search(
            r"^\s*(\d+)\s+F=\s*([-+0-9.EeDd]+)\s+E0=\s*([-+0-9.EeDd]+)\s+d E\s*=\s*([-+0-9.EeDd]+)",
            line,
        )
        if m:
            ionic_lines.append(m)

    if not ionic_lines:
        return out

    last = ionic_lines[-1]
    out["steps_done"] = int(last.group(1))
    out["last_F_eV"] = parse_float(last.group(2))
    out["last_E0_eV"] = parse_float(last.group(3))
    out["last_dE_eV"] = parse_float(last.group(4))

    return out


def parse_force_blocks(outcar: Path):
    if not outcar.exists():
        return []

    lines = outcar.read_text(errors="ignore").splitlines()
    starts = [i for i, line in enumerate(lines) if "TOTAL-FORCE (eV/Angst)" in line]

    max_forces = []

    for idx in starts:
        block_forces = []

        for line in lines[idx + 2:]:
            parts = line.split()

            if len(parts) < 6:
                if block_forces:
                    break
                continue

            try:
                fx, fy, fz = map(float, parts[3:6])
                block_forces.append(math.sqrt(fx * fx + fy * fy + fz * fz))
            except Exception:
                if block_forces:
                    break

        if block_forces:
            max_forces.append(max(block_forces))

    return max_forces


def parse_slurm_time_to_seconds(s: str):
    """
    Accepts common SLURM forms:
      HH:MM:SS
      MM:SS
      D-HH:MM:SS
      minutes
    """
    if s is None:
        return np.nan

    s = str(s).strip()

    if not s:
        return np.nan

    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)

    parts = s.split(":")

    try:
        if len(parts) == 3:
            h, m, sec = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, sec = map(int, parts)
        elif len(parts) == 1:
            h = 0
            m = int(parts[0])
            sec = 0
        else:
            return np.nan
    except Exception:
        return np.nan

    return days * 86400 + h * 3600 + m * 60 + sec


def parse_submit_walltime(submit: Path):
    if not submit.exists():
        return None, np.nan

    text = submit.read_text(errors="ignore")

    patterns = [
        r"#SBATCH\s+--time=([^\s]+)",
        r"#SBATCH\s+--time\s+([^\s]+)",
        r"#SBATCH\s+-t\s+([^\s]+)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            t = m.group(1).strip()
            return t, parse_slurm_time_to_seconds(t)

    return None, np.nan


def format_seconds(seconds):
    if seconds is None or not np.isfinite(seconds):
        return ""

    seconds = int(math.ceil(seconds / 60.0) * 60)

    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)

    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:00"

    return f"{hours:02d}:{minutes:02d}:00"


def estimate_remaining_steps(force_series, force_target):
    if not force_series or not np.isfinite(force_target) or force_target <= 0:
        return np.nan, np.nan, "no_force_estimate"

    current = force_series[-1]

    if not np.isfinite(current) or current <= 0:
        return np.nan, np.nan, "no_force_estimate"

    ratio_to_target = current / force_target

    if ratio_to_target <= 1.0:
        return 1, 3, "force_already_near_target_but_vasp_did_not_stop"

    recent = force_series[-6:]
    ratios = []

    for a, b in zip(recent[:-1], recent[1:]):
        if a > 0 and b > 0:
            r = b / a
            if 0.2 < r < 0.98:
                ratios.append(r)

    if ratios:
        r = float(np.median(ratios))
        est = math.ceil(math.log(force_target / current) / math.log(r))
        est = max(1, min(est, 80))
        return max(1, math.floor(0.7 * est)), max(3, math.ceil(1.8 * est) + 2), "log_force_decay"

    if ratio_to_target <= 1.25:
        return 2, 5, "heuristic_force_ratio"
    if ratio_to_target <= 1.5:
        return 4, 8, "heuristic_force_ratio"
    if ratio_to_target <= 2.0:
        return 6, 12, "heuristic_force_ratio"
    if ratio_to_target <= 3.0:
        return 8, 18, "heuristic_force_ratio"

    return 12, 30, "heuristic_force_ratio"

def patch_incar_restart_only(incar: Path, use_wavecar: bool, use_chgcar: bool):
    lines = incar.read_text(errors="ignore").splitlines()

    remove_keys = {"ISTART", "ICHARG"}
    kept = []

    for line in lines:
        m = re.match(r"^\s*([A-Za-z_]+)\s*=", line)
        if m and m.group(1).upper() in remove_keys:
            continue
        kept.append(line)

    istart = 1 if use_wavecar else 0
    icharg = 1 if use_chgcar else 2

    kept += [
        "",
        "# Restart after walltime kill",
        f"ISTART = {istart}",
        f"ICHARG = {icharg}",
    ]

    incar.write_text("\n".join(kept).rstrip() + "\n")

def prepare_restart_folder(dft_dir: Path):
    contcar = dft_dir / "CONTCAR"
    poscar = dft_dir / "POSCAR"
    incar = dft_dir / "INCAR"
    wavecar = dft_dir / "WAVECAR"
    chgcar = dft_dir / "CHGCAR"
    submit = dft_dir / "submit.sh"

    # These are genuinely required.
    for f in [contcar, incar, submit]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required restart file: {f}")
        if f.stat().st_size == 0:
            raise RuntimeError(f"Required restart file is empty: {f}")

    use_wavecar = usable_file(wavecar)
    use_chgcar = usable_file(chgcar)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = dft_dir / f"pre_restart_{stamp}"
    backup.mkdir(exist_ok=False)

    if poscar.exists():
        shutil.copy2(poscar, backup / "POSCAR.before_restart")
    shutil.copy2(incar, backup / "INCAR.before_restart")
    shutil.copy2(submit, backup / "submit.before_restart.sh")

    for name in OUTPUT_FILES_TO_BACKUP:
        src = dft_dir / name
        if src.exists():
            shutil.move(str(src), str(backup / name))

    # Move empty restart files out of the way so they do not confuse us later.
    if wavecar.exists() and wavecar.stat().st_size == 0:
        shutil.move(str(wavecar), str(backup / "WAVECAR.empty"))

    if chgcar.exists() and chgcar.stat().st_size == 0:
        shutil.move(str(chgcar), str(backup / "CHGCAR.empty"))

    # Geometry continuation.
    shutil.copy2(contcar, poscar)

    # Electronic continuation if possible, otherwise fresh electronic start.
    patch_incar_restart_only(
        incar=incar,
        use_wavecar=use_wavecar,
        use_chgcar=use_chgcar,
    )

    return backup, use_wavecar, use_chgcar

def analyse_restart_job(row):
    dft_dir = Path(row["dft_dir"])
    incar = dft_dir / "INCAR"
    oszicar = dft_dir / "OSZICAR"
    outcar = dft_dir / "OUTCAR"
    submit = dft_dir / "submit.sh"

    osz = parse_oszicar(oszicar)
    forces = parse_force_blocks(outcar)

    nsw = parse_int(read_incar_value(incar, "NSW"))
    ediffg = parse_float(read_incar_value(incar, "EDIFFG"))

    force_target = abs(ediffg) if np.isfinite(ediffg) and ediffg < 0 else np.nan
    max_force = forces[-1] if forces else np.nan

    low_steps, high_steps, estimate_method = estimate_remaining_steps(forces, force_target)

    walltime_text, walltime_sec = parse_submit_walltime(submit)
    steps_done = osz["steps_done"]

    sec_per_step = np.nan
    if np.isfinite(walltime_sec) and steps_done > 0:
        sec_per_step = walltime_sec / steps_done

    est_low_sec = np.nan
    est_high_sec = np.nan
    suggested_sec = np.nan

    if np.isfinite(sec_per_step) and np.isfinite(low_steps) and np.isfinite(high_steps):
        est_low_sec = low_steps * sec_per_step
        est_high_sec = high_steps * sec_per_step
        suggested_sec = est_high_sec * 1.5

    force_ratio = np.nan
    if np.isfinite(max_force) and np.isfinite(force_target) and force_target > 0:
        force_ratio = max_force / force_target

    return {
        "model_label": row["model_label"],
        "cluster_id": int(row["cluster_id"]),
        "folder": dft_dir.name,
        "steps_done": steps_done,
        "NSW": nsw,
        "EDIFFG": ediffg,
        "force_target_eVA": force_target,
        "last_max_force_eVA": max_force,
        "force_over_target": force_ratio,
        "last_E0_eV": osz["last_E0_eV"],
        "last_dE_eV": osz["last_dE_eV"],
        "old_walltime": walltime_text or "",
        "sec_per_ionic_step_est": sec_per_step,
        "remaining_steps_low": low_steps,
        "remaining_steps_high": high_steps,
        "estimate_method": estimate_method,
        "est_remaining_walltime_low": format_seconds(est_low_sec),
        "est_remaining_walltime_high": format_seconds(est_high_sec),
        "suggested_walltime": format_seconds(suggested_sec),
        "dft_dir": str(dft_dir),
    }


def main():
    if not RESTART_CSV.exists():
        raise FileNotFoundError(f"Missing {RESTART_CSV}. Run 05_check_dft_jobs.py first.")

    df = pd.read_csv(RESTART_CSV)

    if df.empty:
        print("No jobs need restart.")
        return

    estimates = pd.DataFrame([analyse_restart_job(row) for _, row in df.iterrows()])
    estimates.to_csv(ESTIMATE_CSV, index=False, float_format="%.10f")

    show_cols = [
        "folder",
        "steps_done",
        "NSW",
        "EDIFFG",
        "last_max_force_eVA",
        "force_over_target",
        "last_dE_eV",
        "old_walltime",
        "remaining_steps_low",
        "remaining_steps_high",
        "suggested_walltime",
        "estimate_method",
    ]

    human_table = estimates[show_cols].to_string(index=False, max_colwidth=80)

    print("\nRestart estimates:")
    print(human_table)

    HUMAN_ESTIMATE_TXT.write_text(
        "DFT restart estimates\n"
        "=====================\n\n"
        + human_table
        + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote estimate tables:\n  {ESTIMATE_CSV}\n  {HUMAN_ESTIMATE_TXT}")

    commands = []

    for _, row in estimates.iterrows():
        dft_dir = Path(row["dft_dir"])

        print(f"\nPreparing restart: {dft_dir.name}")
        backup, use_wavecar, use_chgcar = prepare_restart_folder(dft_dir)

        print(f"  backed up old outputs to: {backup.name}")
        print("  copied CONTCAR -> POSCAR")
        print(f"  patched INCAR with ISTART={1 if use_wavecar else 0}")
        print(f"  patched INCAR with ICHARG={1 if use_chgcar else 2}")
        print("  left NSW unchanged")
        print("  left submit.sh unchanged")


    folders = list(estimates["folder"])

    lines = []

    lines += [
        "# ============================================================",
        "# 1) RUN THIS LOCALLY: copy restart-prepared folders to Bunya",
        "# ============================================================",
        "",
        f"cd {LOCAL_DFT_VALIDATE_DIR}",
        "",
        f'REMOTE="{BUNYA_REMOTE}"',
        f'REMOTE_DIR="{BUNYA_DFT_VALIDATE_DIR}"',
        "",
        'ssh "$REMOTE" "mkdir -p $REMOTE_DIR"',
        "",
        "rsync -avh --delete \\",
    ]

    for folder in folders:
        lines.append(f"  {folder} \\")

    lines.append('  "$REMOTE:$REMOTE_DIR/"')

    lines += [
        "",
        "",
        "# ============================================================",
        "# 2) RUN THIS ON BUNYA: submit restarted jobs",
        "# ============================================================",
        "",
        f"cd {BUNYA_DFT_VALIDATE_DIR}",
        "",
    ]

    for folder in folders:
        lines += [
            f"cd {BUNYA_DFT_VALIDATE_DIR}/{folder}",
            "sbatch submit.sh",
            "",
        ]

    SUBMIT_HELPER.write_text("\n".join(lines).rstrip() + "\n")

    print(f"\nWrote restart command list:\n  {SUBMIT_HELPER}")
    print("\nOpen it, run the rsync block locally, then run the sbatch block on Bunya.")




if __name__ == "__main__":
    main()