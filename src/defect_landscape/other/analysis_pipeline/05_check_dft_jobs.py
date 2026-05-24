from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.io.vasp.outputs import Vasprun

"""
run: 

rsync -avz --progress \
  s4802880@bunya.rcc.uq.edu.au:/scratch/user/s4802880/mlip_phonons/defect_landscape/dft_validate/dft_validate/ \
  /home/rnpla/projects/mlip_phonons/src/defect_landscape/runs/dft_validate/





"""
ROOT = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape")
ANALYSIS_DIR = ROOT / "runs" / "analysis"
DFT_MANIFEST = ANALYSIS_DIR / "dft_validation_manifest.csv"

STATUS_CSV = ANALYSIS_DIR / "dft_job_status.csv"
RESTART_CSV = ANALYSIS_DIR / "dft_jobs_to_restart.csv"


def read_vasprun(dft_dir: Path):
    vasprun = dft_dir / "vasprun.xml"

    out = {
        "vasprun_ok": False,
        "converged_electronic": False,
        "converged_ionic": False,
        "final_energy_eV": np.nan,
        "max_final_force_eVA": np.nan,
        "n_ionic_steps": np.nan,
    }

    try:
        vr = Vasprun(
            vasprun,
            parse_dos=False,
            parse_eigen=False,
            parse_potcar_file=False,
        )

        out["vasprun_ok"] = True
        out["converged_electronic"] = bool(vr.converged_electronic)
        out["converged_ionic"] = bool(vr.converged_ionic)
        out["final_energy_eV"] = float(vr.final_energy)
        out["n_ionic_steps"] = len(vr.ionic_steps)

        forces = np.array(vr.ionic_steps[-1]["forces"], dtype=float)
        out["max_final_force_eVA"] = float(np.linalg.norm(forces, axis=1).max())

    except Exception:
        pass

    return out


def check_outcar(dft_dir: Path):
    outcar = dft_dir / "OUTCAR"

    if not outcar.exists():
        return False, "missing_OUTCAR"

    text = outcar.read_text(errors="ignore")

    reached_accuracy = (
        "reached required accuracy - stopping structural energy minimisation"
        in text
    )

    bad_terms = [
        "ZBRENT: fatal error",
        "BRMIX: very serious problems",
        "EDDDAV",
        "Sub-Space-Matrix is not hermitian",
        "NaN",
        "nan",
    ]

    found_bad = [term for term in bad_terms if term in text]

    reasons = []
    if not reached_accuracy:
        reasons.append("no_required_accuracy_message")
    if found_bad:
        reasons.append("OUTCAR_bad_terms=" + "|".join(found_bad))

    return reached_accuracy and not found_bad, "; ".join(reasons)


def main():
    df = pd.read_csv(DFT_MANIFEST)
    rows = []

    for _, row in df.iterrows():
        dft_dir = Path(row["dft_dir"])
        folder = dft_dir.name

        v = read_vasprun(dft_dir)
        outcar_clean, outcar_reason = check_outcar(dft_dir)

        reasons = []

        if not v["vasprun_ok"]:
            reasons.append("vasprun_unreadable_or_truncated")
        if not v["converged_electronic"]:
            reasons.append("not_electronically_converged")
        if not v["converged_ionic"]:
            reasons.append("not_ionically_converged")
        if outcar_reason:
            reasons.append(outcar_reason)
        if not (dft_dir / "CONTCAR").exists():
            reasons.append("missing_CONTCAR")
        if not (dft_dir / "WAVECAR").exists():
            reasons.append("missing_WAVECAR")
        if not (dft_dir / "CHGCAR").exists():
            reasons.append("missing_CHGCAR")

        clean = (
            v["vasprun_ok"]
            and v["converged_electronic"]
            and v["converged_ionic"]
            and outcar_clean
        )

        rows.append({
            "status": "clean" if clean else "restart",
            "restart_needed": not clean,
            "model_label": row["model_label"],
            "cluster_id": int(row["cluster_id"]),
            "folder": folder,
            "final_energy_eV": v["final_energy_eV"],
            "max_final_force_eVA": v["max_final_force_eVA"],
            "n_ionic_steps": v["n_ionic_steps"],
            "vasprun_ok": v["vasprun_ok"],
            "converged_electronic": v["converged_electronic"],
            "converged_ionic": v["converged_ionic"],
            "reason": "ok" if clean else "; ".join(reasons),
            "dft_dir": str(dft_dir),
        })

    out = pd.DataFrame(rows)

    cols = [
        "status",
        "restart_needed",
        "model_label",
        "cluster_id",
        "folder",
        "final_energy_eV",
        "max_final_force_eVA",
        "n_ionic_steps",
        "vasprun_ok",
        "converged_electronic",
        "converged_ionic",
        "reason",
        "dft_dir",
    ]

    out = out[cols]
    out.to_csv(STATUS_CSV, index=False, float_format="%.10f")

    restart = out[out["restart_needed"]].copy()
    restart.to_csv(RESTART_CSV, index=False, float_format="%.10f")

    print("\nDFT job status:")
    print(
        out[
            [
                "status",
                "model_label",
                "cluster_id",
                "folder",
                "final_energy_eV",
                "max_final_force_eVA",
                "n_ionic_steps",
                "reason",
            ]
        ].to_string(index=False, max_colwidth=80)
    )

    print(f"\nWrote:\n  {STATUS_CSV}\n  {RESTART_CSV}")

    if not restart.empty:
        print("\nJobs needing restart:")
        print(restart[["model_label", "cluster_id", "folder", "reason"]].to_string(index=False, max_colwidth=100))
        raise RuntimeError("Some DFT validation jobs need restart.")

    print("\nAll DFT validation jobs are clean.")


if __name__ == "__main__":
    main()