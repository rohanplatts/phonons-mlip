#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libgomp.so.1" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

cd "$(dirname "$0")"

ANALYSIS_NAME="${1:-cspbi3_neb_endpoint_preservation}"
RUN_DIR="/home/rnpla/projects/mlip_phonons/src/defect_landscape/runs/${ANALYSIS_NAME}"
POSITIVE_MODEL_NAME="mace-mpa-0-medium-ft-cspbi3-positive"
NEGATIVE_MODEL_NAME="mace-mpa-0-medium-ft-cspbi3-negative"

cases_for_charge() {
  local charge="$1"
  python - "$ANALYSIS_NAME" "$charge" <<'PY'
import csv
import sys
from pathlib import Path

analysis_name, charge = sys.argv[1], sys.argv[2]
path = Path("/home/rnpla/projects/mlip_phonons/src/defect_landscape/runs") / analysis_name / "analysis" / "endpoint_case_metadata.csv"
with path.open(newline="") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if row["charge"] == charge:
        print(row["case_label"])
PY
}

run_model_all_cases() {
  local model_label="$1"
  local model_name="$2"
  local -a cmd=(python 01_relax_mlip.py --analysis-name "$ANALYSIS_NAME" --model "$model_label")
  if [[ -n "$model_name" ]]; then
    cmd+=(--model-name "$model_name")
  fi
  "${cmd[@]}"
}

run_model_for_cases() {
  local model_label="$1"
  local model_name="$2"
  shift 2
  local cases=("$@")
  if [[ "${#cases[@]}" -eq 0 ]]; then
    return
  fi
  local -a cmd=(python 01_relax_mlip.py --analysis-name "$ANALYSIS_NAME" --model "$model_label" --case-list "${cases[@]}")
  if [[ -n "$model_name" ]]; then
    cmd+=(--model-name "$model_name")
  fi
  "${cmd[@]}"
}

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  if [[ -d "$RUN_DIR" && "${RESET:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing analysis folder:"
    echo "  $RUN_DIR"
    echo "Choose a new analysis name, rerun with RESET=1, or use SKIP_PREPARE=1 to continue from staged inputs."
    exit 1
  fi

  python 11_prepare_neb_endpoint_cases.py --analysis-name "$ANALYSIS_NAME" --reset
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared CsPbI3 NEB endpoint cases under $RUN_DIR"
  echo "PREPARE_ONLY=1 was set, so MLIP relaxations and comparisons were not run."
  exit 0
fi

if [[ ! -f "${RUN_DIR}/analysis/endpoint_case_metadata.csv" ]]; then
  echo "Missing endpoint metadata. Run preparation first:"
  echo "  PREPARE_ONLY=1 $0 $ANALYSIS_NAME"
  exit 1
fi

mapfile -t QPLUS_CASES < <(cases_for_charge "+1")
mapfile -t QMINUS_CASES < <(cases_for_charge "-1")

run_model_all_cases base_mace ""
run_model_all_cases finetuned_mace ""
run_model_for_cases finetuned_mace_positive "$POSITIVE_MODEL_NAME" "${QPLUS_CASES[@]}"
run_model_for_cases finetuned_mace_negative "$NEGATIVE_MODEL_NAME" "${QMINUS_CASES[@]}"

python 12_compare_neb_endpoint_preservation.py \
  --analysis-name "$ANALYSIS_NAME" \
  --models base_mace finetuned_mace finetuned_mace_positive finetuned_mace_negative

python 13_write_neb_endpoint_report.py --analysis-name "$ANALYSIS_NAME"
