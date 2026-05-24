#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libgomp.so.1" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

cd "$(dirname "$0")"

QPLUS_ANALYSIS_NAME="${1:-primary_cs_pbi2br_q+1_snb}"
NEB_ANALYSIS_NAME="${2:-cspbi3_neb_endpoint_preservation}"
RUNS_ROOT="/home/rnpla/projects/mlip_phonons/src/defect_landscape/runs"

run_stage() {
  local script_name="$1"
  local analysis_name="$2"
  local run_dir="${RUNS_ROOT}/${analysis_name}"
  local skip_prepare="${SKIP_PREPARE:-0}"

  if [[ "${RESET:-0}" == "1" ]]; then
    skip_prepare="0"
  elif [[ "$skip_prepare" != "1" && -d "$run_dir" ]]; then
    skip_prepare="1"
    echo "Existing analysis folder found; continuing from staged inputs:"
    echo "  $run_dir"
  fi

  SKIP_PREPARE="$skip_prepare" "./${script_name}" "$analysis_name"
}

echo "Running q+1 CsPbI2Br SnB analysis:"
echo "  ${QPLUS_ANALYSIS_NAME}"
run_stage run_qplus_snb.sh "$QPLUS_ANALYSIS_NAME"

echo "Running CsPbI3 NEB endpoint-preservation analysis:"
echo "  ${NEB_ANALYSIS_NAME}"
run_stage run_cspbi3_neb_endpoints.sh "$NEB_ANALYSIS_NAME"

echo "Completed both analyses."
