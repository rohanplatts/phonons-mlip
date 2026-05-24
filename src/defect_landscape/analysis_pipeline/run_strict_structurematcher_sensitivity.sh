#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libgomp.so.1" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1"
fi
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

cd "$(dirname "$0")"

python 14_strict_structurematcher_sensitivity.py \
  --analysis-name \
    primary_cs_pbi2br_q0_snb \
    primary_cs_pbi2br_q+1_snb
