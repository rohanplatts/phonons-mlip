#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libgomp.so.1" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

cd "$(dirname "$0")"

ANALYSIS_NAME="${1:-primary_cs_pbi2br_q+1_snb}"
RUN_DIR="/home/rnpla/projects/mlip_phonons/src/defect_landscape/runs/${ANALYSIS_NAME}"
SNB_ROOT="/home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma"
POSITIVE_MODEL_NAME="mace-mpa-0-medium-ft-cspbi3-positive"

prepare_case() {
  local case_label="$1"
  local defect_dir="$2"
  python 00_prepare_case.py \
    --analysis-name "$ANALYSIS_NAME" \
    --case-label "$case_label" \
    --input-poscar "${SNB_ROOT}/${defect_dir}/q+1/${case_label}/input/POSCAR" \
    --dft-references-dir "${SNB_ROOT}/${defect_dir}/q+1/${case_label}/dft_references"
}

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  if [[ -d "$RUN_DIR" && "${RESET:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing analysis folder:"
    echo "  $RUN_DIR"
    echo "Choose a new analysis name, rerun with RESET=1, or use SKIP_PREPARE=1 to continue from staged inputs."
    exit 1
  fi

  python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --reset --init-empty

  prepare_case VBr_q+1_end_Br4c_test1 V_Br
  prepare_case VBr_q+1_end_Br4c_test2 V_Br
  prepare_case VBr_q+1_end_Br8d_test1 V_Br
  prepare_case VBr_q+1_end_Br8d_test2 V_Br
  prepare_case VBr_q+1_end_I4c_test1 V_Br
  prepare_case VBr_q+1_end_I4c_test2 V_Br
  prepare_case VBr_q+1_end_I8d_test1 V_Br
  prepare_case VBr_q+1_end_I8d_test2 V_Br
  prepare_case VBr_q+1_start_8d_test1 V_Br
  prepare_case VBr_q+1_start_8d_test2 V_Br

  prepare_case VI_q+1_end_Br4c_test1 V_I
  prepare_case VI_q+1_end_Br4c_test2 V_I
  prepare_case VI_q+1_end_Br8d_test1 V_I
  prepare_case VI_q+1_end_Br8d_test2 V_I
  prepare_case VI_q+1_end_I4c_test1 V_I
  prepare_case VI_q+1_end_I4c_test2 V_I
  prepare_case VI_q+1_end_I8d_test1 V_I
  prepare_case VI_q+1_end_I8d_test2 V_I
  prepare_case VI_q+1_start_8d_test1 V_I
  prepare_case VI_q+1_start_8d_test2 V_I
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared q+1 CsPbI2Br SnB cases under $RUN_DIR"
  echo "PREPARE_ONLY=1 was set, so MLIP relaxations and comparisons were not run."
  exit 0
fi

python 01_relax_mlip.py --analysis-name "$ANALYSIS_NAME" --model base_mace
python 01_relax_mlip.py --analysis-name "$ANALYSIS_NAME" --model finetuned_mace
python 01_relax_mlip.py \
  --analysis-name "$ANALYSIS_NAME" \
  --model finetuned_mace_positive \
  --model-name "$POSITIVE_MODEL_NAME"

python 02_compare_to_existing_dft.py \
  --analysis-name "$ANALYSIS_NAME" \
  --models base_mace finetuned_mace finetuned_mace_positive

python 03_write_carla_report.py --analysis-name "$ANALYSIS_NAME"
python 04_plot_energy_geometry_summary.py --analysis-name "$ANALYSIS_NAME"
