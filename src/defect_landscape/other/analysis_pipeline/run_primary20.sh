#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libgomp.so.1" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

cd "$(dirname "$0")"

ANALYSIS_NAME="${1:-primary_cs_pbi2br_q0_snb}"
RUN_DIR="/home/rnpla/projects/mlip_phonons/src/defect_landscape/runs/${ANALYSIS_NAME}"

if [[ -d "$RUN_DIR" && "${RESET:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing analysis folder:"
  echo "  $RUN_DIR"
  echo "Choose a new analysis name or rerun with RESET=1."
  exit 1
fi

python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --reset --init-empty

python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_Br4c_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br4c_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br4c_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_Br4c_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br4c_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br4c_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_Br8d_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br8d_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br8d_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_Br8d_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br8d_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_Br8d_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_I4c_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I4c_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I4c_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_I4c_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I4c_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I4c_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_I8d_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I8d_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I8d_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_end_I8d_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I8d_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_end_I8d_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_start_8d_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_start_8d_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_start_8d_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VBr_q0_start_8d_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_start_8d_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_Br/q0/VBr_q0_start_8d_test2/dft_references

python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_Br4c_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br4c_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br4c_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_Br4c_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br4c_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br4c_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_Br8d_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br8d_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br8d_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_Br8d_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br8d_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_Br8d_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_I4c_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I4c_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I4c_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_I4c_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I4c_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I4c_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_I8d_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I8d_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I8d_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_end_I8d_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I8d_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_end_I8d_test2/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_start_8d_test1 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_start_8d_test1/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_start_8d_test1/dft_references
python 00_prepare_case.py --analysis-name "$ANALYSIS_NAME" --case-label VI_q0_start_8d_test2 --input-poscar /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_start_8d_test2/input/POSCAR --dft-references-dir /home/rnpla/projects/mlip_phonons/assets/SNB_data/CsPbI2Br/gamma/V_I/q0/VI_q0_start_8d_test2/dft_references

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared primary 20 under $RUN_DIR"
  echo "PREPARE_ONLY=1 was set, so MLIP relaxations and comparisons were not run."
  exit 0
fi

python 01_relax_mlip.py --analysis-name "$ANALYSIS_NAME" --model base_mace
python 01_relax_mlip.py --analysis-name "$ANALYSIS_NAME" --model finetuned_mace
python 02_compare_to_existing_dft.py --analysis-name "$ANALYSIS_NAME"
python 03_write_carla_report.py --analysis-name "$ANALYSIS_NAME"
