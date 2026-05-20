#!/usr/bin/env bash
#SBATCH --job-name=neg_mace_replay_ft_400_5000_samples
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

WORKDIR="/scratch/user/s4802880/mlip_phonons/replay_ft"
TRAIN_DIR="${WORKDIR}/training_data"
FOUNDATION_MODEL="/scratch/user/s4802880/mlip_phonons/assets/models/mace/mace-mpa-0-medium-f32.model"
NAME="negative_400_samples_replay_v1_mh"
RUN_DIR="${WORKDIR}/runs/${NAME}"

module purge
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate mace_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export XDG_CACHE_HOME="/scratch/user/$USER/.cache"
export HF_HOME="/scratch/user/$USER/huggingface"
export TORCH_HOME="/scratch/user/$USER/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$RUN_DIR"
cd "$RUN_DIR"
mkdir -p checkpoints logs results downloads

nvidia-smi

PREFIX="negative_all_neb_data_400_samples"


TRAIN_FILE="${TRAIN_DIR}/${PREFIX}_train.extxyz"
VALID_FILE="${TRAIN_DIR}/${PREFIX}_val.extxyz"
TEST_FILE="${TRAIN_DIR}/${PREFIX}_test.extxyz"

python - <<'PY'
import os
import sys

import numpy as np

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"NumPy import: {np.__version__} ({np.__file__})")
if not hasattr(np, "ndarray"):
    raise RuntimeError(
        "Imported 'numpy' does not expose numpy.ndarray. "
        "This usually means the job is picking up a shadowing "
        "numpy.py/numpy/ from the working directory, or the "
        "NumPy install in mace_env is broken."
    )

import torch

print(f"Torch import: {torch.__version__} ({torch.__file__})")
import mace

print(f"MACE import: {mace.__version__} ({mace.__file__})")
import scipy.fft
import scipy.signal

print("SciPy import: OK")
PY

exec srun python -u -m mace.cli.run_train \
  --name "${NAME}" \
  --foundation_model "${FOUNDATION_MODEL}" \
  --E0s foundation \
  --pt_train_file mp \
  --atomic_numbers "[55, 82, 53]" \
  --num_samples_pt 5000 \
  --filter_type_pt combinations \
  --subselect_pt fps \
  --multiheads_finetuning True \
  --weight_pt_head 1.0 \
  --train_file "${TRAIN_FILE}" \
  --valid_file "${VALID_FILE}" \
  --test_file "${TEST_FILE}" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --energy_weight 40.0 \
  --forces_weight 100.0 \
  --stress_weight 0.0 \
  --batch_size 4 \
  --valid_batch_size 8 \
  --max_num_epochs 30 \
  --patience 8 \
  --eval_interval 1 \
  --device cuda \
  --default_dtype float32 \
  --seed 7 \
  --lr 1e-4 \
  --force_mh_ft_lr True \
  --ema \
  --ema_decay 0.99 \
  --weight_decay 5e-7 \
  "$@"
