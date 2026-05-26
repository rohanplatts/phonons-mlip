#!/usr/bin/env bash
#SBATCH --job-name=ivac0_neb_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

module purge
module load miniforge/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

NAME="ivac0_neb_ft"
FOUNDATION_MODEL="assets/models/mace/mace-mpa-0-medium-f32.model"
DATA_DIR="$ROOT/assets/training_data/CsPbI3/I_vac_0/processed_mace"
PREFIX="ivac0_neb_stride5"

TRAIN_FILE="${DATA_DIR}/${PREFIX}_train.extxyz"
VALID_FILE="${DATA_DIR}/${PREFIX}_val.extxyz"
TEST_FILE="${DATA_DIR}/${PREFIX}_test.extxyz"

DEVICE="cuda"
DTYPE="float32"
BATCH_SIZE="2"
MAX_EPOCHS="200"
LEARNING_RATE="5e-5"

exec python -m mace.cli.run_train \
  --name "${NAME}" \
  --foundation_model "${FOUNDATION_MODEL}" \
  --E0s foundation \
  --multiheads_finetuning False \
  --train_file "${TRAIN_FILE}" \
  --valid_file "${VALID_FILE}" \
  --test_file "${TEST_FILE}" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --energy_weight 20 \
  --forces_weight 100.0 \
  --stress_weight 0.0 \
  --batch_size "${BATCH_SIZE}" \
  --max_num_epochs "${MAX_EPOCHS}" \
  --patience 50 \
  --device "${DEVICE}" \
  --default_dtype "${DTYPE}" \
  --lr "${LEARNING_RATE}" \
  "$@"
