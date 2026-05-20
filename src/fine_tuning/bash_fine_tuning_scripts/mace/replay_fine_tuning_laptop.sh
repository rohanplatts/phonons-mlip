#!/usr/bin/env bash
set -euo pipefail

# Local laptop version of the replay fine-tuning run.
# This removes SLURM/module assumptions and runs the training command directly.

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env


# Keep cache and output directories local to the repo by default.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export HF_HOME="${HF_HOME:-$ROOT/.huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" checkpoints logs results downloads

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

PREFIX="${PREFIX:-total_70_bias}"
NAME="${NAME:-neutral_charge_70_total_hq_replay_v1_mh}"
FOUNDATION_MODEL="${FOUNDATION_MODEL:-assets/models/mace/mace-mpa-0-medium-f32.model}"
DATA_DIR="${DATA_DIR:-$ROOT/assets/training_data/curated_data/neutral_model}"

TRAIN_FILE="${DATA_DIR}/${PREFIX}_train.extxyz"
VALID_FILE="${DATA_DIR}/${PREFIX}_val.extxyz"
TEST_FILE="${DATA_DIR}/${PREFIX}_test.extxyz"

for data_file in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
  if [[ ! -f "$data_file" ]]; then
    echo "Missing dataset file: $data_file" >&2
    exit 1
  fi
done

# Laptop-friendly defaults:
# - no srun
# - fewer workers implied by MACE defaults
# - keep the same data and optimization settings as the cluster job
exec python -u -m mace.cli.run_train \
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
