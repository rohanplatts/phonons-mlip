#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export HF_HOME="${HF_HOME:-$ROOT/.huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" checkpoints logs results downloads

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

PREFIX="${PREFIX:-total_70_bias}"
NAME="${NAME:-${PREFIX}_naive_lora}"
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

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"

LEARNING_RATE="${LEARNING_RATE:-5.974111e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"

exec python -u -m mace.cli.run_train \
  --name "${NAME}" \
  --foundation_model "${FOUNDATION_MODEL}" \
  --E0s foundation \
  --multiheads_finetuning False \
  --train_file "${TRAIN_FILE}" \
  --valid_file "${VALID_FILE}" \
  --test_file "${TEST_FILE}" \
  --lora=True \
  --lora_rank="${LORA_RANK}" \
  --lora_alpha="${LORA_ALPHA}" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --energy_weight 20 \
  --forces_weight 100.0 \
  --stress_weight 0.0 \
  --batch_size "${BATCH_SIZE}" \
  --max_num_epochs "${MAX_EPOCHS}" \
  --patience 8 \
  --device "${DEVICE}" \
  --default_dtype "${DTYPE}" \
  --lr "${LEARNING_RATE}" \
  "$@"
