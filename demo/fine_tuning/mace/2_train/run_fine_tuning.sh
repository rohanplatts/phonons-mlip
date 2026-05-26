#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

export XDG_CACHE_HOME="$WORKFLOW_DIR/3_results/.cache"
export HF_HOME="$WORKFLOW_DIR/3_results/.huggingface"
export TORCH_HOME="$WORKFLOW_DIR/3_results/.torch"
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$WORKFLOW_DIR/3_results/checkpoints" \
  "$WORKFLOW_DIR/3_results/logs" \
  "$WORKFLOW_DIR/3_results/results" \
  "$WORKFLOW_DIR/3_results/downloads"

PREFIX="mace_demo_neb_ft"
NAME="mace_demo_neb_ft_lora"
FOUNDATION_MODEL="$ROOT/assets/models/mace/mace-mpa-0-medium-f32.model"
DATA_DIR="$WORKFLOW_DIR/1_curated_data"

TRAIN_FILE="$DATA_DIR/${PREFIX}_train.extxyz"
VALID_FILE="$DATA_DIR/${PREFIX}_val.extxyz"
TEST_FILE="$DATA_DIR/${PREFIX}_test.extxyz"

for data_file in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
  if [[ ! -f "$data_file" ]]; then
    echo "Missing dataset file: $data_file" >&2
    exit 1
  fi
done

cd "$WORKFLOW_DIR/3_results"

exec python -u -m mace.cli.run_train \
  --name "$NAME" \
  --foundation_model "$FOUNDATION_MODEL" \
  --E0s foundation \
  --multiheads_finetuning False \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --test_file "$TEST_FILE" \
  --lora=True \
  --lora_rank 4 \
  --lora_alpha 4 \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --energy_weight 50 \
  --forces_weight 100.0 \
  --stress_weight 0.0 \
  --batch_size 2 \
  --max_num_epochs 12 \
  --patience 8 \
  --device cuda \
  --default_dtype float32 \
  --lr 5.974111e-4 \
  "$@"
