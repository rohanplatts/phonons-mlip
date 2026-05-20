#!/usr/bin/env bash
set -euo pipefail

# Fine-tune MACE from the pretrained foundation model on the IVac0 NEB dataset.
#
# Edit the variables below once, then re-run this script whenever you want.
# If you want to override/append flags without editing: pass them as arguments:
#   ./src/fine_tuning/train_ivac0_neb.sh --dry_run

# Output run name (used for output directories).
NAME="total_70_bias"

# Foundation model to start from.
# Use the float32 copy for speed; switch back to mace-mpa-0-medium.model if needed.
FOUNDATION_MODEL="assets/models/mace/mace-mpa-0-medium-f32.model"
DATA_DIR="assets/training_data/curated_data/neutral_model"
PREFIX="total_70_bias"


TRAIN_FILE="${DATA_DIR}/${PREFIX}_train.extxyz"
VALID_FILE="${DATA_DIR}/${PREFIX}_val.extxyz"
TEST_FILE="${DATA_DIR}/${PREFIX}_test.extxyz"

DEVICE="cuda"
DTYPE="float32"
BATCH_SIZE="2"
MAX_EPOCHS="30"

# key parameters
LEARNING_RATE="5.974111e-4"
LORA_RANK=16
LORA_ALPHA=16


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
