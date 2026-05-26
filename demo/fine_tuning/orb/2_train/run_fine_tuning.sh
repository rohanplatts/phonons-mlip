#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env2

export XDG_CACHE_HOME="$WORKFLOW_DIR/3_results/.cache"
export HF_HOME="$WORKFLOW_DIR/3_results/.huggingface"
export TORCH_HOME="$WORKFLOW_DIR/3_results/.torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$WORKFLOW_DIR/3_results"

PREFIX="orb_demo_neb_ft"
NAME="orb_demo_neb_ft_lora"
BASE_MODEL="orb-v3-conservative-inf-omat"
OUTPUT_DIR="$WORKFLOW_DIR/3_results"
TRAINABLE_REFERENCE_ENERGIES="0"
NO_AUGMENTATION="0"
LORA_TARGET_REGEX='(^model\.|^heads\.(energy|forces|stress)\.)'
LORA_EXCLUDE_REGEX='(reference|confidence)'
UNFREEZE_REGEX=""
REPLAY_DB=""
REPLAY_RATIO="1.0"
DB_DIR="$WORKFLOW_DIR/1_curated_data"

TRAIN_DB="$DB_DIR/${PREFIX}_train.db"
VALID_DB="$DB_DIR/${PREFIX}_val.db"
TEST_DB="$DB_DIR/${PREFIX}_test.db"

for db in "$TRAIN_DB" "$VALID_DB" "$TEST_DB"; do
  if [[ ! -f "$db" ]]; then
    echo "Missing ORB DB: $db" >&2
    exit 1
  fi
done

args=(
  --run-name "$NAME"
  --output-dir "$OUTPUT_DIR"
  --base-model "$BASE_MODEL"
  --train-db "$TRAIN_DB"
  --valid-db "$VALID_DB"
  --test-db "$TEST_DB"
  --batch-size 2
  --eval-batch-size 2
  --max-epochs 12
  --num-steps 50
  --num-workers 4
  --lr 1e-4
  --weight-decay 5e-7
  --energy-loss-weight 40.0
  --forces-loss-weight 100.0
  --stress-loss-weight 0.0
  --confidence-loss-weight 0.0
  --device-id 0
  --random-seed 7
  --lora
  --lora-rank 6
  --lora-alpha 6
  --lora-dropout 0.0
)

if [[ -n "$REPLAY_DB" ]]; then
  args+=(--replay-db "$REPLAY_DB" --replay-ratio "$REPLAY_RATIO")
fi
if [[ "$TRAINABLE_REFERENCE_ENERGIES" == "1" ]]; then
  args+=(--trainable-reference-energies)
fi
if [[ "$NO_AUGMENTATION" == "1" ]]; then
  args+=(--no-augmentation)
fi
if [[ -n "$LORA_TARGET_REGEX" ]]; then
  args+=(--lora-target-regex "$LORA_TARGET_REGEX")
fi
if [[ -n "$LORA_EXCLUDE_REGEX" ]]; then
  args+=(--lora-exclude-regex "$LORA_EXCLUDE_REGEX")
fi
if [[ -n "$UNFREEZE_REGEX" ]]; then
  args+=(--unfreeze-regex "$UNFREEZE_REGEX")
fi

exec python -u "$ROOT/src/fine_tuning/fine_tuning_scripts/orb/train_orb.py" "${args[@]}" "$@"
