#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ORB_CONDA_ENV:-orb_env}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export HF_HOME="${HF_HOME:-$ROOT/.huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$ROOT/results/orb"

PREFIX="${PREFIX:-all_neb_data_300_samples}"
DATA_DIR="${DATA_DIR:-${ROOT}/assets/training_data/curated_data/neutral_model}"
DB_DIR="${DB_DIR:-${ROOT}/assets/training_data/curated_data/orb_ase_db}"
NAME="${NAME:-${PREFIX}_orb_replay_v1}"
BASE_MODEL="${BASE_MODEL:-orb-v3-conservative-inf-omat}"
export PREFIX DATA_DIR DB_DIR

if [[ -n "${REPLAY_EXTXYZ:-}" && -z "${REPLAY_DB:-}" ]]; then
  REPLAY_DB="${DB_DIR}/$(basename "${REPLAY_EXTXYZ%.*}").db"
  export REPLAY_DB
fi

bash "$SCRIPT_DIR/prepare_orb_dbs.sh"

TRAIN_DB="${TRAIN_DB:-${DB_DIR}/${PREFIX}_train.db}"
VALID_DB="${VALID_DB:-${DB_DIR}/${PREFIX}_val.db}"
TEST_DB="${TEST_DB:-${DB_DIR}/${PREFIX}_test.db}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

args=(
  --run-name "$NAME"
  --output-dir "${OUTPUT_DIR:-$ROOT/results/orb}"
  --base-model "$BASE_MODEL"
  --train-db "$TRAIN_DB"
  --valid-db "$VALID_DB"
  --test-db "$TEST_DB"
  --batch-size "${BATCH_SIZE:-2}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-2}"
  --max-epochs "${MAX_EPOCHS:-20}"
  --num-steps "${NUM_STEPS:-100}"
  --num-workers "${NUM_WORKERS:-4}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-5e-7}"
  --energy-loss-weight "${ENERGY_LOSS_WEIGHT:-40.0}"
  --forces-loss-weight "${FORCES_LOSS_WEIGHT:-100.0}"
  --stress-loss-weight "${STRESS_LOSS_WEIGHT:-0.0}"
  --confidence-loss-weight "${CONFIDENCE_LOSS_WEIGHT:-0.0}"
  --device-id "${DEVICE_ID:-0}"
  --random-seed "${SEED:-7}"
)

if [[ -n "${WEIGHTS_PATH:-}" ]]; then
  args+=(--weights-path "$WEIGHTS_PATH")
fi
if [[ -n "${REPLAY_DB:-}" ]]; then
  args+=(--replay-db "$REPLAY_DB" --replay-ratio "${REPLAY_RATIO:-1.0}")
else
  echo "No REPLAY_DB or REPLAY_EXTXYZ set; this will run target-only fine-tuning." >&2
fi
if [[ "${TRAINABLE_REFERENCE_ENERGIES:-0}" == "1" ]]; then
  args+=(--trainable-reference-energies)
fi
if [[ "${NO_AUGMENTATION:-0}" == "1" ]]; then
  args+=(--no-augmentation)
fi

exec python -u "$SCRIPT_DIR/train_orb.py" "${args[@]}" "$@"
