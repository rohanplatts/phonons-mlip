#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREFIX="${PREFIX:-all_neb_data_50_samples}"
NAME="${NAME:-${PREFIX}_orb_quick_test}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
NUM_STEPS="${NUM_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

export PREFIX NAME MAX_EPOCHS NUM_STEPS BATCH_SIZE EVAL_BATCH_SIZE NUM_WORKERS

exec bash "$SCRIPT_DIR/replay_fine_tuning_laptop.sh" --max-eval-batches "${MAX_EVAL_BATCHES:-1}" "$@"
