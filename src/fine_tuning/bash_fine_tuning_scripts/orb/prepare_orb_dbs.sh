#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

PREFIX="${PREFIX:-all_neb_data_300_samples}"
DATA_DIR="${DATA_DIR:-${ROOT}/assets/training_data/curated_data/neutral_model}"
DB_DIR="${DB_DIR:-${ROOT}/assets/training_data/curated_data/orb_ase_db}"
ENERGY_KEY="${ENERGY_KEY:-REF_energy}"
FORCES_KEY="${FORCES_KEY:-REF_forces}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$DB_DIR"

convert_split() {
  local split="$1"
  local input="${DATA_DIR}/${PREFIX}_${split}.extxyz"
  local output="${DB_DIR}/${PREFIX}_${split}.db"

  if [[ ! -f "$input" ]]; then
    echo "Missing ${split} extxyz: $input" >&2
    exit 1
  fi

  if [[ -f "$output" && "$OVERWRITE" != "1" ]]; then
    echo "Using existing ${split} DB: $output"
    return
  fi

  echo "Converting ${input} -> ${output}"
  local args=(
    --input "$input"
    --output "$output"
    --energy-key "$ENERGY_KEY"
    --forces-key "$FORCES_KEY"
    --overwrite
  )
  if [[ -n "${CHARGE:-}" ]]; then
    args+=(--charge "$CHARGE")
  fi
  if [[ -n "${SPIN:-}" ]]; then
    args+=(--spin "$SPIN")
  fi
  python "$SCRIPT_DIR/extxyz_to_orb_ase_db.py" "${args[@]}"
}

convert_split train
convert_split val
convert_split test

if [[ -n "${REPLAY_EXTXYZ:-}" ]]; then
  REPLAY_DB="${REPLAY_DB:-${DB_DIR}/$(basename "${REPLAY_EXTXYZ%.*}").db}"
  if [[ -f "$REPLAY_DB" && "$OVERWRITE" != "1" ]]; then
    echo "Using existing replay DB: $REPLAY_DB"
  else
    echo "Converting replay ${REPLAY_EXTXYZ} -> ${REPLAY_DB}"
    python "$SCRIPT_DIR/extxyz_to_orb_ase_db.py" \
      --input "$REPLAY_EXTXYZ" \
      --output "$REPLAY_DB" \
      --energy-key "$ENERGY_KEY" \
      --forces-key "$FORCES_KEY" \
      --overwrite
  fi
fi

cat <<EOF
ORB DB paths:
  TRAIN_DB=${DB_DIR}/${PREFIX}_train.db
  VALID_DB=${DB_DIR}/${PREFIX}_val.db
  TEST_DB=${DB_DIR}/${PREFIX}_test.db
EOF
if [[ -n "${REPLAY_DB:-}" ]]; then
  echo "  REPLAY_DB=${REPLAY_DB}"
fi
