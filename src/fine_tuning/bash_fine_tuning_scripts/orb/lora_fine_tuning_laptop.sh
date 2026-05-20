#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NAME="${NAME:-${PREFIX:-all_neb_data_300_samples}_orb_lora_v1}"
export NAME

exec bash "$SCRIPT_DIR/replay_fine_tuning_laptop.sh" \
  --lora \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  "$@"
