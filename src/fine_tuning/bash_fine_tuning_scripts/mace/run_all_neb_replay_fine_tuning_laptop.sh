#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LORA_TRAIN_SCRIPT="${SCRIPT_DIR}/naive_lora_fine_tuning_laptop.sh"
REPLAY_TRAIN_SCRIPT="${SCRIPT_DIR}/replay_fine_tuning_laptop.sh"
LOG_DIR="${ROOT}/logs/neb_laptop_batch"
LORA_LOG_DIR="${LOG_DIR}/naive_lora"
REPLAY_LOG_DIR="${LOG_DIR}/replay"

PREFIXES=(
  "all_neb_data_50_samples"
  "all_neb_data_100_samples"
  "all_neb_data_300_samples"
  "all_neb_data_600_samples"
  "all_neb_data_1200_samples"
)

REPLAY_PREFIXES=(
  "all_neb_data_1200_samples"
  "all_neb_data_100_samples"
)

mkdir -p "$LORA_LOG_DIR" "$REPLAY_LOG_DIR"

failures=()

handle_interrupt() {
  echo "Interrupted; stopping NEB batch run." >&2
  exit 130
}

trap handle_interrupt INT TERM

for prefix in "${PREFIXES[@]}"; do
  name="${prefix}_naive_lora"
  log_file="${LORA_LOG_DIR}/${prefix}.log"

  echo "============================================================"
  echo "Starting naive LoRA fine-tuning for ${prefix}"
  echo "Run name: ${name}"
  echo "Log file: ${log_file}"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if PREFIX="$prefix" NAME="$name" bash "$LORA_TRAIN_SCRIPT" 2>&1 | tee "$log_file"; then
    echo "Completed naive LoRA ${prefix} at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    status=$?
    if ((status >= 128)); then
      echo "Interrupted during naive LoRA ${prefix}; stopping NEB batch run." >&2
      exit "$status"
    fi
    echo "Failed naive LoRA ${prefix} at $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failures+=("naive_lora:${prefix}")
  fi
done

for prefix in "${REPLAY_PREFIXES[@]}"; do
  name="${prefix}_replay_v1_mh"
  log_file="${REPLAY_LOG_DIR}/${prefix}.log"

  echo "============================================================"
  echo "Starting replay fine-tuning for ${prefix}"
  echo "Run name: ${name}"
  echo "Log file: ${log_file}"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if PREFIX="$prefix" NAME="$name" bash "$REPLAY_TRAIN_SCRIPT" 2>&1 | tee "$log_file"; then
    echo "Completed ${prefix} at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    status=$?
    if ((status >= 128)); then
      echo "Interrupted during replay ${prefix}; stopping NEB batch run." >&2
      exit "$status"
    fi
    echo "Failed ${prefix} at $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failures+=("replay:${prefix}")
  fi
done

echo "============================================================"
if ((${#failures[@]} == 0)); then
  echo "All NEB LoRA and replay fine-tuning runs completed successfully."
  exit 0
fi

echo "NEB fine-tuning finished with failures in:"
for prefix in "${failures[@]}"; do
  echo "  - ${prefix}"
done
exit 1
