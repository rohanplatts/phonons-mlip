#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${ROOT}/logs/orb_lora_batch"

PREFIXES=(
  "all_neb_data_50_samples"
  "all_neb_data_100_samples"
  "all_neb_data_300_samples"
  "all_neb_data_600_samples"
  "all_neb_data_1200_samples"
)

mkdir -p "$LOG_DIR"
failures=()

for prefix in "${PREFIXES[@]}"; do
  name="${prefix}_orb_lora_v1"
  log_file="${LOG_DIR}/${prefix}.log"

  echo "============================================================"
  echo "Starting ORB LoRA fine-tuning for ${prefix}"
  echo "Run name: ${name}"
  echo "Log file: ${log_file}"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if PREFIX="$prefix" NAME="$name" bash "$SCRIPT_DIR/lora_fine_tuning_laptop.sh" 2>&1 | tee "$log_file"; then
    echo "Completed ${prefix} at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "Failed ${prefix} at $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failures+=("$prefix")
  fi
done

echo "============================================================"
if ((${#failures[@]} == 0)); then
  echo "All ORB LoRA fine-tuning runs completed successfully."
  exit 0
fi

echo "ORB LoRA fine-tuning finished with failures in:"
for prefix in "${failures[@]}"; do
  echo "  - ${prefix}"
done
exit 1
