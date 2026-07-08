#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

export XDG_CACHE_HOME="$WORKFLOW_DIR/3_results/.cache"
export HF_HOME="$WORKFLOW_DIR/3_results/.huggingface"
export TORCH_HOME="$WORKFLOW_DIR/3_results/.torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$WORKFLOW_DIR/3_results/full_extensions"

BASE_MODEL="$WORKFLOW_DIR/3_results/pet-mad-v1.1.0.ckpt"
BASE_MODEL_URL="https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
CONFIG_FILE="$SCRIPT_DIR/petmad_full_options.yml"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing PET-MAD full fine-tuning config: $CONFIG_FILE" >&2
  exit 1
fi

mkdir -p "$WORKFLOW_DIR/3_results"
rm -rf "$WORKFLOW_DIR/3_results/selected_data"
ln -s "$WORKFLOW_DIR/1_curated_data" "$WORKFLOW_DIR/3_results/selected_data"

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Downloading PET-MAD finetuning checkpoint to $BASE_MODEL"
  python - "$BASE_MODEL_URL" "$BASE_MODEL" <<'PY'
import pathlib
import shutil
import sys
import urllib.request

url = sys.argv[1]
dst = pathlib.Path(sys.argv[2])
tmp = dst.with_suffix(dst.suffix + ".tmp")
with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
  shutil.copyfileobj(response, handle)
tmp.replace(dst)
PY
fi

cd "$WORKFLOW_DIR/3_results"

exec mtt train "$CONFIG_FILE" \
  -o "model_full.pt" \
  -e "full_extensions" \
  "$@"
