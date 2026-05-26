#!/usr/bin/env bash
#SBATCH --job-name=petmad_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module purge
module load miniforge/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env2

export XDG_CACHE_HOME="$SCRIPT_DIR/results/.cache"
export HF_HOME="$SCRIPT_DIR/results/.huggingface"
export TORCH_HOME="$SCRIPT_DIR/results/.torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$SCRIPT_DIR/results/petmad/extensions"

OUTPUT_MODEL="results/petmad/model.pt"
EXTENSIONS_DIR="results/petmad/extensions"
BASE_MODEL="$SCRIPT_DIR/pet-mad-v1.1.0.ckpt"
BASE_MODEL_URL="https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
CONFIG_FILE="$SCRIPT_DIR/petmad_options.yml"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing PET-MAD config: $CONFIG_FILE" >&2
  exit 1
fi

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

if [[ ! -d "$SCRIPT_DIR/selected_data" ]]; then
  echo "Missing PET-MAD selected_data directory: $SCRIPT_DIR/selected_data" >&2
  echo "Run the stage 2 demo handoff first so the curated extxyz files are linked next to the family script directory." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_MODEL")" "$EXTENSIONS_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

cd "$SCRIPT_DIR"

exec srun mtt train "$CONFIG_FILE" \
  -o "$OUTPUT_MODEL" \
  -e "$EXTENSIONS_DIR" \
  "$@"
