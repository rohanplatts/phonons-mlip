#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ORB_CONDA_ENV:-orb_env}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda activate "$ENV_NAME"
  python -m pip install --upgrade pip
  python -m pip install --upgrade orb-models wandb
else
  conda env create -n "$ENV_NAME" -f "${SCRIPT_DIR}/orb_env.yml"
  conda activate "$ENV_NAME"
fi

python - <<'PY'
import ase
import torch
import orb_models

print(f"orb_models: {getattr(orb_models, '__version__', 'unknown')} ({orb_models.__file__})")
print(f"torch: {torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"ase: {ase.__version__}")
PY
