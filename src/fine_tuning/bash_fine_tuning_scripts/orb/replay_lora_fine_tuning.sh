#!/usr/bin/env bash
#SBATCH --job-name=orb_lora_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

module purge
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate "${ORB_CONDA_ENV:-orb_env}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/user/$USER/.cache}"
export HF_HOME="${HF_HOME:-/scratch/user/$USER/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/scratch/user/$USER/torch}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$ROOT/results/orb"

nvidia-smi

exec srun bash "$SCRIPT_DIR/lora_fine_tuning_laptop.sh" "$@"
