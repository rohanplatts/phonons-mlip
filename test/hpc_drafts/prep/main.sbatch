#!/usr/bin/env bash
#SBATCH --job-name=neb_mace_mpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

REPO_DIR="/scratch/user/s4802880/mlip_phonons"
ENV_NAME="mace_env"
MODEL_NAME="mace-mpa-0-medium"

cd "$REPO_DIR"

module purge
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"

export XDG_CACHE_HOME="/scratch/user/$USER/.cache"
export HF_HOME="/scratch/user/$USER/huggingface"
export TORCH_HOME="/scratch/user/$USER/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"

conda activate "$ENV_NAME"

# D3 can require libgomp from the active conda environment.
if [ -f "$CONDA_PREFIX/lib/libgomp.so.1" ]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

nvidia-smi

srun mlip-neb "$MODEL_NAME" --config config.yml --device cuda --dtype float32

conda deactivate
