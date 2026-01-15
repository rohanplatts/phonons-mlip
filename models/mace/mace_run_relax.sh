#!/bin/bash --login
#SBATCH --job-name=mace_relax
#SBATCH --account=a_smp
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=relax_%j.out
#SBATCH --error=relax_%j.err


module purge # clean slate

module load cuda/12.1
module load anaconda3

# apparently this part is important. #TODO: Understand why line 22 is required.
source "$(conda info --base)/THIS_SHOULD_BE_A_PATH_IDK_WHAT_THO"

conda activate mace_env

# ======================
# Safety / debug info
# ======================

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ======================
# Scratch setup
# ======================

# make directory to run script

SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH_DIR"

# we need the scripts, obviously, and the model config + structures to run on.
cp -r "$SLURM_SUBMIT_DIR/scripts" "$SCRATCH_DIR/"
cp -r "$SLURM_SUBMIT_DIR/models" "$SCRATCH_DIR/"
cp -r "$SLURM_SUBMIT_DIR/structures" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR" || exit 1


# run relax.py #TODO: insert error catches.

python scfipts/relax.py --config models/mace/config.yaml


RESULTS_HOME="$SLURM_SUBMIT_DIR/results/mace/raw"
mkdir -p "$RESULTS_HOME"

cp -r results/* "$RESULTS_HOME/"

