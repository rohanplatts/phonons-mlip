#!/usr/bin/env bash
#SBATCH --job-name=orb_replay_ft
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
cd "$ROOT"

module purge
module load miniforge/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XDG_CACHE_HOME="/scratch/user/$USER/.cache"
export HF_HOME="/scratch/user/$USER/huggingface"
export TORCH_HOME="/scratch/user/$USER/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$ROOT/results/orb"

PREFIX="all_neb_data_300_samples"
DB_DIR="$ROOT/assets/training_data/curated_data/orb_ase_db"
NAME="${PREFIX}_orb_replay_v1"
BASE_MODEL="orb-v3-conservative-inf-omat"
OUTPUT_DIR="$ROOT/results/orb"
WEIGHTS_PATH=""
REPLAY_DB=""
REPLAY_RATIO="1.0"
TRAINABLE_REFERENCE_ENERGIES="0"
NO_AUGMENTATION="0"

TRAIN_DB="${DB_DIR}/${PREFIX}_train.db"
VALID_DB="${DB_DIR}/${PREFIX}_val.db"
TEST_DB="${DB_DIR}/${PREFIX}_test.db"

for db in "$TRAIN_DB" "$VALID_DB" "$TEST_DB"; do
  if [[ ! -f "$db" ]]; then
    echo "Missing ORB DB: $db" >&2
    exit 1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

args=(
  --run-name "$NAME"
  --output-dir "$OUTPUT_DIR"
  --base-model "$BASE_MODEL"
  --train-db "$TRAIN_DB"
  --valid-db "$VALID_DB"
  --test-db "$TEST_DB"
  --batch-size 2
  --eval-batch-size 2
  --max-epochs 20
  --num-steps 100
  --num-workers 4
  --lr 1e-4
  --weight-decay 5e-7
  --energy-loss-weight 40.0
  --forces-loss-weight 100.0
  --stress-loss-weight 0.0
  --confidence-loss-weight 0.0
  --device-id 0
  --random-seed 7
)

if [[ -n "$WEIGHTS_PATH" ]]; then
  args+=(--weights-path "$WEIGHTS_PATH")
fi
if [[ -n "$REPLAY_DB" ]]; then
  args+=(--replay-db "$REPLAY_DB" --replay-ratio "$REPLAY_RATIO")
else
  echo "No REPLAY_DB set; this will run target-only fine-tuning." >&2
fi
if [[ "$TRAINABLE_REFERENCE_ENERGIES" == "1" ]]; then
  args+=(--trainable-reference-energies)
fi
if [[ "$NO_AUGMENTATION" == "1" ]]; then
  args+=(--no-augmentation)
fi

exec srun python -u "$SCRIPT_DIR/train_orb.py" "${args[@]}" "$@"
