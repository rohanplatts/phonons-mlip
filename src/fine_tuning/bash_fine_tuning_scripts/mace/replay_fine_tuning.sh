#!/usr/bin/env bash
#SBATCH --job-name=replay_ft_neutral_hq
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

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

module purge
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate mace_env
python -m pip install --no-deps --upgrade mace-torch==0.3.15

export XDG_CACHE_HOME="/scratch/user/$USER/.cache"
export HF_HOME="/scratch/user/$USER/huggingface"
export TORCH_HOME="/scratch/user/$USER/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" checkpoints logs results downloads

nvidia-smi

NAME="neutral_charge_all_data_hq_replay_v1_mh"
FOUNDATION_MODEL="assets/models/mace/mace-mpa-0-medium-f32.model"
DATA_DIR="assets/training_data/0_model_prepared_training_data"
PREFIX="neutral_charge_all_data_hq_replay_v1"

TRAIN_FILE="${DATA_DIR}/${PREFIX}_train.extxyz"
VALID_FILE="${DATA_DIR}/${PREFIX}_val.extxyz"
TEST_FILE="${DATA_DIR}/${PREFIX}_test.extxyz"

exec srun python -u -m mace.cli.run_train \
  --name "${NAME}" \
  --foundation_model "${FOUNDATION_MODEL}" \
  --E0s foundation \
  --pt_train_file mp \
  --atomic_numbers "[55, 82, 53]" \
  --num_samples_pt 30000 \
  --filter_type_pt combinations \
  --subselect_pt fps \
  --multiheads_finetuning True \
  --weight_pt_head 1.0 \
  --train_file "${TRAIN_FILE}" \
  --valid_file "${VALID_FILE}" \
  --test_file "${TEST_FILE}" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --energy_weight 40.0 \
  --forces_weight 100.0 \
  --stress_weight 0.0 \
  --batch_size 4 \
  --valid_batch_size 8 \
  --max_num_epochs 20 \
  --patience 8 \
  --eval_interval 1 \
  --device cuda \
  --default_dtype float32 \
  --seed 7 \
  --lr 1e-4 \
  --force_mh_ft_lr True \
  --ema \
  --ema_decay 0.99 \
  --weight_decay 5e-7 \
  "$@"
