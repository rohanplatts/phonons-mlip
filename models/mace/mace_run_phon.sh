#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=mace_relax
#SBATCH --time=1:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_smp
#SBATCH -o phon-%j.output
#SBATCH -e phon-%j.error


module purge # clean slate

module load cuda/12.1
module load anaconda3
