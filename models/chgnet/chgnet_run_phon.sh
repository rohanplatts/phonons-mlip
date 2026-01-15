#!/bin/bash --login
#SBATCH --job-name=mace_relax
#SBATCH --account=a_smp
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=relax_%j.out
#SBATCH --error=relax_%j.err
#SBATCH --qos=normal


#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=CHANGE-ME
#SBATCH --time=1:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --account=AccountString
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

module-loads-go-here

srun executable < input > output