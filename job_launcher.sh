#!/bin/bash
#SBATCH --job-name=rlaif-qwen-0.5b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00 # Example time limit, adjust as per slot
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Ensure configured environment
source ~/.bashrc
# Activate the specific conda environment found in logs
source activate nlp_fix_env || conda activate nlp_fix_env

echo "Starting RLAIF training job on $(hostname) at $(date)"

# Create directories
mkdir -p logs
mkdir -p checkpoints

# Run the training script
# We use accelerate for optimized launch
export PYTHONUNBUFFERED=1
# Enable WandB logging
export WANDB_API_KEY=ef2da50d021e41130a9c9d762f7e56c79dbed703
export WANDB_PROJECT=rlaif-qwen
export CUDA_LAUNCH_BLOCKING=1
# Using python3 directly to debug startup issues
python3 train.py 2>&1

echo "Job finished/interrupted at $(date)"
