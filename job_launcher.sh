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
# source activate myenv # Uncomment if using conda

echo "Starting RLAIF training job on $(hostname) at $(date)"

# Create directories
mkdir -p logs
mkdir -p checkpoints

# Run the training script
# We use accelerate for optimized launch
export PYTHONUNBUFFERED=1
accelerate launch train.py

echo "Job finished/interrupted at $(date)"
