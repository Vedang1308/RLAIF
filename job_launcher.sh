#!/bin/bash
#SBATCH --job-name=rlaif-qwen-0.5b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00 # Long duration for full training

# ... (environment setup unchanged) ...
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Ensure configured environment
source ~/.bashrc
# Activate the specific conda environment found in logs
source activate nlp_fix_env || conda activate nlp_fix_env

echo "Starting RLAIF training job on $(hostname) at $(date)"

# Create directories
mkdir -p logs
mkdir -p trainer_output

# Run the training script
# We use accelerate for optimized launch
export PYTHONUNBUFFERED=1
# Enable WandB logging (OFFLINE to prevent network timeouts)
export WANDB_API_KEY=ef2da50d021e41130a9c9d762f7e56c79dbed703
export WANDB_PROJECT=rlaif-research
export WANDB_MODE=offline # Critical for compute nodes without internet

# Using python3 directly to debug startup issues
# Run in RESEARCH mode for best results
python3 train.py --mode research --num_epochs 4 --save_freq 50 --push_repo_id "vedang1308/RLAIF-Qwen" 2>&1

echo "Training complete. Uploading to Hugging Face..."
# Auto-upload to HF (using your token and repo)
python3 push_to_hub.py \
  --repo_id "vedang1308/RLAIF-Qwen" \
  --private

echo "Job finished at $(date)"
