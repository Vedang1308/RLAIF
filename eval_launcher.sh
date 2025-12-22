#!/bin/bash
#SBATCH --job-name=rlaif-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=eval_logs/job_%j.out
#SBATCH --error=eval_logs/job_%j.err

# Safe Evaluation Launcher for Slurm Cluster
# This ensures evaluation runs on a GPU node, not the login node.

# 1. Environment
source ~/.bashrc
source activate nlp_fix_env || conda activate nlp_fix_env

echo "Starting Evaluation Job on $(hostname) at $(date)"
mkdir -p eval_logs

# 2. Run the master evaluation script
# Ensure it is executable
chmod +x run_eval.sh
./run_eval.sh

echo "Evaluation Job finished at $(date)"
