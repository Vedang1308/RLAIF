#!/bin/bash
# Master script for Full RLAIF Evaluation

# Config
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
# Using LOCAL path is faster and avoids HF Auth/Subfolder errors since we are already on the cluster
TRAINED_ADAPTER="trainer_output/checkpoint-739" 
NUM_SAMPLES=100  # Set to 0 for FULL run (1300+ samples), 100 for fast check

echo "🚀 STARTING FULL EVALUATION PIPELINE"
echo "samples: $NUM_SAMPLES (0 = full)"

# 1. Run Baseline
echo ""
echo "---------------------------------------------------"
echo "📦 PHASE 1: Running Baseline Evaluation..."
echo "---------------------------------------------------"
python evaluate.py \
    --base_model "$BASE_MODEL" \
    --num_samples $NUM_SAMPLES \
    --output_file "eval_baseline.jsonl"

# 2. Run Trained
echo ""
echo "---------------------------------------------------"
echo "🧠 PHASE 2: Running RLAIF Trained Evaluation..."
echo "---------------------------------------------------"
python evaluate.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$TRAINED_ADAPTER" \
    --num_samples $NUM_SAMPLES \
    --output_file "eval_trained.jsonl"

# 3. Compare
echo ""
echo "---------------------------------------------------"
echo "📊 PHASE 3: Generating Comparison Report..."
echo "---------------------------------------------------"
python compare_evals.py eval_baseline.jsonl eval_trained.jsonl
