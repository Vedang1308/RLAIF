#!/bin/bash
# Master script for Full RLAIF Evaluation

# Config
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
# Comparison: Eval the model hosted on Hugging Face (Must re-upload checkpoint-739 first!)
# Comparison: Eval the LOCAL model we just trained
TRAINED_ADAPTER="trainer_output"
NUM_SAMPLES=0  # 0 = FULL run (1319 samples). Set to 100 for fast debugging.

echo "🚀 STARTING FULL EVALUATION PIPELINE"
echo "samples: $NUM_SAMPLES (0 = full)"

# 1. Run Baseline
echo ""
echo "---------------------------------------------------"
echo "📦 PHASE 1: Baseline Evaluation"
echo "---------------------------------------------------"
if [ -f "eval_baseline.jsonl" ]; then
    echo "✅ Found existing 'eval_baseline.jsonl'. Skipping re-run."
else
    echo "▶️ Running Baseline Evaluation..."
    python -u evaluate.py \
        --base_model "$BASE_MODEL" \
        --num_samples $NUM_SAMPLES \
        --output_file "eval_baseline.jsonl"
fi

# 2. Run Trained
echo ""
echo "---------------------------------------------------"
echo "🧠 PHASE 2: Running RLAIF Trained Evaluation..."
echo "---------------------------------------------------"
# Delete old result to ensure freshness
if [ -f "eval_trained.jsonl" ]; then
    echo "🗑️  Removing old 'eval_trained.jsonl'..."
    rm eval_trained.jsonl
fi
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
python -u compare_evals.py eval_baseline.jsonl eval_trained.jsonl
