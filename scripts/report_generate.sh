#!/bin/bash

# Batch evaluation script
# Runs multiple models (default: with tools enabled)

# Base configuration
BASE_CONFIG="configs/report_generate.yaml"

# Models to evaluate
MODELS=(
    "openai/gpt-4.1"
    "deepseek/deepseek-v3.2"
    "meta-llama/llama-4-maverick"
    "qwen/qwen3-max"
    "qwen/qwen3-235b-a22b-2507"
    "anthropic/claude-opus-4.5"
    "anthropic/claude-sonnet-4.5"
    "openai/gpt-5.2"
)

# Run each model
for model in "${MODELS[@]}"; do
    # Extract model name for output directory (replace / with _)
    model_safe=$(echo "$model" | sed 's/\//_/g' | sed 's/:/_/g')
    
    echo "============================================================"
    echo "Running: $model"
    echo "============================================================"
    
    echo ""
    echo ">>> Running $model..."
    python3 scripts/report_generate.py \
        --config "$BASE_CONFIG" \
        --model "$model" \
        --output-dir "cache/$model_safe"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to run $model"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed $model"
    echo ""
done

echo "============================================================"
echo "All evaluations completed!"
echo "============================================================"

