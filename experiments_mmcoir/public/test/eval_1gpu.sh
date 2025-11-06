#!/usr/bin/env bash

set -euo pipefail

# === Basic Config (you can override via env when calling) ===
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}

# Path to local MMCoIR test data root (images/jsonl etc.)
DATA_BASEDIR=${DATA_BASEDIR:-data/MMCoIR_test}

# YAML describing tasks to evaluate (MMCoIR test config)
DATA_CONFIG=${DATA_CONFIG:-experiments_mmcoir/public/test/image.yaml}

# Output base dir for embeddings, scores, and predictions
OUTPUT_BASEDIR=${OUTPUT_BASEDIR:-exps/mmcoir_test}

mkdir -p "$OUTPUT_BASEDIR"

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Data base dir: $DATA_BASEDIR"
echo "Dataset YAML: $DATA_CONFIG"
echo "Output base dir: $OUTPUT_BASEDIR"

# === Model Specs (edit as needed) ===
# Example: Qwen2-VL-2B-Instruct
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}
MODEL_TYPE=${MODEL_TYPE:-vlm}

# Sanitize model name for folder
MODEL_SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_' )
RUN_OUTPUT_DIR="$OUTPUT_BASEDIR/$MODEL_SAFE_NAME"
mkdir -p "$RUN_OUTPUT_DIR"

echo "Evaluating model: $MODEL_NAME"

python eval_legacy.py \
  --model_name "$MODEL_NAME" \
  --model_type "$MODEL_TYPE" \
  --dataset_config "$DATA_CONFIG" \
  --encode_output_path "$RUN_OUTPUT_DIR" \
  --data_basedir "$DATA_BASEDIR" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --dataloader_num_workers "$NUM_WORKERS"

echo "Done. Outputs saved under: $RUN_OUTPUT_DIR"