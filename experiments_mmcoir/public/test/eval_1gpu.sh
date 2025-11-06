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

# === Model Specs (edit as needed; format: "MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH") ===
declare -a MODEL_SPECS
# 示例：与 experiments/public/eval/eval_1gpu.sh 保持一致的写法
MODEL_SPECS+=( "VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B" )

for spec in "${MODEL_SPECS[@]}"; do
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH <<< "$spec"
  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "  - Backbone: $MODEL_BACKBONE"
  echo "  - Output Path: $BASE_OUTPUT_PATH"
  mkdir -p "$BASE_OUTPUT_PATH"

  cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval_legacy.py \
    --model_backbone \"$MODEL_BACKBONE\" \
    --model_name \"$MODEL_NAME\" \
    --dataset_config \"$DATA_CONFIG\" \
    --encode_output_path \"$BASE_OUTPUT_PATH\" \
    --data_basedir \"$DATA_BASEDIR\" \
    --per_device_eval_batch_size $BATCH_SIZE \
    --dataloader_num_workers $NUM_WORKERS"

  echo "  - Executing command..."
  eval "$cmd"
  echo "  - Done."
  echo "-------------------------------------------------"
done

echo "✅ All jobs completed."