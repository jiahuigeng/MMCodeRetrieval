#!/bin/bash
# NOTE: 将下面的占位路径替换为你本地实际路径
# export LD_LIBRARY_PATH=...
# export PATH=...
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# export HF_DATASETS_CACHE=...
# export HF_HOME=...
# export WANDB_DISABLED=false
# export WANDB_PROJECT=...
# export WANDB_API_KEY=...
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_RUN_GROUP=...
############################################################
# 可配置参数（用于自动命名 EXP_NAME）
############################################################
MODEL_TAG=Qwen2vl_2B
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct

CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2
MASTER_PORT=2207

LORA=true
LORA_R=8
BF16=true
POOLING=eos
NORMALIZE=True
TEMPERATURE=0.02
DATALOADER_NUM_WORKERS=8
# 使用 sample 配置，每个子数据集仅 20 条样本
DATASET_CONFIG=experiments_mmcoir/public/train/image_sample.yaml
DATASET_TAG=$(basename "$DATASET_CONFIG" .yaml)

PER_DEVICE_TRAIN_BATCH_SIZE=16
GC_Q_CHUNK_SIZE=4
GC_P_CHUNK_SIZE=48
INTERLEAVE_BATCH_SIZE=16
LR_SCHEDULER_TYPE=linear
LEARNING_RATE=5e-5
MAX_STEPS=5000
WARMUP_STEPS=100
SAVE_STEPS=50
LOGGING_STEPS=1
SAVE_SAFETENSORS=True
REMOVE_UNUSED_COLUMNS=False
RESUME_FROM=auto
REPORT_TO=wandb

# 任务标签，用于区分实验名；这里标记为 imageonly-sample
TASK_TAG=imageonly-sample

# 基于上述参数自动构建实验名
TEMP_FMT=$(printf "%03d" $(echo "${TEMPERATURE} * 100" | bc | awk '{printf("%d", $1)}'))
EXP_NAME="${MODEL_TAG}.${TASK_TAG}.${DATASET_TAG}.lora${LORA_R}.bf16${BF16}.BS$((PER_DEVICE_TRAIN_BATCH_SIZE*NPROC_PER_NODE)).IB${INTERLEAVE_BATCH_SIZE}.GCq${GC_Q_CHUNK_SIZE}p${GC_P_CHUNK_SIZE}.Norm${NORMALIZE}.Temp${TEMP_FMT}.lr${LEARNING_RATE}.${LR_SCHEDULER_TYPE}.step${MAX_STEPS}warm${WARMUP_STEPS}"

export WANDB_NAME="$EXP_NAME"
# 修改为你的实验目录根路径
EXP_DIR_BASE=...
export EXP_DIR="$EXP_DIR_BASE/$EXP_NAME"
export WANDB_DIR="$EXP_DIR"
echo "$EXP_DIR"

mkdir -p "$EXP_DIR/wandb"
rm -rf "$EXP_DIR/wandb/*"


# 修改为你的仓库根目录（包含 train.py）
REPO_DIR=PATH_TO_VLM2VEC_REPO
cd "$REPO_DIR"
cmd="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} --max_restarts=0 train.py $( [ \"${LORA}\" = \"true\" ] && echo --lora ) --lora_r ${LORA_R} --model_name ${MODEL_NAME} $( [ \"${BF16}\" = \"true\" ] && echo --bf16 ) --pooling ${POOLING} --normalize ${NORMALIZE} --temperature ${TEMPERATURE} --dataloader_num_workers ${DATALOADER_NUM_WORKERS} --dataset_config ${DATASET_CONFIG} --run_name ${EXP_NAME} --output_dir ${EXP_DIR} --grad_cache True --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} --gc_q_chunk_size ${GC_Q_CHUNK_SIZE} --gc_p_chunk_size ${GC_P_CHUNK_SIZE} --interleave_batch_size ${INTERLEAVE_BATCH_SIZE} --lr_scheduler_type ${LR_SCHEDULER_TYPE} --learning_rate ${LEARNING_RATE} --max_steps ${MAX_STEPS} --warmup_steps ${WARMUP_STEPS} --save_steps ${SAVE_STEPS} --logging_steps ${LOGGING_STEPS} --save_safetensors ${SAVE_SAFETENSORS} --remove_unused_columns ${REMOVE_UNUSED_COLUMNS} --resume_from ${RESUME_FROM} --report_to ${REPORT_TO} 2>&1 | tee ${EXP_DIR}/train.log"

echo "$cmd"
eval "$cmd"