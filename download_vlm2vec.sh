#!/bin/bash
# VLM2Vec 模型下载脚本
# 使用方法: bash download_vlm2vec.sh

set -e

echo "=== VLM2Vec 模型下载脚本 ==="
echo

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python"
    exit 1
fi

# 检查是否在正确的conda环境中
if [[ "$CONDA_DEFAULT_ENV" != "embodied" ]]; then
    echo "警告: 当前不在embodied环境中，建议先激活环境:"
    echo "conda activate embodied"
    echo
fi

# 设置变量
MODEL_NAME="VLM2Vec/VLM2Vec-V2.0"
CACHE_DIR="./models"
LOG_FILE="download.log"

echo "模型名称: $MODEL_NAME"
echo "缓存目录: $CACHE_DIR"
echo "日志文件: $LOG_FILE"
echo

# 创建缓存目录
mkdir -p "$CACHE_DIR"

# 检查网络连接
echo "检查网络连接..."
if ping -c 1 huggingface.co &> /dev/null; then
    echo "✓ 可以访问 huggingface.co"
    USE_MIRROR=""
else
    echo "✗ 无法访问 huggingface.co，将使用镜像源"
    USE_MIRROR="--use_mirror"
fi
echo

# 检查模型是否已存在
echo "检查模型是否已存在..."
if python download_hf_model.py --model_name "$MODEL_NAME" --cache_dir "$CACHE_DIR" --check_only; then
    echo "模型已存在，跳过下载"
    echo "模型路径: $CACHE_DIR"
    exit 0
fi
echo

# 下载模型
echo "开始下载模型..."
echo "这可能需要几分钟到几小时，取决于网络速度"
echo "下载日志将保存到: $LOG_FILE"
echo

# 构建下载命令
DOWNLOAD_CMD="python download_hf_model.py --model_name '$MODEL_NAME' --cache_dir '$CACHE_DIR' $USE_MIRROR"

echo "执行命令: $DOWNLOAD_CMD"
echo

# 执行下载
if eval "$DOWNLOAD_CMD" 2>&1 | tee "$LOG_FILE"; then
    echo
    echo "✓ 模型下载成功！"
    echo "模型路径: $CACHE_DIR"
    echo
    
    # 显示模型信息
    echo "模型文件列表:"
    find "$CACHE_DIR" -name "*$MODEL_NAME*" -type d | head -5
    echo
    
    # 设置环境变量提示
    echo "使用模型时，请设置以下环境变量:"
    echo "export HF_HOME=$(realpath $CACHE_DIR)"
    echo "export TRANSFORMERS_CACHE=$(realpath $CACHE_DIR)"
    echo
    
else
    echo
    echo "✗ 模型下载失败"
    echo "请检查日志文件: $LOG_FILE"
    echo
    
    # 提供故障排除建议
    echo "故障排除建议:"
    echo "1. 检查网络连接"
    echo "2. 尝试使用代理: python download_hf_model.py --proxy http://127.0.0.1:7890 ..."
    echo "3. 尝试使用镜像源: python download_hf_model.py --use_mirror ..."
    echo "4. 检查磁盘空间是否足够 (需要约8-10GB)"
    echo
    
    exit 1
fi

echo "下载完成！现在可以运行VLM2Vec评估了。"