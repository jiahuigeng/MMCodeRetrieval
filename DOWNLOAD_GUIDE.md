# VLM2Vec 模型下载指南

本指南提供了多种方法来下载VLM2Vec模型，解决网络连接问题。

## 快速开始

### 方法1: 使用自动下载脚本（推荐）

```bash
# 给脚本执行权限
chmod +x download_vlm2vec.sh

# 运行下载脚本
./download_vlm2vec.sh
```

### 方法2: 使用Python下载器

```bash
# 基础下载
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0

# 使用镜像源（推荐用于国内网络）
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --use_mirror

# 使用代理
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --proxy http://127.0.0.1:7890

# 自定义缓存目录
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --cache_dir ./models
```

## 详细说明

### 网络问题解决方案

#### 1. 无法访问huggingface.co

**问题**: `OSError: We couldn't connect to 'https://huggingface.co'`

**解决方案**:
```bash
# 使用HF-Mirror镜像源
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --use_mirror --mirror_endpoint hf-mirror

# 或使用ModelScope镜像源
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --use_mirror --mirror_endpoint modelscope
```

#### 2. 网络代理设置

```bash
# HTTP代理
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --proxy http://127.0.0.1:7890

# SOCKS代理
export HTTP_PROXY=socks5://127.0.0.1:1080
export HTTPS_PROXY=socks5://127.0.0.1:1080
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0
```

### 高级选项

#### 1. 仅下载特定文件

```bash
# 仅下载配置文件
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --file_only config.json

# 仅下载模型权重
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --file_only pytorch_model.bin
```

#### 2. 过滤文件

```bash
# 忽略大文件
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --ignore_patterns "*.safetensors" "*.bin"

# 仅下载配置文件
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --allow_patterns "*.json" "*.txt"
```

#### 3. 检查模型状态

```bash
# 检查模型是否已下载
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --check_only
```

### 环境配置

#### 1. 设置缓存目录

下载完成后，设置环境变量以便VLM2Vec使用本地模型：

```bash
# 临时设置（当前会话）
export HF_HOME=/path/to/your/models
export TRANSFORMERS_CACHE=/path/to/your/models
export HF_HUB_OFFLINE=1  # 启用离线模式

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_HOME=/path/to/your/models' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/path/to/your/models' >> ~/.bashrc
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
```

#### 2. 验证下载

```python
# 测试模型加载
from transformers import AutoModel, AutoTokenizer

model_name = "VLM2Vec/VLM2Vec-V2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModel.from_pretrained(model_name, local_files_only=True)

print("模型加载成功！")
print(f"模型参数量: {model.num_parameters():,}")
```

## 故障排除

### 常见错误及解决方案

#### 1. 认证错误 (401)
```bash
# 设置HuggingFace token
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --token YOUR_HF_TOKEN
```

#### 2. 模型不存在 (404)
```bash
# 检查模型名称是否正确
# 确认模型是否为私有仓库
```

#### 3. 磁盘空间不足
```bash
# 检查可用空间（VLM2Vec-V2.0约需要8-10GB）
df -h

# 清理旧的缓存
rm -rf ~/.cache/huggingface/hub/models--*
```

#### 4. 网络超时
```bash
# 增加超时时间
export HF_HUB_DOWNLOAD_TIMEOUT=300

# 使用断点续传
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 # 会自动续传
```

### 日志分析

下载过程中的日志保存在 `download_hf_model.log` 文件中：

```bash
# 查看实时日志
tail -f download_hf_model.log

# 查看错误信息
grep -i error download_hf_model.log
```

## 镜像源配置

### 国内用户推荐镜像源

1. **HF-Mirror** (推荐)
   - 地址: https://hf-mirror.com
   - 使用: `--use_mirror --mirror_endpoint hf-mirror`

2. **ModelScope**
   - 地址: https://www.modelscope.cn
   - 使用: `--use_mirror --mirror_endpoint modelscope`

### 手动配置镜像源

```bash
# 方法1: 环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 方法2: 配置文件
mkdir -p ~/.huggingface
echo 'endpoint = "https://hf-mirror.com"' > ~/.huggingface/config.ini
```

## 性能优化

### 并行下载

```bash
# 设置并行下载线程数
export HF_HUB_DOWNLOAD_THREADS=4
```

### 内存优化

```bash
# 对于内存较小的系统，可以分批下载
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --allow_patterns "config.json" "tokenizer*"
python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --allow_patterns "*.safetensors"
```

## 完整示例

```bash
#!/bin/bash
# 完整的VLM2Vec模型下载和配置脚本

# 1. 激活环境
conda activate embodied

# 2. 下载模型
python download_hf_model.py \
    --model_name VLM2Vec/VLM2Vec-V2.0 \
    --cache_dir ./models \
    --use_mirror \
    --mirror_endpoint hf-mirror

# 3. 设置环境变量
export HF_HOME=$(pwd)/models
export TRANSFORMERS_CACHE=$(pwd)/models
export HF_HUB_OFFLINE=1

# 4. 验证安装
python -c "from transformers import AutoModel; print('模型可用:', AutoModel.from_pretrained('VLM2Vec/VLM2Vec-V2.0', local_files_only=True))"

# 5. 运行评估
python run_vlm2vec_qwen2vl_2b_eval.py --modality image --batch_size 4
```

现在您可以成功下载和使用VLM2Vec模型了！