# VLM2Vec-Qwen2VL-2B 评估功能使用指南

## 概述

VLM2Vec-Qwen2VL-2B 是一个强大的多模态嵌入模型，支持图像、视频和视觉文档三种模态的评估。本指南将帮助您快速上手并运行评估功能。

## 模型特点

- **统一框架**: 支持图像、视频、视觉文档三种模态
- **高性能**: 在MMEB-V2基准测试中表现优异
- **易用性**: 简单的配置和运行方式
- **灵活性**: 支持自定义数据集和评估指标

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
# 安装基础依赖
pip install torch torchvision torchaudio
pip install transformers datasets
pip install numpy pandas tqdm pyyaml
```

### 2. 使用评估脚本

我们提供了一个简化的评估脚本 `run_vlm2vec_qwen2vl_2b_eval.py`，支持多种使用方式：

#### 基础用法
```bash
# 评估图像模态（默认）
python run_vlm2vec_qwen2vl_2b_eval.py

# 评估特定模态
python run_vlm2vec_qwen2vl_2b_eval.py --modality image
python run_vlm2vec_qwen2vl_2b_eval.py --modality video
python run_vlm2vec_qwen2vl_2b_eval.py --modality visdoc

# 评估所有模态
python run_vlm2vec_qwen2vl_2b_eval.py --modality all
```

#### 高级配置
```bash
# 自定义批处理大小和GPU
python run_vlm2vec_qwen2vl_2b_eval.py \
    --modality image \
    --batch_size 16 \
    --gpu_ids "0,1"

# 自定义数据和输出路径
python run_vlm2vec_qwen2vl_2b_eval.py \
    --data_basedir "/path/to/your/data" \
    --output_basedir "/path/to/output"

# 使用不同的模型
python run_vlm2vec_qwen2vl_2b_eval.py \
    --model_name "VLM2Vec/VLM2Vec-V2.0" \
    --model_backbone "qwen2_vl"
```

### 3. 使用测试配置

对于快速测试，可以使用提供的测试配置文件：
```bash
python eval.py \
    --model_name "VLM2Vec/VLM2Vec-V2.0" \
    --model_backbone "qwen2_vl" \
    --dataset_config "test_eval_config.yaml" \
    --encode_output_path "./test_results/" \
    --per_device_eval_batch_size 8 \
    --pooling eos \
    --normalize true
```

## 评估模态说明

### 1. 图像模态 (image)
支持的任务类型：
- **图像分类**: ImageNet-1K, N24News, HatefulMemes等
- **视觉问答**: VQA, GQA, OK-VQA等
- **图像-文本检索**: MSCOCO, Flickr30K等
- **文档理解**: DocVQA, InfographicsVQA等

### 2. 视频模态 (video)
支持的任务类型：
- **视频分类**: Kinetics-700, UCF101, HMDB51等
- **视频问答**: NExTQA, MVBench, Video-MME等
- **视频-文本检索**: MSR-VTT, MSVD, DiDeMo等
- **时序定位**: QVHighlight, Charades-STA等

### 3. 视觉文档模态 (visdoc)
支持的任务类型：
- **文档问答**: ViDoRe系列任务
- **图表理解**: VisRAG-ChartQA, VisRAG-PlotQA等
- **学术文档**: VisRAG-ArxivQA等
- **多页文档**: MMLongBench-doc等

## 评估结果

### 输出文件说明
评估完成后，会在输出目录生成以下文件：
- `{dataset_name}_qry`: 查询嵌入文件
- `{dataset_name}_tgt`: 目标嵌入文件
- `{dataset_name}_info.jsonl`: 数据集信息文件
- `{dataset_name}_score.json`: 评估分数文件
- `{dataset_name}_pred.jsonl`: 预测结果文件

### 评估指标
支持的评估指标包括：
- **Hit@K**: 前K命中率
- **NDCG@K**: 归一化折扣累积增益
- **Precision@K**: 精确率
- **Recall@K**: 召回率
- **F1@K**: F1分数
- **MAP@K**: 平均精度均值
- **MRR@K**: 平均倒数排名

## 性能基准

### VLM2Vec-V2.0-Qwen2VL-2B 在主要任务上的表现：

**图像任务**:
- ImageNet-1K: Hit@1=80.8%, NDCG@10=89.96%
- DocVQA: Hit@1=90.1%, NDCG@10=94.42%
- MSCOCO_i2t: Hit@1=71.4%, NDCG@10=84.47%

**视频任务**:
- MVBench: Hit@1=33.7%, NDCG@10=70.12%
- Video-MME: Hit@1=30.7%, NDCG@10=67.46%
- ActivityNetQA: Hit@1=52.3%, NDCG@10=82.40%

**视觉文档任务**:
- ViDoRe_arxivqa: Hit@1=73.4%, NDCG@10=81.88%
- VisRAG_PlotQA: Hit@1=50.5%, NDCG@10=69.16%
- MMLongBench-doc: Hit@1=61.6%, NDCG@10=41.47%

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少批处理大小：`--batch_size 4`
   - 使用梯度检查点或混合精度训练

2. **数据集未找到**
   - 检查数据路径配置
   - 确保已下载相应的评估数据集

3. **模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 验证模型名称是否正确

### 获取帮助

如果遇到问题，可以：
1. 查看详细的错误日志
2. 检查GitHub Issues
3. 参考官方文档
4. 联系项目维护者

## 扩展功能

### 自定义数据集
可以通过创建自定义数据加载器来支持新的数据集：
1. 在 `src/data/eval_dataset/` 目录下创建新的数据集类
2. 注册数据集解析器
3. 更新配置文件

### 自定义评估指标
可以在 `src/eval_utils/metrics.py` 中添加新的评估指标。

## 总结

VLM2Vec-Qwen2VL-2B 提供了强大而灵活的多模态评估功能。通过本指南，您应该能够：
- 快速运行基础评估
- 理解不同模态的评估任务
- 解读评估结果
- 解决常见问题
- 扩展自定义功能

开始您的多模态嵌入评估之旅吧！