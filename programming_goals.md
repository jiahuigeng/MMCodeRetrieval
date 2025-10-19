# 多模态检索数据集创建目标

## 项目概述
创建多模态检索的数据集，将已有的数据集转化为检索需要的数据类型，存储为 JSONL 格式。

## 数据格式规范

### 训练集格式
每个 item 包含以下字段：
- `qry`: 查询文本
- `qry_image_path`: 查询图片路径
- `pos_text`: 正样本文本
- `pos_image_path`: 正样本图片路径
- `neg_text`: 负样本文本
- `neg_image_path`: 负样本图片路径

**训练集示例：**
```json
{
  "qry": "查询文本内容",
  "qry_image_path": "path/to/query/image.jpg",
  "pos_text": "正样本文本内容",
  "pos_image_path": "path/to/positive/image.jpg",
  "neg_text": "负样本文本内容",
  "neg_image_path": "path/to/negative/image.jpg"
}
```

### 测试集格式
每个 item 包含以下字段：
- `qry_text`: 查询文本
- `qry_img_path`: 查询图片路径
- `tgt_text`: 目标文本
- `tgt_img_path`: 目标图片路径

**测试集示例：**
```json
{
  "qry_text": "查询文本内容",
  "qry_img_path": "path/to/query/image.jpg",
  "tgt_text": "目标文本内容",
  "tgt_img_path": "path/to/target/image.jpg"
}
```

## 当前可用数据集
- `rootsautomation/RICO-Screen2Words` - UI截图和描述数据集
- `xxxllz/Chart2Code-160k` - 图表到代码数据集
- `whale99/Interaction2Code` - 交互到代码数据集

## 实施步骤
1. 分析现有数据集结构
2. 设计数据转换逻辑
3. 实现数据预处理脚本
4. 生成训练集和测试集的 JSONL 文件
5. 验证数据质量和格式正确性

## 输出文件
- `train_multimodal_retrieval.jsonl` - 训练集
- `test_multimodal_retrieval.jsonl` - 测试集

---
*创建时间: $(date)*
*项目路径: /home/a/Projects/MMCodeRetrieval*