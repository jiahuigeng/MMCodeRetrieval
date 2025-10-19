#!/usr/bin/env python3
"""
Chart2Code数据集转换脚本
将chart2code.json转换为训练集和测试集的JSONL格式，用于多模态代码检索任务
"""

import json
import os
from pathlib import Path

def load_chart2code_data(json_path):
    """加载chart2code.json数据"""
    print(f"正在加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载完成，共 {len(data)} 个样本")
    return data

def convert_sample_to_train_format(sample):
    """将样本转换为训练集格式"""
    # 获取human和gpt的对话内容
    conversations = sample['conversations']
    human_msg = conversations[0]['value']  # human输入
    gpt_msg = conversations[1]['value']    # gpt输出（代码）
    
    # 移除human消息中的<image>标记，添加<|image_1|>
    query_text = human_msg.replace('<image>', '').strip()
    query = f"<|image_1|>\n{query_text}"
    
    # 构建训练集格式
    # 从image字段中提取文件名（去掉images/前缀）
    image_filename = sample['image'].replace('images/', '')
    
    train_sample = {
        "qry": query,
        "qry_image_path": f"MMCoIR/chart2code/images/{image_filename}",
        "pos_text": gpt_msg,
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": ""
    }
    
    return train_sample

def convert_sample_to_test_format(sample):
    """将样本转换为测试集格式"""
    # 获取human和gpt的对话内容
    conversations = sample['conversations']
    human_msg = conversations[0]['value']  # human输入
    gpt_msg = conversations[1]['value']    # gpt输出（代码）
    
    # 移除human消息中的<image>标记，添加<|image_1|>
    query_text = human_msg.replace('<image>', '').strip()
    query = f"<|image_1|>\n{query_text}"
    
    # 构建测试集格式
    # 从image字段中提取文件名（去掉images/前缀）
    image_filename = sample['image'].replace('images/', '')
    
    test_sample = {
        "qry_text": query,
        "qry_img_path": f"MMCoIR/chart2code/images/{image_filename}",
        "tgt_text": [gpt_msg],  # 转换为字符串列表
        "tgt_img_path": [f"MMCoIR/chart2code/images/{image_filename}"]  # 转换为字符串列表
    }
    
    return test_sample

def save_jsonl(data, output_path):
    """保存数据为JSONL格式"""
    print(f"正在保存 {len(data)} 个样本到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"保存完成: {output_path}")

def main():
    # 路径配置
    input_json_path = "/home/a/Projects/MMCodeRetrieval/datasets/chart2code_160k/json/chart2code.json"
    output_dir = "/home/a/Projects/MMCodeRetrieval/MMCoIR/chart2code"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data = load_chart2code_data(input_json_path)
    
    # 按ID排序确保一致性
    data.sort(key=lambda x: x['id'])
    
    # 划分训练集和测试集
    # 前150,000个样本作为训练集，剩余作为测试集
    train_data = data[:150000]
    test_data = data[150000:]
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    # 转换训练集
    print("\n正在转换训练集...")
    train_converted = []
    for i, sample in enumerate(train_data):
        if i % 10000 == 0:
            print(f"训练集转换进度: {i}/{len(train_data)}")
        train_converted.append(convert_sample_to_train_format(sample))
    
    # 转换测试集
    print("\n正在转换测试集...")
    test_converted = []
    for i, sample in enumerate(test_data):
        if i % 1000 == 0:
            print(f"测试集转换进度: {i}/{len(test_data)}")
        test_converted.append(convert_sample_to_test_format(sample))
    
    # 保存结果
    train_output_path = os.path.join(output_dir, "train.jsonl")
    test_output_path = os.path.join(output_dir, "test.jsonl")
    
    save_jsonl(train_converted, train_output_path)
    save_jsonl(test_converted, test_output_path)
    
    print(f"\n转换完成！")
    print(f"训练集: {train_output_path} ({len(train_converted)} 样本)")
    print(f"测试集: {test_output_path} ({len(test_converted)} 样本)")
    
    # 显示样本示例
    print("\n=== 训练集样本示例 ===")
    print(json.dumps(train_converted[0], ensure_ascii=False, indent=2))
    
    print("\n=== 测试集样本示例 ===")
    print(json.dumps(test_converted[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()