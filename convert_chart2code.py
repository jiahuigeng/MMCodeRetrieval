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
        "qry_image_path": f"chart2code/images/{image_filename}",
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
        "qry_img_path": f"chart2code/images/{image_filename}",
        "tgt_text": [gpt_msg],  # 转换为字符串列表
        "tgt_img_path": [""]  # 使用空字符串占位的列表
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
    # 命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="将 chart2code.json 转为训练/测试 JSONL")
    parser.add_argument("--input-json", nargs="+", default=None, help="一个或多个 JSON 文件路径（可选）")
    parser.add_argument("--input-dir", default=os.path.join("datasets", "chart2code_160k", "json"), help="输入目录（处理其中所有 .json 文件）")
    parser.add_argument("--output-dir", default=os.path.join("MMCoIR", "chart2code"), help="输出目录，默认 MMCoIR/chart2code")
    parser.add_argument("--limit", type=int, default=None, help="每个文件最多处理的样本数（可选）")
    parser.add_argument("--split-mode", choices=["all_train", "all_test", "train_count", "train_ratio"], default="train_ratio", help="数据划分模式：全部训练/全部测试/按数量/按比例")
    parser.add_argument("--train-count", type=int, default=None, help="当 split-mode=train_count 时，训练集数量")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="当 split-mode=train_ratio 时，训练集比例(0-1)，默认 0.8")
    args = parser.parse_args()

    # 收集输入文件列表
    if args.input_json:
        files = args.input_json
    else:
        files = [os.path.join(args.input_dir, fn) for fn in os.listdir(args.input_dir) if fn.lower().endswith('.json')]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"将处理 {len(files)} 个文件: {', '.join([os.path.basename(p) for p in files])}")

    # 累计转换结果
    train_converted = []
    test_converted = []

    for file_path in files:
        # 加载数据
        data = load_chart2code_data(file_path)
        
        # 可选截断（每文件）
        if args.limit is not None:
            data = data[:args.limit]
        
        # 按ID排序确保一致性
        data.sort(key=lambda x: x['id'])
        
        # 划分训练集和测试集（默认按比例 8/2）
        if args.split_mode == "all_train":
            train_data = data
            test_data = []
        elif args.split_mode == "all_test":
            train_data = []
            test_data = data
        elif args.split_mode == "train_count":
            cnt = args.train_count if args.train_count is not None else len(data)
            cnt = max(0, min(len(data), cnt))
            train_data = data[:cnt]
            test_data = data[cnt:]
        else:  # train_ratio
            ratio = args.train_ratio
            ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
            cnt = int(round(len(data) * ratio))
            train_data = data[:cnt]
            test_data = data[cnt:]
        
        print(f"[{os.path.basename(file_path)}] 训练集样本数: {len(train_data)} | 测试集样本数: {len(test_data)}")
        
        # 转换训练集
        for i, sample in enumerate(train_data):
            if i % 50000 == 0 and i > 0:
                print(f"[{os.path.basename(file_path)}] 训练集转换进度: {i}/{len(train_data)}")
            train_converted.append(convert_sample_to_train_format(sample))
        
        # 转换测试集
        for i, sample in enumerate(test_data):
            if i % 20000 == 0 and i > 0:
                print(f"[{os.path.basename(file_path)}] 测试集转换进度: {i}/{len(test_data)}")
            test_converted.append(convert_sample_to_test_format(sample))
    
    # 保存结果（合并后的）
    train_output_path = os.path.join(output_dir, "train.jsonl")
    test_output_path = os.path.join(output_dir, "test.jsonl")
    
    save_jsonl(train_converted, train_output_path)
    save_jsonl(test_converted, test_output_path)
    
    print(f"\n转换完成！")
    print(f"训练集: {train_output_path} ({len(train_converted)} 样本)")
    print(f"测试集: {test_output_path} ({len(test_converted)} 样本)")
    
    # 显示样本示例
    if train_converted:
        print("\n=== 训练集样本示例 ===")
        print(json.dumps(train_converted[0], ensure_ascii=False, indent=2))
    if test_converted:
        print("\n=== 测试集样本示例 ===")
        print(json.dumps(test_converted[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()