#!/usr/bin/env python3
"""
JSONL格式检查器
用于验证训练集和测试集的JSONL文件格式是否符合要求

训练集格式要求:
- qry: 查询文本
- qry_image_path: 查询图片路径
- pos_text: 正样本文本
- pos_image_path: 正样本图片路径
- neg_text: 负样本文本
- neg_image_path: 负样本图片路径

测试集格式要求:
- qry_text: 查询文本
- qry_img_path: 查询图片路径
- tgt_text: 目标文本
- tgt_img_path: 目标图片路径
"""

import json
import argparse
import os
from typing import Dict, List, Any, Tuple


class JSONLFormatChecker:
    def __init__(self):
        # 定义训练集和测试集的必需字段
        self.train_required_fields = {
            'qry', 'qry_image_path', 'pos_text', 'pos_image_path', 'neg_text', 'neg_image_path'
        }
        self.test_required_fields = {
            'qry_text', 'qry_img_path', 'tgt_text', 'tgt_img_path'
        }
    
    def check_train_format(self, sample: Dict[str, Any], line_num: int) -> List[str]:
        """检查训练集样本格式"""
        errors = []
        
        # 检查必需字段
        missing_fields = self.train_required_fields - set(sample.keys())
        if missing_fields:
            errors.append(f"行 {line_num}: 缺少必需字段: {missing_fields}")
        
        # 检查额外字段
        extra_fields = set(sample.keys()) - self.train_required_fields
        if extra_fields:
            errors.append(f"行 {line_num}: 包含额外字段: {extra_fields}")
        
        # 检查字段类型
        for field in self.train_required_fields:
            if field in sample:
                if not isinstance(sample[field], str):
                    errors.append(f"行 {line_num}: 字段 '{field}' 应该是字符串类型，实际类型: {type(sample[field])}")
        
        # 检查特定字段内容
        if 'qry' in sample:
            if not sample['qry'].startswith('<|image_1|>'):
                errors.append(f"行 {line_num}: 'qry' 字段应该以 '<|image_1|>' 开头")
        
        if 'qry_image_path' in sample:
            if not sample['qry_image_path'].endswith(('.png', '.jpg', '.jpeg')):
                errors.append(f"行 {line_num}: 'qry_image_path' 应该是图片文件路径")
        
        return errors
    
    def check_test_format(self, sample: Dict[str, Any], line_num: int) -> List[str]:
        """检查测试集样本格式"""
        errors = []
        
        # 检查必需字段
        missing_fields = self.test_required_fields - set(sample.keys())
        if missing_fields:
            errors.append(f"行 {line_num}: 缺少必需字段: {missing_fields}")
        
        # 检查额外字段
        extra_fields = set(sample.keys()) - self.test_required_fields
        if extra_fields:
            errors.append(f"行 {line_num}: 包含额外字段: {extra_fields}")
        
        # 检查字段类型
        for field in self.test_required_fields:
            if field in sample:
                if field in ['tgt_text', 'tgt_img_path']:
                    # tgt_text 和 tgt_img_path 应该是字符串列表
                    if not isinstance(sample[field], list):
                        errors.append(f"行 {line_num}: 字段 '{field}' 应该是列表类型，实际类型: {type(sample[field])}")
                    elif not all(isinstance(item, str) for item in sample[field]):
                        errors.append(f"行 {line_num}: 字段 '{field}' 应该是字符串列表，包含非字符串元素")
                else:
                    # qry_text 和 qry_img_path 应该是字符串
                    if not isinstance(sample[field], str):
                        errors.append(f"行 {line_num}: 字段 '{field}' 应该是字符串类型，实际类型: {type(sample[field])}")
        
        # 检查列表长度一致性（tgt_text 与 tgt_img_path）
        if 'tgt_text' in sample and 'tgt_img_path' in sample:
            if isinstance(sample['tgt_text'], list) and isinstance(sample['tgt_img_path'], list):
                if len(sample['tgt_text']) != len(sample['tgt_img_path']):
                    errors.append(
                        f"行 {line_num}: 'tgt_text' 与 'tgt_img_path' 长度不一致: "
                        f"{len(sample['tgt_text'])} != {len(sample['tgt_img_path'])}"
                    )
        
        # 检查特定字段内容
        if 'qry_text' in sample and isinstance(sample['qry_text'], str):
            if not sample['qry_text'].startswith('<|image_1|>'):
                errors.append(f"行 {line_num}: 'qry_text' 字段应该以 '<|image_1|>' 开头")
        
        if 'qry_img_path' in sample and isinstance(sample['qry_img_path'], str):
            if not sample['qry_img_path'].endswith(('.png', '.jpg', '.jpeg')):
                errors.append(f"行 {line_num}: 'qry_img_path' 应该是图片文件路径")
        
        if 'tgt_img_path' in sample and isinstance(sample['tgt_img_path'], list):
            for i, path in enumerate(sample['tgt_img_path']):
                if isinstance(path, str):
                    # 允许空字符串占位，不做后缀校验
                    if path == "":
                        continue
                    if not path.endswith(('.png', '.jpg', '.jpeg')):
                        errors.append(f"行 {line_num}: 'tgt_img_path[{i}]' 应该是图片文件路径或空字符串占位")
        
        return errors
    
    def check_jsonl_file(self, file_path: str, split: str, img_root: str = None, check_images_count: int = 0, allow_missing_images: bool = False) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        检查JSONL文件格式，并可选抽样检查图片文件是否存在
        
        Args:
            file_path: JSONL文件路径
            split: 数据集类型 ('train' 或 'test')
            img_root: 图片根目录（当 JSONL 中为相对路径时，会拼接到该根目录）
            check_images_count: 抽样检查图片存在性的数量（0 表示不检查）
            allow_missing_images: 允许图片缺失不计为错误（仅记录告警）
        
        Returns:
            (is_valid, errors, stats)
        """
        if not os.path.exists(file_path):
            return False, [f"文件不存在: {file_path}"], {}
        
        if split not in ['train', 'test']:
            return False, [f"不支持的split类型: {split}，只支持 'train' 或 'test'"], {}
        
        errors = []
        stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_lines': 0,
            'empty_lines': 0,
            'json_parse_errors': 0,
            # 图片存在性抽样检查统计
            'image_paths_checked': 0,
            'image_paths_missing_count': 0,
            'image_paths_missing_samples': [],  # [(line_num, field, path)]
        }
        
        def make_abs_path(p: str) -> str:
            # 绝对路径直接返回；相对路径拼接 img_root（若提供）
            if os.path.isabs(p):
                return p
            clean = p.lstrip('./\\')
            if img_root:
                # 如果路径已经以 img_root 开头，则直接使用；否则拼接
                if clean.startswith(img_root):
                    return clean
                return os.path.join(img_root, clean)
            return clean
        
        checked = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stats['total_lines'] += 1
                    
                    # 跳过空行
                    if not line.strip():
                        stats['empty_lines'] += 1
                        continue
                    
                    # 尝试解析JSON
                    try:
                        sample = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        stats['json_parse_errors'] += 1
                        errors.append(f"行 {line_num}: JSON解析错误: {e}")
                        continue
                    
                    # 检查格式
                    if split == 'train':
                        sample_errors = self.check_train_format(sample, line_num)
                    else:  # split == 'test'
                        sample_errors = self.check_test_format(sample, line_num)
                    
                    if sample_errors:
                        stats['invalid_lines'] += 1
                        errors.extend(sample_errors)
                    else:
                        stats['valid_lines'] += 1
                    
                    # 图片存在性抽样检查（最多 check_images_count 个）
                    if check_images_count and checked < check_images_count:
                        if split == 'train':
                            for field in ['qry_image_path', 'pos_image_path', 'neg_image_path']:
                                if checked >= check_images_count:
                                    break
                                if field in sample and isinstance(sample[field], str):
                                    rel = sample[field]
                                    abs_p = make_abs_path(rel)
                                    exists = os.path.exists(abs_p)
                                    stats['image_paths_checked'] += 1
                                    checked += 1
                                    if not exists:
                                        stats['image_paths_missing_count'] += 1
                                        stats['image_paths_missing_samples'].append((line_num, field, abs_p))
                                        if not allow_missing_images:
                                            errors.append(f"行 {line_num}: 图片不存在: {field} -> {abs_p}")
                        else:  # test
                            # 先检查 qry_img_path
                            if checked < check_images_count and 'qry_img_path' in sample and isinstance(sample['qry_img_path'], str):
                                rel = sample['qry_img_path']
                                abs_p = make_abs_path(rel)
                                exists = os.path.exists(abs_p)
                                stats['image_paths_checked'] += 1
                                checked += 1
                                if not exists:
                                    stats['image_paths_missing_count'] += 1
                                    stats['image_paths_missing_samples'].append((line_num, 'qry_img_path', abs_p))
                                    if not allow_missing_images:
                                        errors.append(f"行 {line_num}: 图片不存在: qry_img_path -> {abs_p}")
                            # 再检查 tgt_img_path（非空项）
                            if checked < check_images_count and 'tgt_img_path' in sample and isinstance(sample['tgt_img_path'], list):
                                for path in sample['tgt_img_path']:
                                    if checked >= check_images_count:
                                        break
                                    if not isinstance(path, str) or path == "":
                                        continue  # 跳过占位或非字符串
                                    abs_p = make_abs_path(path)
                                    exists = os.path.exists(abs_p)
                                    stats['image_paths_checked'] += 1
                                    checked += 1
                                    if not exists:
                                        stats['image_paths_missing_count'] += 1
                                        stats['image_paths_missing_samples'].append((line_num, 'tgt_img_path', abs_p))
                                        if not allow_missing_images:
                                            errors.append(f"行 {line_num}: 图片不存在: tgt_img_path -> {abs_p}")
        
        except Exception as e:
            return False, [f"读取文件时发生错误: {e}"], stats
        
        is_valid = len(errors) == 0
        return is_valid, errors, stats
    
    def print_report(self, file_path: str, split: str, is_valid: bool, errors: List[str], stats: Dict[str, Any]):
        """打印检查报告"""
        print(f"\n{'='*60}")
        print(f"JSONL格式检查报告")
        print(f"{'='*60}")
        print(f"文件路径: {file_path}")
        print(f"数据集类型: {split}")
        print(f"检查结果: {'✅ 通过' if is_valid else '❌ 失败'}")
        print(f"\n统计信息:")
        print(f"  总行数: {stats.get('total_lines', 0)}")
        print(f"  有效行数: {stats.get('valid_lines', 0)}")
        print(f"  无效行数: {stats.get('invalid_lines', 0)}")
        print(f"  空行数: {stats.get('empty_lines', 0)}")
        print(f"  JSON解析错误: {stats.get('json_parse_errors', 0)}")
        
        # 图片存在性抽样检查汇报
        if stats.get('image_paths_checked', 0) > 0:
            print(f"\n图片存在性抽样检查:")
            print(f"  抽样检查数量: {stats.get('image_paths_checked', 0)}")
            print(f"  缺失图片数量: {stats.get('image_paths_missing_count', 0)}")
            missing = stats.get('image_paths_missing_samples', [])
            if missing:
                print(f"  缺失样例(最多显示10条):")
                for i, (line_num, field, path) in enumerate(missing[:10], 1):
                    print(f"    {i}. 行 {line_num} - {field}: {path}")
        
        if errors:
            print(f"\n发现的错误 ({len(errors)} 个):")
            for i, error in enumerate(errors[:20], 1):  # 只显示前20个错误
                print(f"  {i}. {error}")
            
            if len(errors) > 20:
                print(f"  ... 还有 {len(errors) - 20} 个错误未显示")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='JSONL格式检查器')
    parser.add_argument('jsonl_path', help='JSONL文件路径')
    parser.add_argument('split', choices=['train', 'test'], help='数据集类型 (train 或 test)')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，只输出结果')
    # 新增：图片存在性抽样检查参数
    parser.add_argument('--check-images', type=int, default=10, help='抽样检查图片存在性的数量 (0 禁用，默认 10)')
    parser.add_argument('--img-root', type=str, default='MMCoIR', help='图片根目录，用于拼接相对路径 (默认 MMCoIR)')
    parser.add_argument('--allow-missing-images', action='store_true', help='允许图片缺失不计为错误，仅记录告警')
    
    args = parser.parse_args()
    
    checker = JSONLFormatChecker()
    is_valid, errors, stats = checker.check_jsonl_file(
        args.jsonl_path,
        args.split,
        img_root=args.img_root,
        check_images_count=args.check_images,
        allow_missing_images=args.allow_missing_images,
    )
    
    if not args.quiet:
        checker.print_report(args.jsonl_path, args.split, is_valid, errors, stats)
    else:
        if is_valid:
            print("✅ 格式检查通过")
        else:
            print(f"❌ 格式检查失败，发现 {len(errors)} 个错误")
    
    # 返回适当的退出码
    exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()