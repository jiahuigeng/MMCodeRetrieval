#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 MBZUAI/Web2Code 数据集：

- 确认 train 从 Web2Code.json 采样，test 从 Web2Code_eval.jsonl 采样
- 解析两文件的记录，统计 metadata（记录数、有效 QA 数、角色集合、对话长度分布、图片存在性抽样等）
- 按需进行图片存在性抽样检查（在本地 snapshot 下拼接相对路径）

运行示例：
  python inspect_web2code_metadata.py \
    --input-dir datasets/Web2Code \
    --train-file Web2Code.json \
    --test-file Web2Code_eval.jsonl \
    --check-images 2000 --analyze-limit 200000
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set


def parse_args():
    p = argparse.ArgumentParser(description="检查 Web2Code.json / Web2Code_eval.jsonl 的 metadata")
    p.add_argument("--input-dir", type=str, default=str(Path(__file__).parent / "datasets" / "Web2Code"), help="数据快照根目录")
    p.add_argument("--train-file", type=str, default="Web2Code.json", help="训练源文件（默认 Web2Code.json）")
    p.add_argument("--test-file", type=str, default="Web2Code_eval.jsonl", help="测试源文件（默认 Web2Code_eval.jsonl）")
    p.add_argument("--check-images", type=int, default=1000, help="图片存在性抽样检查的数量（0 禁用）")
    p.add_argument("--analyze-limit", type=int, default=200000, help="对话长度统计最多分析条数（防止过慢）")
    return p.parse_args()


def read_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except Exception:
        return []
    return []


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            out.append(obj)
    return out


def extract_qa(conv: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    q = None
    a = None
    for msg in conv:
        role = (msg or {}).get("from") or (msg or {}).get("role")
        val = (msg or {}).get("value") or (msg or {}).get("text")
        if role in ("human", "user") and isinstance(val, str):
            q = val
            break
    if q is not None:
        for msg in conv:
            role = (msg or {}).get("from") or (msg or {}).get("role")
            val = (msg or {}).get("value") or (msg or {}).get("text")
            if role in ("gpt", "assistant") and isinstance(val, str):
                a = val
                break
    if q and a:
        return q, a
    return None


def analyze_split(records: List[Dict[str, Any]], root: Path, check_images: int, analyze_limit: int) -> Dict[str, Any]:
    total = len(records)
    roles: Set[str] = set()
    with_image = 0
    valid_qa = 0
    conv_lens: List[int] = []

    # 基本统计
    for i, obj in enumerate(records):
        img = obj.get("image")
        conv = obj.get("conversations")
        if isinstance(img, str):
            with_image += 1
        if isinstance(conv, list):
            # 角色集合
            for msg in conv:
                role = (msg or {}).get("from") or (msg or {}).get("role")
                if isinstance(role, str):
                    roles.add(role)
            # 对话长度统计（限量）
            if len(conv_lens) < analyze_limit:
                conv_lens.append(len(conv))
            # QA 可用性
            if extract_qa(conv) is not None:
                valid_qa += 1

    # 图片存在性抽样检查
    img_checked = 0
    img_exists = 0
    def resolve_image_path(root_dir: Path, img_rel: str) -> Path:
        p1 = root_dir / img_rel
        if p1.exists():
            return p1
        p2 = root_dir / "Web2Code_image" / img_rel
        return p2

    if check_images:
        for obj in records:
            img = obj.get("image")
            if not isinstance(img, str):
                continue
            full = resolve_image_path(root, img)
            img_checked += 1
            if full.exists():
                img_exists += 1
            if img_checked >= check_images:
                break

    # 对话长度指标
    if conv_lens:
        conv_min = min(conv_lens)
        conv_max = max(conv_lens)
        conv_mean = sum(conv_lens) / len(conv_lens)
    else:
        conv_min = conv_max = conv_mean = 0.0

    return {
        "total_records": total,
        "with_image_field": with_image,
        "valid_qa_records": valid_qa,
        "roles": sorted(list(roles)),
        "conv_len_min": conv_min,
        "conv_len_max": conv_max,
        "conv_len_mean": conv_mean,
        "image_exists_checked": img_checked,
        "image_exists_count": img_exists,
    }


def main():
    args = parse_args()
    root = Path(args.input_dir)
    train_path = root / args.train_file
    test_path = root / args.test_file

    print(f"Input dir: {root} {'(exists)' if root.exists() else '(missing)'}")
    print(f"Train file: {train_path} {'(exists)' if train_path.exists() else '(missing)'}")
    print(f"Test  file: {test_path} {'(exists)' if test_path.exists() else '(missing)'}")

    # 加载数据
    train_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    if train_path.suffix.lower() == ".jsonl":
        train_records = read_jsonl(train_path)
    else:
        train_records = read_json(train_path)
    if test_path.suffix.lower() == ".jsonl":
        test_records = read_jsonl(test_path)
    else:
        test_records = read_json(test_path)

    print(f"\n[Load] Train records: {len(train_records)} | Test records: {len(test_records)}")

    # 统计
    train_meta = analyze_split(train_records, root, args.check_images, args.analyze_limit)
    test_meta = analyze_split(test_records, root, args.check_images, args.analyze_limit)

    print("\n==== Metadata (Train: Web2Code.json) ====")
    for k, v in train_meta.items():
        print(f"{k}: {v}")

    print("\n==== Metadata (Test: Web2Code_eval.jsonl) ====")
    for k, v in test_meta.items():
        print(f"{k}: {v}")

    # 基于有效 QA + 图片存在性（抽样）给出采样可行性的提示（仅统计，不执行采样）
    print("\n==== Sampling Plan (Preview) ====")
    print("Train source: Web2Code.json -> plan to sample up to 100000")
    print("Test  source: Web2Code_eval.jsonl -> plan to sample up to 2000")
    print("Note: Actual sampling will filter by valid QA and image availability when converting.")


if __name__ == "__main__":
    main()