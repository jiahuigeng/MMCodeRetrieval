#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 DiagramAgent/DiagramGenBenchmark 的 DiagramGeneration 子集转换为“文本到代码搜索(t2c)”测试集。

输出：
- 生成到 `MMCoIR-test/DiagramGenBenchmark_t2c/test.jsonl`。
- JSONL 字段（共 4 个 key）：
  - `qry_text`: 来自 `expanded_query` 字段的文本（不包含 image token）。
  - `qry_img_path`: 始终为空字符串 ""。
  - `tgt_text`: 列表，包含来自 `reference` 字段的代码字符串。
  - `tgt_img_path`: 列表，始终为 [""]。

说明：本任务不涉及图片，所有 image_path 字段均为空；不会复制任何图片文件。

示例用法：
  python DiagramGenBenchmark/convert_diagramgenbenchmark_t2c.py \
    --input-json datasets/DiagramGenBenchmark/DiagramGeneration.json \
    --out-dir MMCoIR-test/DiagramGenBenchmark_t2c

可选参数：
- `--limit N`              仅处理前 N 条样本

"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


DATASET_NAME = "DiagramGenBenchmark"               # 原项目名
OUT_DATASET_NAME = "DiagramGenBenchmark_t2c"        # 输出数据集目录名
OUT_ROOT = "MMCoIR-test"                            # JSONL 输出根目录


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> List[Dict[str, Any]]:
    print(f"[LOAD] {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for key in ("data", "items", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    else:
        return []


def get_expanded_query(sample: Dict[str, Any]) -> Optional[str]:
    v = sample.get("expanded_query") or sample.get("query")
    if v is None:
        return None
    return str(v)


def get_reference_code(sample: Dict[str, Any]) -> Optional[str]:
    v = sample.get("reference") or sample.get("answer") or sample.get("code")
    if v is None:
        return None
    return str(v)


def to_test_item(qry_text: str, code_str: str) -> Dict[str, Any]:
    return {
        "qry_text": qry_text,
        "qry_img_path": "",
        "tgt_text": [code_str],
        "tgt_img_path": [""]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DiagramGenBenchmark DiagramGeneration subset to text-to-code (t2c) test JSONL")
    parser.add_argument("--input-json", type=str, default=str(Path("datasets")/DATASET_NAME/"DiagramGeneration.json"), help="输入 JSON（DiagramGeneration.json）路径")
    parser.add_argument("--out-dir", type=str, default=str(Path(OUT_ROOT)/OUT_DATASET_NAME), help="输出目录（将生成 test.jsonl）")
    parser.add_argument("--limit", type=int, default=None, help="仅转换前 N 条样本")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / "test.jsonl"
    ensure_dir(out_dir)

    # 读取数据
    try:
        samples = read_json(input_json)
    except FileNotFoundError:
        print(f"[ERROR] 未找到输入文件: {input_json}")
        return
    except Exception as e:
        print(f"[ERROR] 读取输入失败: {e}")
        return

    total = len(samples)
    if args.limit is not None:
        samples = samples[:args.limit]
    print(f"[INFO] 总样本: {total}，本次处理: {len(samples)}")

    test_items: List[Dict[str, Any]] = []
    drop_missing_q, drop_missing_code = 0, 0

    for i, sp in enumerate(samples, 1):
        qry_text = get_expanded_query(sp)
        code_str = get_reference_code(sp)

        if not qry_text:
            drop_missing_q += 1
            continue
        if not code_str:
            drop_missing_code += 1
            continue

        test_items.append(to_test_item(qry_text, code_str))

        if i % 2000 == 0:
            print(f"  [proc] {i}/{len(samples)}")

    # 保存 JSONL
    print(f"[SAVE] Test JSONL -> {out_jsonl}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] DiagramGenBenchmark DiagramGeneration -> T2C 测试集 完成")
    print(f"Test JSONL:  {out_jsonl}")
    print(f"[INFO] 丢弃样本 - 无 expanded_query: {drop_missing_q}, 无 reference/code: {drop_missing_code}")

    # 打印一个示例
    if test_items:
        print("\n=== 测试样本示例 ===")
        print(json.dumps(test_items[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()