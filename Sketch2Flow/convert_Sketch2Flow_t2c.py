#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ServiceNow/BigDocs-Sketch2Flow 转换为“文本到代码 (t2c)”训练/测试集。

数据来源（本地已下载）：datasets/Sketch2Flow/data/*.parquet
- 列：
  - queries: 一个只有一个字符串的 list，作为查询文本
  - annotations: 一个只有一个字符串的 list，作为目标代码（结构化 JSON / DSL）

采样：
- 训练集：从 train+val 合并后随机采样 20000 条
- 测试集：从 test 采样 2000 条

输出：
- 训练 JSONL -> MMCoIR-train/Sketch2Flow_t2c/train.jsonl
- 测试 JSONL -> MMCoIR-test/Sketch2Flow_t2c/test.jsonl

JSONL 字段（与项目约定一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

说明：本任务不涉及图片作为查询或目标，所有 *_img_path / *_image_path 字段均置为空字符串，占位一致。
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


REPO_ROOT = Path(__file__).parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datasets" / "Sketch2Flow" / "data"
DEFAULT_TRAIN_ROOT = REPO_ROOT.parent / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT.parent / "MMCoIR-test"
DEFAULT_OUT_DATASET_NAME = "Sketch2Flow_t2c"


def extract_single_str(x: Any) -> Optional[str]:
    """取出列表中的第一个字符串；若已是字符串则直接返回；否则返回 None。"""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0:
        v = x[0]
        if isinstance(v, str):
            return v
    return None


def to_train_item(query_text: str, code_text: str) -> Dict[str, Any]:
    return {
        "qry": query_text,
        "qry_image_path": "",
        "pos_text": code_text,
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(query_text: str, code_text: str) -> Dict[str, Any]:
    return {
        "qry_text": query_text,
        "qry_img_path": "",
        "tgt_text": [code_text],
        "tgt_img_path": [""]
    }


def load_sketch2flow_local_parquets(data_dir: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    读取本地 parquet 文件，返回：
    - train+val 合并后的 (query, code) 列表
    - test 的 (query, code) 列表
    """
    train_files = sorted([str(p) for p in data_dir.glob("train-*.parquet")])
    val_files = sorted([str(p) for p in data_dir.glob("val-*.parquet")])
    test_files = sorted([str(p) for p in data_dir.glob("test-*.parquet")])

    if not train_files and not val_files and not test_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    data_files: Dict[str, List[str]] = {}
    if train_files or val_files:
        data_files["trainval"] = train_files + val_files
    if test_files:
        data_files["test"] = test_files

    ds_dict = load_dataset("parquet", data_files=data_files)

    def collect_pairs(dataset_split_name: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if dataset_split_name not in ds_dict:
            return pairs
        ds = ds_dict[dataset_split_name]
        for rec in ds:
            q = extract_single_str(rec.get("queries"))
            c = extract_single_str(rec.get("annotations"))
            if isinstance(q, str) and isinstance(c, str) and q.strip() and c.strip():
                pairs.append((q.strip(), c))
        return pairs

    trainval_pairs = collect_pairs("trainval")
    test_pairs = collect_pairs("test")
    return trainval_pairs, test_pairs


def main():
    parser = argparse.ArgumentParser(description="Convert Sketch2Flow parquet to t2c JSONL for retrieval")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="本地数据根：datasets/Sketch2Flow/data")
    parser.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="训练输出根：MMCoIR-train")
    parser.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="测试输出根：MMCoIR-test")
    parser.add_argument("--out-dataset-name", type=str, default=DEFAULT_OUT_DATASET_NAME, help="输出数据集目录名（默认 Sketch2Flow_t2c）")
    parser.add_argument("--train-count", type=int, default=20000, help="训练采样数量（来自 train+val 合并）")
    parser.add_argument("--test-count", type=int, default=2000, help="测试采样数量（来自 test）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    data_dir = Path(args.data_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    out_name = str(args.out_dataset_name)
    train_count = int(args.train_count)
    test_count = int(args.test_count)
    seed = int(args.seed)

    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Train root: {train_root}")
    print(f"[INFO] Test  root: {test_root}")
    print(f"[INFO] Output dataset name: {out_name}")

    trainval_pairs, test_pairs = load_sketch2flow_local_parquets(data_dir)
    print(f"[INFO] Train+Val usable pairs: {len(trainval_pairs)}")
    print(f"[INFO] Test       usable pairs: {len(test_pairs)}")

    random.seed(seed)
    random.shuffle(trainval_pairs)
    random.shuffle(test_pairs)

    if train_count > 0:
        train_sel = trainval_pairs[: min(train_count, len(trainval_pairs))]
    else:
        train_sel = []
    test_sel = test_pairs[: min(test_count, len(test_pairs))]

    # 准备输出路径
    out_train_json = train_root / out_name / "train.jsonl" if train_sel else None
    out_test_json = test_root / out_name / "test.jsonl"

    if out_train_json is not None:
        out_train_json.parent.mkdir(parents=True, exist_ok=True)
    out_test_json.parent.mkdir(parents=True, exist_ok=True)

    # 写训练 JSONL
    if out_train_json is not None:
        print(f"[SAVE] Train JSONL -> {out_train_json}")
        with out_train_json.open("w", encoding="utf-8") as f:
            for q, c in train_sel:
                item = to_train_item(q, c)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        print("[SAVE] Skipped train JSONL (train-count=0 or no data)")

    # 写测试 JSONL
    print(f"[SAVE] Test JSONL  -> {out_test_json}")
    with out_test_json.open("w", encoding="utf-8") as f:
        for q, c in test_sel:
            item = to_test_item(q, c)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] Sketch2Flow t2c split completed.")
    if out_train_json is not None:
        print(f"Train JSONL: {out_train_json}")
    else:
        print("Train JSONL: (skipped)")
    print(f"Test JSONL:  {out_test_json}")


if __name__ == "__main__":
    main()