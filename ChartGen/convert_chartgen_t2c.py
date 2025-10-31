#!/usr/bin/env python3
"""
任务说明（t2c 版本）
- 参考 convert_chartgen_i2c.py，将 datasets/ChartGen-200K 下的 Parquet 数据转换为文本到代码（text-to-code）检索格式。
- 随机采样：训练 100000、测试 2000（可通过参数调整）。
- 输出：
  - 训练集 JSONL -> `MMCoIR-train/ChartGen_t2c/train.jsonl`
  - 测试集 JSONL -> `MMCoIR-test/ChartGen_t2c/test.jsonl`
- 本任务不涉及图片：所有 *_img_path / *_image_path 字段均为空字符串，占位一致；不会复制任何图片。

JSONL 字段约定（与项目格式一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

字段映射（ChartGen-200K）：
- 训练：`qry` <- summary；`pos_text` <- code；其余 *_image_* 为空，neg_* 为空
- 测试：`qry_text` <- summary；`tgt_text` <- [code]；`qry_img_path` <- ""；`tgt_img_path` <- [""]
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "ChartGen-200K" / "data"
TRAIN_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-train" / "ChartGen_t2c"
TEST_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-test" / "ChartGen_t2c"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


def find_parquet_files(root: Path) -> List[Path]:
    parquet_files: List[Path] = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                parquet_files.append(Path(r) / fn)
    return sorted(parquet_files)


def to_train_item(summary: str, code: str) -> dict:
    return {
        "qry": str(summary),
        "qry_image_path": "",
        "pos_text": str(code),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(summary: str, code: str) -> dict:
    return {
        "qry_text": str(summary),
        "qry_img_path": "",
        "tgt_text": [str(code)],
        "tgt_img_path": [""],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ChartGen-200K text-to-code: split into train/test JSONL (no images)"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--train-root", default=str(TRAIN_ROOT_DEFAULT), help="Output root for train JSONL")
    parser.add_argument("--test-root", default=str(TEST_ROOT_DEFAULT), help="Output root for test JSONL")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of train samples")
    parser.add_argument("--test-count", type=int, default=2_000, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    print(f"[INFO] Scanning parquet under {input_root}")
    parquets = find_parquet_files(input_root)
    if not parquets:
        raise SystemExit(f"No parquet files found under {input_root}")
    print(f"[INFO] Found {len(parquets)} parquet files")

    train_candidates: List[Tuple[str, str]] = []  # (summary, code)
    test_candidates: List[Tuple[str, str]] = []

    for pq in parquets:
        print(f"[INFO] Reading {pq} ...")
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[WARN] Failed to read {pq}: {e}")
            continue

        cols = df.columns.tolist()
        if "image_path" not in cols:
            print(f"[WARN] 'image_path' missing in {pq}; skip")
            continue
        keep_cols = [c for c in ["image_path", "summary", "code"] if c in cols]
        if "summary" not in keep_cols or "code" not in keep_cols:
            print(f"[WARN] 'summary' or 'code' missing in {pq}; skip")
            continue
        df = df[keep_cols]

        for _idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                summary = str(row.get("summary", "")).strip()
                code = row.get("code", "")
                if not image_path or not summary or code is None:
                    continue
                # 按 image_path 中的 train/test 归属划分
                if f"{TRAIN_SPLIT}/" in image_path:
                    train_candidates.append((summary, str(code)))
                elif f"{TEST_SPLIT}/" in image_path:
                    test_candidates.append((summary, str(code)))
            except Exception as e:
                print(f"[WARN] Row parse error in {pq}: {e}")
                continue

    print(f"[INFO] Candidates: train={len(train_candidates)} test={len(test_candidates)}")

    # Random sample
    random.seed(args.seed)
    train_n = min(args.train_count, len(train_candidates))
    test_n = min(args.test_count, len(test_candidates))
    train_sample = random.sample(train_candidates, train_n)
    test_sample = random.sample(test_candidates, test_n)

    # Convert to JSON items
    train_items = [to_train_item(summary=s, code=c) for (s, c) in train_sample]
    test_items = [to_test_item(summary=s, code=c) for (s, c) in test_sample]

    # Save JSONL（目录后缀为 _t2c，文件名无后缀）
    train_json = train_root / "train.jsonl"
    test_json = test_root / "test.jsonl"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving train ({len(train_items)}) -> {train_json}")
    with open(train_json, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[INFO] Saving test ({len(test_items)}) -> {test_json}")
    with open(test_json, "w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print("[DONE] ChartGen t2c conversion completed.")
    print(f"Train JSONL: {train_json}")
    print(f"Test JSONL:  {test_json}")


if __name__ == "__main__":
    main()