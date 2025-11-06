#!/usr/bin/env python3
"""
任务说明（请勿擅自更改）
- 参考 convert_chartgen.py，将 datasets/ChartGen-200K 下的 Parquet 数据合并处理。
- 随机采样：训练 100000、测试 2000（可通过参数调整）。
- 输出：
  - 训练集 JSONL -> `MMCoIR-train/ChartGen/train.jsonl`
  - 测试集 JSONL -> `MMCoIR-test/ChartGen/test.jsonl`
- 复制图片：
  - 从 `datasets/ChartGen-200K/train/images/` 复制训练图片到 `MMCoIR-train/ChartGen/images/`
  - 从 `datasets/ChartGen-200K/test/images/` 复制测试图片到 `MMCoIR-test/ChartGen/images/`
- JSONL 中图片路径统一为相对路径 `images/<filename>`。
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "ChartGen-200K" / "data"
TRAIN_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-train" / "ChartGen"
TEST_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-test" / "ChartGen"
IMAGES_SUBDIR = "images"
TRAIN_SUBDIR = "train"
TEST_SUBDIR = "test"

PROMPT = (
    "Please take a look at this chart image.\n"
    "Consider you are a data visualization expert, and generate Python code that perfectly reconstructs this chart image.\n"
    "Make sure to redraw both the data points and the overall semantics and style of the chart as best as possible.\n"
    "In addition, ensure that the Python code is executable, and enclosed within triple backticks and labeled with python, like this:\n"
    "```python\n"
    "<your code here>\n"
    "```"
)


def ensure_dirs(train_root: Path, test_root: Path) -> Tuple[Path, Path]:
    train_images_dir = train_root / IMAGES_SUBDIR
    test_images_dir = test_root / IMAGES_SUBDIR
    train_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    return train_images_dir, test_images_dir


def find_parquet_files(root: Path) -> List[Path]:
    parquet_files: List[Path] = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                parquet_files.append(Path(r) / fn)
    return sorted(parquet_files)


def to_train_item(code: str, basename: str) -> dict:
    return {
        "qry": f"<|image_1|>\n{PROMPT}",
        "qry_image_path": f"ChartGen/{IMAGES_SUBDIR}/{basename}",
        "pos_text": str(code),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(code: str, basename: str) -> dict:
    return {
        "qry_text": f"<|image_1|>\n{PROMPT}",
        "qry_img_path": f"ChartGen/{IMAGES_SUBDIR}/{basename}",
        "tgt_text": [str(code)],
        "tgt_img_path": [""],
    }


def copy_image(src: Path, dst: Path) -> bool:
    try:
        if not src.exists():
            print(f"[WARN] Missing: {src}")
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[WARN] Copy fail {src} -> {dst}: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split ChartGen-200K into train/test JSONL and copy images")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--train-root", default=str(TRAIN_ROOT_DEFAULT), help="Output root for train")
    parser.add_argument("--test-root", default=str(TEST_ROOT_DEFAULT), help="Output root for test")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of train samples")
    parser.add_argument("--test-count", type=int, default=2_000, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    train_images_dir, test_images_dir = ensure_dirs(train_root, test_root)

    print(f"[INFO] Scanning parquet under {input_root}")
    parquets = find_parquet_files(input_root)
    if not parquets:
        raise SystemExit(f"No parquet files found under {input_root}")
    print(f"[INFO] Found {len(parquets)} parquet files")

    train_candidates: List[Tuple[str, str]] = []  # (basename, code)
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
        df = df[keep_cols]

        for _idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    continue
                code = row.get("code", "")
                basename = Path(image_path).name
                if "train/" in image_path:
                    train_candidates.append((basename, code))
                elif "test/" in image_path:
                    test_candidates.append((basename, code))
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

    # Convert
    train_items = [to_train_item(code=c, basename=b) for (b, c) in train_sample]
    test_items = [to_test_item(code=c, basename=b) for (b, c) in test_sample]

    # Save JSONL
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

    # Copy images
    # Source dirs in input dataset (parent of 'data')
    src_train_images = input_root.parent / TRAIN_SUBDIR / IMAGES_SUBDIR
    src_test_images = input_root.parent / TEST_SUBDIR / IMAGES_SUBDIR

    print(f"[INFO] Copying train images -> {train_images_dir}")
    miss_train = 0
    for i, (b, _c) in enumerate(train_sample, 1):
        ok = copy_image(src_train_images / b, train_images_dir / b)
        if not ok:
            miss_train += 1
        if i % 5000 == 0:
            print(f"[train] {i}/{train_n}")
    if miss_train:
        print(f"[WARN] Missing train images: {miss_train}/{train_n}")

    print(f"[INFO] Copying test images -> {test_images_dir}")
    miss_test = 0
    for i, (b, _c) in enumerate(test_sample, 1):
        ok = copy_image(src_test_images / b, test_images_dir / b)
        if not ok:
            miss_test += 1
        if i % 2000 == 0:
            print(f"[test] {i}/{test_n}")
    if miss_test:
        print(f"[WARN] Missing test images: {miss_test}/{test_n}")

    print("[DONE] ChartGen split completed.")
    print(f"Train JSONL: {train_json}")
    print(f"Test JSONL:  {test_json}")
    print(f"Train images: {train_images_dir}")
    print(f"Test images:  {test_images_dir}")


if __name__ == "__main__":
    main()