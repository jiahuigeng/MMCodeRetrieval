#!/usr/bin/env python3
"""
任务说明（t2i 版本）
- 参考 convert_chartgen_i2c.py，将 datasets/ChartGen-200K 下的 Parquet 数据转换为文本到图片（text-to-image）检索格式。
- 文本来源：使用原始 `summary` 列作为查询文本。
- 随机采样：训练 100000、测试 2000（可通过参数调整）。
- 输出：
  - 训练集 JSONL -> `MMCoIR-train/ChartGen_t2i/train.jsonl`
  - 测试集 JSONL -> `MMCoIR-test/ChartGen_t2i/test.jsonl`
- 复制图片：
  - 从 `datasets/ChartGen-200K/train/images/` 复制训练图片到 `MMCoIR-train/images/ChartGen/images/`
  - 从 `datasets/ChartGen-200K/test/images/` 复制测试图片到 `MMCoIR-test/images/ChartGen/images/`
- JSONL 图片路径统一为相对路径 `images/ChartGen/images/<filename>`；
  - 训练集字段：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
  - 测试集字段：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}
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
TRAIN_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-train" / "ChartGen_t2i"
TEST_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-test" / "ChartGen_t2i"
IMAGES_SUBDIR = "images"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

IMAGE_TOKEN = "<|image_1|>"


def ensure_dirs(train_root: Path, test_root: Path) -> Tuple[Path, Path]:
    """确保公共图片目录存在：
    - 训练图片：MMCoIR-train/images/ChartGen/images/
    - 测试图片： MMCoIR-test/images/ChartGen/images/
    """
    train_images_dir = train_root.parent / IMAGES_SUBDIR / "ChartGen" / IMAGES_SUBDIR
    test_images_dir = test_root.parent / IMAGES_SUBDIR / "ChartGen" / IMAGES_SUBDIR
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


def to_train_item(summary: str, basename: str) -> dict:
    return {
        "qry": str(summary),
        "qry_image_path": "",
        "pos_text": IMAGE_TOKEN,
        "pos_image_path": f"images/ChartGen/images/{basename}",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(summary: str, basename: str) -> dict:
    return {
        "qry_text": str(summary),
        "qry_img_path": "",
        "tgt_text": [IMAGE_TOKEN],
        "tgt_img_path": [f"images/ChartGen/images/{basename}"],
    }


def copy_image(src: Path, dst: Path, overwrite: bool = False) -> bool:
    try:
        if not src.exists():
            return False
        if dst.exists() and not overwrite:
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ChartGen-200K text-to-image: split into train/test JSONL and copy images"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--train-root", default=str(TRAIN_ROOT_DEFAULT), help="Output root for train JSONL")
    parser.add_argument("--test-root", default=str(TEST_ROOT_DEFAULT), help="Output root for test JSONL")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of train samples")
    parser.add_argument("--test-count", type=int, default=2_000, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ids", type=str, default=None, help="Optional file with train image basenames (one per line) to use as exact IDs")
    parser.add_argument("--test-ids", type=str, default=None, help="Optional file with test image basenames (one per line) to use as exact IDs")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite destination images if exist")
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

    train_candidates: List[Tuple[str, str]] = []  # (basename, summary)
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
        keep_cols = [c for c in ["image_path", "summary"] if c in cols]
        if "summary" not in keep_cols:
            print(f"[WARN] 'summary' missing in {pq}; skip")
            continue
        df = df[keep_cols]

        for _idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                summary = str(row.get("summary", "")).strip()
                if not image_path or not summary:
                    continue
                basename = Path(image_path).name
                if f"{TRAIN_SPLIT}/" in image_path:
                    train_candidates.append((basename, summary))
                elif f"{TEST_SPLIT}/" in image_path:
                    test_candidates.append((basename, summary))
            except Exception as e:
                print(f"[WARN] Row parse error in {pq}: {e}")
                continue

    print(f"[INFO] Candidates: train={len(train_candidates)} test={len(test_candidates)}")

    def read_id_list(p: Path):
        try:
            with p.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return []

    # Build maps for fast lookup
    train_map = {b: s for (b, s) in train_candidates}
    test_map = {b: s for (b, s) in test_candidates}

    # Choose samples
    if args.train_ids:
        ids = read_id_list(Path(args.train_ids))
        train_sample = [(b, train_map[b]) for b in ids if b in train_map]
        train_n = len(train_sample)
    else:
        random.seed(args.seed)
        train_n = min(args.train_count, len(train_candidates))
        train_sample = random.sample(train_candidates, train_n)

    if args.test_ids:
        ids = read_id_list(Path(args.test_ids))
        test_sample = [(b, test_map[b]) for b in ids if b in test_map]
        test_n = len(test_sample)
    else:
        random.seed(args.seed)
        test_n = min(args.test_count, len(test_candidates))
        test_sample = random.sample(test_candidates, test_n)

    # Convert to JSON items
    train_items = [to_train_item(summary=s, basename=b) for (b, s) in train_sample]
    test_items = [to_test_item(summary=s, basename=b) for (b, s) in test_sample]

    # Save JSONL under ChartGen_t2i folder
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

    # Copy images to the common images bucket
    # Source dirs in input dataset (parent of 'data')
    src_train_images = input_root.parent / TRAIN_SPLIT / IMAGES_SUBDIR
    src_test_images = input_root.parent / TEST_SPLIT / IMAGES_SUBDIR

    print(f"[INFO] Copying train images -> {train_images_dir}")
    miss_train = 0
    for i, (b, _s) in enumerate(train_sample, 1):
        ok = copy_image(src_train_images / b, train_images_dir / b, overwrite=args.overwrite_images)
        if not ok:
            miss_train += 1
        if i % 5000 == 0:
            print(f"[train] {i}/{train_n}")
    if miss_train:
        print(f"[WARN] Missing train images: {miss_train}/{train_n}")

    print(f"[INFO] Copying test images -> {test_images_dir}")
    miss_test = 0
    for i, (b, _s) in enumerate(test_sample, 1):
        ok = copy_image(src_test_images / b, test_images_dir / b, overwrite=args.overwrite_images)
        if not ok:
            miss_test += 1
        if i % 2000 == 0:
            print(f"[test] {i}/{test_n}")
    if miss_test:
        print(f"[WARN] Missing test images: {miss_test}/{test_n}")

    print("[DONE] ChartGen t2i conversion completed.")
    print(f"Train JSONL: {train_json}")
    print(f"Test JSONL:  {test_json}")
    print(f"Train images: {train_images_dir}")
    print(f"Test images:  {test_images_dir}")


if __name__ == "__main__":
    main()