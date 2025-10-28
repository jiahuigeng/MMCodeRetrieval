#!/usr/bin/env python3
"""
将 SD122025/ChartGen-200K 转换为基于 summary 的多模态检索格式，
在同一目标文件夹下生成 train_sum.jsonl 与 test_sum.jsonl。

- Input: 本地数据快照位于 datasets/ChartGen-200K（通过 download_chartgen.py 下载）
- Output: JSONL 文件位于 MMCoIR/chartgen/{train_sum.jsonl, test_sum.jsonl}
- Images: 期望位于输出目录的 train/images 与 test/images，相对路径指向这些目录；脚本不复制、不校验图片存在性。

训练集 JSONL 字段（与规范一致）:
- qry: 带图像 token 的查询文本（包含 <|image_1|> 和指定 PROMPT）
- qry_image_path: 查询图片相对路径（train/images/<filename>）
- pos_text: 正样本文本（此脚本使用 summary 列）
- pos_image_path: 正样本图片路径（无 -> 空字符串）
- neg_text: 负样本文本（无 -> 空字符串）
- neg_image_path: 负样本图片路径（无 -> 空字符串）

测试集 JSONL 字段（与规范一致）:
- qry_text: 带图像 token 的查询文本（包含 <|image_1|> 和指定 PROMPT）
- qry_img_path: 查询图片相对路径（test/images/<filename>）
- tgt_text: 目标文本列表（List[str]，此脚本使用 summary 列）
- tgt_img_path: 目标图片路径列表（List[str]；此脚本无目标图像 -> [""]）
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "ChartGen-200K"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MMCoIR" / "chartgen"
IMAGES_SUBDIR = "images"
TRAIN_SUBDIR = "train"
TEST_SUBDIR = "test"

PROMPT = (
    "Please take a look at this chart image.\n"
    "Consider you are a data visualization expert and technical writer.\n"
    "Your task is to generate a concise yet informative natural language summary of what this chart depicts.\n"
)


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_images_dir = out_dir / TRAIN_SUBDIR / IMAGES_SUBDIR
    test_images_dir = out_dir / TEST_SUBDIR / IMAGES_SUBDIR
    train_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, train_images_dir, test_images_dir


def find_parquet_files(root: Path) -> List[Path]:
    parquet_files: List[Path] = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                parquet_files.append(Path(r) / fn)
    return sorted(parquet_files)


def to_train_item(summary: str, rel_img_path: str) -> dict:
    return {
        "qry": f"<|image_1|>\n{PROMPT}",
        "qry_image_path": rel_img_path,
        "pos_text": str(summary),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(summary: str, rel_img_path: str) -> dict:
    return {
        "qry_text": f"<|image_1|>\n{PROMPT}",
        "qry_img_path": rel_img_path,
        "tgt_text": [str(summary)],
        "tgt_img_path": [""],  # 无目标图像，使用空字符串占位
    }


def convert_chartgen_sum(
    input_root: Path,
    output_dir: Path,
    limit_train: int = None,
    limit_test: int = None,
):
    out_dir, train_images_dir, test_images_dir = ensure_dirs(output_dir)
    parquet_files = find_parquet_files(input_root)
    if not parquet_files:
        print(f"[ERROR] No parquet files found under {input_root}")
        return

    print(f"[INFO] Found {len(parquet_files)} parquet files")

    train_items: List[dict] = []
    test_items: List[dict] = []

    for pq in parquet_files:
        print(f"[INFO] Reading {pq} ...")
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[WARN] Failed to read {pq}: {e}")
            continue

        # 期望列: image_path, summary；若缺失 image_path 则跳过
        cols = df.columns.tolist()
        if "image_path" not in cols:
            print(f"[WARN] 'image_path' missing in {pq}; skipping this file")
            continue

        keep_cols = [c for c in ["image_path", "summary"] if c in cols]
        df = df[keep_cols]

        for idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    continue
                # 从路径判断分割
                split = "train" if "train/" in image_path else ("test" if "test/" in image_path else None)
                if split is None:
                    continue

                summary = row.get("summary", "")
                basename = Path(image_path).name
                rel_img_path = (
                    f"chartgen/{TRAIN_SUBDIR}/{IMAGES_SUBDIR}/{basename}" if split == "train"
                    else f"chartgen/{TEST_SUBDIR}/{IMAGES_SUBDIR}/{basename}"
                )

                if split == "train":
                    item = to_train_item(summary, rel_img_path)
                    train_items.append(item)
                else:  # test
                    item = to_test_item(summary, rel_img_path)
                    test_items.append(item)
            except Exception as e:
                print(f"[WARN] Failed processing row {idx} in {pq}: {e}")
                continue

    # 应用限制（可选）
    if limit_train is not None:
        train_items = train_items[:limit_train]
    if limit_test is not None:
        test_items = test_items[:limit_test]

    # 保存输出
    train_out = out_dir / "train_sum.jsonl"
    test_out = out_dir / "test_sum.jsonl"

    print(f"[INFO] Saving train set ({len(train_items)}) to {train_out}")
    with open(train_out, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Saving test set ({len(test_items)}) to {test_out}")
    with open(test_out, "w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] Conversion completed.")
    print(f"Train images directory: {train_images_dir}")
    print(f"Test images directory: {test_images_dir}")
    print(f"Train SUM JSONL: {train_out}")
    print(f"Test SUM JSONL: {test_out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert ChartGen-200K to MMCoIR SUM JSONL (no image copy; paths use train/images & test/images)"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory under MMCoIR/chartgen")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional limit for train samples")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional limit for test samples")
    args = parser.parse_args()

    convert_chartgen_sum(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )


if __name__ == "__main__":
    main()