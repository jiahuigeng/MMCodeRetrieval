#!/usr/bin/env python3
"""
Convert SD122025/ChartGen-200K dataset into multimodal retrieval format
according to programming_goals.md，生成 JSONL 并使用 train/images 与 test/images 相对路径（不复制图片、不校验存在性）。

- Input: local snapshot under datasets/ChartGen-200K (downloaded via download_chartgen.py)
- Output: JSONL files under MMCoIR/chartgen/{train.jsonl, test.jsonl}
- Images: 期望位于输出目录的 train/images/ 与 test/images/；脚本不复制也不校验。

Train JSONL fields:
- qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path

Test JSONL fields:
- qry_text, qry_img_path, tgt_text (list[str]), tgt_img_path (list[str])
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "ChartGen-200K"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MMCoIR" / "chartgen"
IMAGES_SUBDIR = "images"
TRAIN_SUBDIR = "train"
TEST_SUBDIR = "test"
CHARTGEN_DIRNAME = "chartgen"

PROMPT = (
    "Please take a look at this chart image.\n"
    "Consider you are a data visualization expert, and generate Python code that perfectly reconstructs this chart image.\n"
    "Make sure to redraw both the data points and the overall semantics and style of the chart as best as possible.\n"
    "In addition, ensure that the Python code is executable, and enclosed within triple backticks and labeled with python, like this:\n"
    "```python\n"
    "<your code here>\n"
    "```"
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


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> bool:
    try:
        if not src.exists():
            print(f"[WARN] Source image missing: {src}")
            return False
        if dst.exists():
            if overwrite:
                shutil.copy2(src, dst)
            else:
                # Already copied previously
                return True
        else:
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[WARN] Copy failed {src} -> {dst}: {e}")
        return False


def to_train_item(summary: str, code: str, rel_img_path: str) -> dict:
    return {
        "qry": f"<|image_1|>\n{PROMPT}",
        "qry_image_path": rel_img_path,
        "pos_text": str(code),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(summary: str, code: str, rel_img_path: str) -> dict:
    return {
        "qry_text": f"<|image_1|>\n{PROMPT}",
        "qry_img_path": rel_img_path,
        "tgt_text": [str(code)],
        "tgt_img_path": [rel_img_path],
    }


def convert_chartgen(input_root: Path, output_dir: Path, limit_train: int = None, limit_test: int = None, overwrite_images: bool = False):
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

        # Expect columns: image_path, summary, code; fallback when missing
        cols = df.columns.tolist()
        if "image_path" not in cols:
            print(f"[WARN] 'image_path' missing in {pq}; skipping this file")
            continue

        # Keep only useful columns when available
        keep_cols = [c for c in ["image_path", "summary", "code"] if c in cols]
        df = df[keep_cols]

        for idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    continue
                # Determine split from path
                split = "train" if "train/" in image_path else ("test" if "test/" in image_path else None)
                if split is None:
                    # Unknown split, skip
                    continue

                summary = row.get("summary", "")
                code = row.get("code", "")

                basename = Path(image_path).name
                rel_img_path = f"{CHARTGEN_DIRNAME}/{TRAIN_SUBDIR}/{IMAGES_SUBDIR}/{basename}" if split == "train" else f"{CHARTGEN_DIRNAME}/{TEST_SUBDIR}/{IMAGES_SUBDIR}/{basename}"

                if split == "train":
                    item = to_train_item(summary, code, rel_img_path)
                    train_items.append(item)
                    if limit_train is not None and len(train_items) >= limit_train:
                        # Early stop for train
                        pass
                elif split == "test":
                    item = to_test_item(summary, code, rel_img_path)
                    test_items.append(item)
                    if limit_test is not None and len(test_items) >= limit_test:
                        # Early stop for test
                        pass
            except Exception as e:
                print(f"[WARN] Failed processing row {idx} in {pq}: {e}")
                continue

    # Apply limits if specified (post-filter)
    if limit_train is not None:
        train_items = train_items[:limit_train]
    if limit_test is not None:
        test_items = test_items[:limit_test]

    # Save outputs
    train_out = out_dir / "train.jsonl"
    test_out = out_dir / "test.jsonl"

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
    print(f"Train JSONL: {train_out}")
    print(f"Test JSONL: {test_out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert ChartGen-200K to MMCoIR JSONL (no image copy; paths use train/images & test/images)")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory under MMCoIR/chartgen")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional limit for train samples")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional limit for test samples")
    parser.add_argument("--overwrite-images", action="store_true", help="[Deprecated] No copying; kept for compatibility")
    args = parser.parse_args()

    convert_chartgen(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        overwrite_images=args.overwrite_images,
    )


if __name__ == "__main__":
    main()