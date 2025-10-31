#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参照 convert_omnisvg.py，实现将 MMSVG-Icon / MMSVG-Illustration 数据集拆分为 train/test 两部分：
- 读取 datasets/<dataset>/train.jsonl（包含 id, svg, description 等字段）
- 训练集采样 100000，测试集采样 2000（默认；可参数覆盖），二者不重叠
- 输出：
  - 训练 JSONL -> MMCoIR-train/<dataset>_i2c/train.jsonl
  - 测试 JSONL -> MMCoIR-test/<dataset>_i2c/test.jsonl
- 复制图片：
  - 从 MMCoIR/<dataset>/imgs/<id>.png 复制到 MMCoIR-train/images/<dataset>/images/<id>.png （训练采样）
  - 从 MMCoIR/<dataset>/imgs/<id>.png 复制到 MMCoIR-test/images/<dataset>/images/<id>.png （测试采样）
- JSONL 中图片路径统一为相对路径：images/<dataset>/images/<id>.png（不含根目录，且以 images 开头）
- JSONL 字段符合 format_checker.py 规范：
  - 训练集：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
  - 测试集：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

提示：若 MMCoIR/<dataset>/imgs 缺失或尚未生成 PNG，请先运行 convert_omnisvg.py 完成 SVG->PNG 转换。
"""

import os
import re
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

REPO_ROOT = Path(__file__).parent
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_IMG_ROOT = REPO_ROOT / "MMCoIR"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DEFAULT_DATASETS = ["MMSVG-Icon", "MMSVG-Illustration"]

IMG_SRC_SUBDIR = "imgs"
SVG_SRC_SUBDIR = "svgs"
IMG_DST_SUBDIR = "images"

PROMPT_QRY = (
    "<|image_1|>\n"
    "You are a helpful SVG Generation assistant, designed to generate SVG. \n"
    " We provide an image as input, generate SVG for this image."
)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(name)).strip("._") or "item"


# ---------- IO helpers ----------

def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            data.append(obj)
            if limit is not None and len(data) >= limit:
                break
    return data


def ensure_dirs(train_root: Path, test_root: Path, dataset: str) -> Tuple[Path, Path]:
    train_images = train_root / dataset / IMG_DST_SUBDIR
    test_images = test_root / dataset / IMG_DST_SUBDIR
    train_images.mkdir(parents=True, exist_ok=True)
    test_images.mkdir(parents=True, exist_ok=True)
    return train_images, test_images


def copy_image(src: Path, dst: Path) -> bool:
    try:
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


# ---------- JSONL builders ----------

def to_train_item(dataset_images_name: str, sample_id: str, svg_code: str) -> Dict[str, Any]:
    # 使用原始文件基名，保持与源 imgs/<id>.png 一致（不做 sanitize）
    basename = f"{sample_id}.png"
    # JSONL 中图片路径以 images 开头，目录名不带后缀
    rel_img = f"images/{dataset_images_name}/{IMG_DST_SUBDIR}/{basename}"
    return {
        "qry": PROMPT_QRY,
        "qry_image_path": rel_img,
        "pos_text": str(svg_code),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(dataset_images_name: str, sample_id: str, svg_code: str) -> Dict[str, Any]:
    # 使用原始文件基名，保持与源 imgs/<id>.png 一致（不做 sanitize）
    basename = f"{sample_id}.png"
    # JSONL 中图片路径以 images 开头，目录名不带后缀
    rel_img = f"images/{dataset_images_name}/{IMG_DST_SUBDIR}/{basename}"
    return {
        "qry_text": PROMPT_QRY,
        "qry_img_path": rel_img,
        "tgt_text": [str(svg_code)],
        "tgt_img_path": [""]
    }


# ---------- main logic ----------

def process_dataset(
    dataset: str,
    data_root: Path,
    img_root: Path,
    train_root: Path,
    test_root: Path,
    train_count: int,
    test_count: int,
    seed: int,
) -> None:
    print(f"\n== Dataset: {dataset} ==")
    # Discover JSONL under datasets/<dataset>/train.jsonl (fallback: first *.jsonl)
    ds_dir = data_root / dataset
    jsonl_path = ds_dir / "train.jsonl"
    if not jsonl_path.exists():
        # fallback to any jsonl
        cand = None
        for p in ds_dir.glob("*.jsonl"):
            cand = p
            break
        jsonl_path = cand if cand else jsonl_path
    if not jsonl_path or not jsonl_path.exists():
        print(f"[WARN] JSONL missing: {jsonl_path}")
        return

    print(f"[INFO] Reading JSONL: {jsonl_path}")
    records = read_jsonl(jsonl_path)
    # 仅保留同时存在源 svgs 与 imgs 的样本（id 与文件基名一致）
    pairs_all: List[Tuple[str, str]] = []  # (id, svg_code)
    for rec in records:
        sid = rec.get("id")
        svg_code = rec.get("svg")
        if isinstance(sid, (str, int)) and isinstance(svg_code, str):
            pairs_all.append((str(sid), svg_code))

    if not pairs_all:
        print("[WARN] No usable records; skipping.")
        return

    src_imgs = img_root / dataset / IMG_SRC_SUBDIR
    src_svgs = img_root / dataset / SVG_SRC_SUBDIR
    if not src_imgs.exists() or not src_svgs.exists():
        print(f"[WARN] Source dirs missing (imgs/svgs): imgs={src_imgs.exists()} svgs={src_svgs.exists()} -> {src_imgs} | {src_svgs}")
        return

    pairs: List[Tuple[str, str]] = []
    for sid, svg_code in pairs_all:
        png_path = src_imgs / f"{sid}.png"
        svg_path = src_svgs / f"{sid}.svg"
        if png_path.exists() and svg_path.exists():
            pairs.append((sid, svg_code))

    total = len(pairs)
    print(f"[INFO] Available records with id+svg and both files exist: {total}")
    if total == 0:
        print("[WARN] No records with both PNG & SVG; skipping.")
        return

    # Deterministic shuffle
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    train_n = min(train_count, total)
    test_n = min(test_count, max(0, total - train_n))
    train_sel = pairs[:train_n]
    test_sel = pairs[train_n:train_n + test_n]
    print(f"[PLAN] Train: {len(train_sel)} | Test: {len(test_sel)}")

    # 统一 JSONL 的图片相对路径使用 images/<dataset>/images/<id>.png
    dataset_images_name = dataset

    # Build JSONL items
    train_items = [to_train_item(dataset_images_name, sid, svg) for sid, svg in train_sel]
    test_items = [to_test_item(dataset_images_name, sid, svg) for sid, svg in test_sel]

    # Save JSONL
    # 将 JSONL 保存到带 _i2c 后缀的目录（不变）
    out_test_json = test_root / f"{dataset}_i2c" / "test.jsonl"
    out_test_json.parent.mkdir(parents=True, exist_ok=True)
    out_train_json = None
    if train_n > 0:
        out_train_json = train_root / f"{dataset}_i2c" / "train.jsonl"
        out_train_json.parent.mkdir(parents=True, exist_ok=True)

    if out_train_json is not None:
        print(f"[SAVE] Train JSONL -> {out_train_json}")
        with out_train_json.open("w", encoding="utf-8") as f:
            for it in train_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    else:
        print("[SAVE] Skipped train JSONL (train-count=0)")

    print(f"[SAVE] Test JSONL  -> {out_test_json}")
    with out_test_json.open("w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Copy images（源文件基名与 id 完全一致，无 sanitize）
    # 目标路径为 <root>/images/<dataset>/images
    test_images_dir = test_root / "images" / dataset / IMG_DST_SUBDIR
    test_images_dir.mkdir(parents=True, exist_ok=True)
    if out_train_json is not None:
        train_images_dir = train_root / "images" / dataset / IMG_DST_SUBDIR
        train_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[COPY] Train images -> {train_images_dir}")
        miss_train = 0
        for i, (sid, _svg) in enumerate(train_sel, 1):
            base = f"{sid}.png"
            ok = copy_image(src_imgs / base, train_images_dir / base)
            if not ok:
                miss_train += 1
            if i % 5000 == 0:
                print(f"  [train] {i}/{len(train_sel)}")
        if miss_train:
            print(f"[WARN] Missing train PNGs: {miss_train}/{len(train_sel)}")
    else:
        print("[COPY] Skipped train images (train-count=0)")

    print(f"[COPY] Test images  -> {test_images_dir}")
    miss_test = 0
    for i, (sid, _svg) in enumerate(test_sel, 1):
        base = f"{sid}.png"
        ok = copy_image(src_imgs / base, test_images_dir / base)
        if not ok:
            miss_test += 1
        if i % 2000 == 0:
            print(f"  [test] {i}/{len(test_sel)}")
    if miss_test:
        print(f"[WARN] Missing test PNGs: {miss_test}/{len(test_sel)}")

    print("[DONE] OmniSVG split completed.")
    if out_train_json is not None:
        print(f"Train JSONL: {out_train_json}")
        print(f"Train images: {train_root / 'images' / dataset / IMG_DST_SUBDIR}")
    else:
        print("Train JSONL: (skipped)")
        print("Train images: (skipped)")
    print(f"Test JSONL:  {out_test_json}")
    print(f"Test images:  {test_images_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split MMSVG datasets into train/test JSONL and copy images")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Input datasets root (JSONL under <dataset>/train.jsonl)")
    parser.add_argument("--img-root", type=str, default=str(DEFAULT_IMG_ROOT), help="Source image root containing <dataset>/imgs")
    parser.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="Output root for train (<root>/<dataset>/train.jsonl & images)")
    parser.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="Output root for test (<root>/<dataset>/test.jsonl & images)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset names to process")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of train samples per dataset")
    parser.add_argument("--test-count", type=int, default=2_000, help="Number of test samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    img_root = Path(args.img_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    print(f"Data root: {data_root} {'(exists)' if data_root.exists() else '(missing)'}")
    print(f"Image root: {img_root} {'(exists)' if img_root.exists() else '(missing)'}")
    print(f"Train root: {train_root}")
    print(f"Test root:  {test_root}")

    for ds in args.datasets:
        process_dataset(
            dataset=ds,
            data_root=data_root,
            img_root=img_root,
            train_root=train_root,
            test_root=test_root,
            train_count=args.train_count,
            test_count=args.test_count,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()