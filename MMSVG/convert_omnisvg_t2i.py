#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 OmniSVG 的 MMSVG-Icon / MMSVG-Illustration 数据集转换为“文本到图片(t2i)”训练/测试集。

输出：
- 训练 JSONL -> MMCoIR-train/<dataset>_t2i/train.jsonl
- 测试 JSONL -> MMCoIR-test/<dataset>_t2i/test.jsonl

图片复制目标：
- 训练图片 -> MMCoIR-train/images/<dataset>/images/<id>.png（已存在则跳过）
- 测试图片 -> MMCoIR-test/images/<dataset>/images/<id>.png（已存在则跳过）

JSONL 字段（与 format_checker.py 一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

字段含义（t2i）：
- qry_text / qry: 通用 prompt + 原始数据集的 description 字段内容
  prompt: "You are a helpful assistant that writes code from text descriptions.\n"
- qry_img_path / qry_image_path: 空字符串 ""
- tgt_img_path / pos_image_path: 目标图片相对路径（images/<dataset>/images/<id>.png）
- tgt_text / pos_text: 仅包含一个 image token（"<|image_1|>"）

数据来源：datasets/<dataset>/train.jsonl（需要 `id`, `description` 字段）。
图片来源：MMCoIR/<dataset>/imgs/<id>.png
"""

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_IMG_ROOT = REPO_ROOT / "MMCoIR"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DEFAULT_DATASETS = ["MMSVG-Icon", "MMSVG-Illustration"]

IMG_SRC_SUBDIR = "imgs"
IMG_DST_SUBDIR = "images"

PROMPT_T2I = (
    "You are a helpful assistant that writes code from text descriptions.\n"
)


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


# ---------- JSONL builders ----------

def to_train_item(dataset_images_name: str, sample_id: str, description: str) -> Dict[str, Any]:
    basename = f"{sample_id}.png"
    rel_img = f"images/{dataset_images_name}/{IMG_DST_SUBDIR}/{basename}"
    return {
        "qry": f"{PROMPT_T2I}{str(description)}",
        "qry_image_path": "",
        "pos_text": "<|image_1|>",
        "pos_image_path": rel_img,
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(dataset_images_name: str, sample_id: str, description: str) -> Dict[str, Any]:
    basename = f"{sample_id}.png"
    rel_img = f"images/{dataset_images_name}/{IMG_DST_SUBDIR}/{basename}"
    return {
        "qry_text": f"{PROMPT_T2I}{str(description)}",
        "qry_img_path": "",
        "tgt_text": ["<|image_1|>"],
        "tgt_img_path": [rel_img],
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
    overwrite_images: bool = False,
) -> None:
    print(f"\n== Dataset (t2i): {dataset} ==")
    # 找到 datasets/<dataset>/train.jsonl（若不存在则回退为第一个 *.jsonl）
    ds_dir = data_root / dataset
    jsonl_path = ds_dir / "train.jsonl"
    if not jsonl_path.exists():
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

    # 收集 id 与 description
    pairs_all: List[Tuple[str, str]] = []  # (id, description)
    for rec in records:
        sid = rec.get("id")
        description = rec.get("description")
        if isinstance(sid, (str, int)) and isinstance(description, str):
            pairs_all.append((str(sid), description))

    if not pairs_all:
        print("[WARN] No usable records (need id, description); skipping.")
        return

    src_imgs = img_root / dataset / IMG_SRC_SUBDIR
    if not src_imgs.exists():
        print(f"[WARN] Source image dir missing: {src_imgs}")
        return

    # 仅保留存在 PNG 的样本
    pairs: List[Tuple[str, str]] = []
    for sid, desc in pairs_all:
        png_path = src_imgs / f"{sid}.png"
        if png_path.exists():
            pairs.append((sid, desc))

    total = len(pairs)
    print(f"[INFO] Available records with id+description and PNG exists: {total}")
    if total == 0:
        print("[WARN] No records with PNG; skipping.")
        return

    # 固定随机划分
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    train_n = min(train_count, total)
    test_n = min(test_count, max(0, total - train_n))
    train_sel = pairs[:train_n]
    test_sel = pairs[train_n:train_n + test_n]
    print(f"[PLAN] Train: {len(train_sel)} | Test: {len(test_sel)}")

    dataset_images_name = dataset  # JSONL 图片路径使用 images/<dataset>/images/<id>.png

    # 构造 JSONL 项
    train_items = [to_train_item(dataset_images_name, sid, desc) for sid, desc in train_sel]
    test_items  = [to_test_item(dataset_images_name, sid, desc)  for sid, desc in test_sel]

    # 保存到 _t2i 目录
    out_test_json = test_root / f"{dataset}_t2i" / "test.jsonl"
    out_test_json.parent.mkdir(parents=True, exist_ok=True)
    out_train_json = None
    if train_n > 0:
        out_train_json = train_root / f"{dataset}_t2i" / "train.jsonl"
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

    # 复制图片到 <root>/images/<dataset>/images，已存在则跳过（除非指定覆盖）
    test_images_dir = test_root / "images" / dataset / IMG_DST_SUBDIR
    test_images_dir.mkdir(parents=True, exist_ok=True)
    if out_train_json is not None:
        train_images_dir = train_root / "images" / dataset / IMG_DST_SUBDIR
        train_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[COPY] Train images -> {train_images_dir}")
        miss_train = 0
        for i, (sid, _desc) in enumerate(train_sel, 1):
            base = f"{sid}.png"
            ok = copy_image(src_imgs / base, train_images_dir / base, overwrite=False)
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
    for i, (sid, _desc) in enumerate(test_sel, 1):
        base = f"{sid}.png"
        ok = copy_image(src_imgs / base, test_images_dir / base, overwrite=False)
        if not ok:
            miss_test += 1
        if i % 2000 == 0:
            print(f"  [test] {i}/{len(test_sel)}")
    if miss_test:
        print(f"[WARN] Missing test PNGs: {miss_test}/{len(test_sel)}")

    print("[DONE] OmniSVG t2i split completed.")
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
    parser = argparse.ArgumentParser(description="Convert MMSVG datasets to t2i train/test JSONL and copy images")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Input datasets root (JSONL under <dataset>/train.jsonl)")
    parser.add_argument("--img-root", type=str, default=str(DEFAULT_IMG_ROOT), help="Source image root containing <dataset>/imgs")
    parser.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="Output root for train (<root>/<dataset>_t2i/train.jsonl & images/<dataset>/images)")
    parser.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="Output root for test (<root>/<dataset>_t2i/test.jsonl & images/<dataset>/images)")
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