#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于本地 OmniSVG MMSVG-Icon / MMSVG-Illustration 的 train.jsonl 构建多模态检索训练/测试集。

输出符合 programming_goals.md 与 format_checker.py 规范：
- 训练集（split=train）：
  {"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试集（split=test）：
  {"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

约定：
- 图像路径输出为相对路径：`<dataset>/imgs/<id>.png`（不包含根 `MMCoIR`）。
- PNG 存在性检查仍基于 `--img-root`，默认 `MMCoIR`。
- 数据读取默认来自 `datasets/<dataset>/train.jsonl`（可通过 --data-root 覆盖）。
- 若 PNG 文件缺失，可使用 `--allow-missing-imgs` 仍生成路径字符串（仅用于格式校验）。
- 负样本从同域随机采样（若 Illustration 存在，可混合作为更强负样本）。

示例运行：
- 生成训练集（允许缺失图片）：
  python build_omnisvg_retrieval.py --split train --limit 1000 --img-root MMCoIR --data-root datasets --allow-missing-imgs
- 生成测试集：
  python build_omnisvg_retrieval.py --split test --limit 500 --img-root MMCoIR --data-root datasets --allow-missing-imgs
"""
import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

DATA_ROOT_DEFAULT = "datasets"
IMG_ROOT_DEFAULT = "MMCoIR"
ICON_DIR = "MMSVG-Icon"
ILLUS_DIR = "MMSVG-Illustration"

PROMPT_QRY = (
    "<|image_1|>\n"
    "You are a helpful SVG Generation assistant, designed to generate SVG. \n"
    " We provide an image as input, generate SVG for this image."
)

def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
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

# 绝对路径（用于文件存在性检查）
def abs_png_path(img_root: str, dataset_name: str, sample_id: str) -> str:
    return os.path.join(img_root, dataset_name, "imgs", f"{sample_id}.png")

# 相对路径（输出到 JSONL，不含根目录）
def rel_png_path(dataset_name: str, sample_id: str) -> str:
    return os.path.join(dataset_name, "imgs", f"{sample_id}.png")


def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def build_pool(data_root: str, use_icon: bool = True, use_illus: bool = True, limit_icon: Optional[int] = None, limit_illus: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """读取两个数据集的 train.jsonl，构建 id->record 池"""
    pool: Dict[str, Dict[str, Any]] = {}
    if use_icon:
        icon_path = os.path.join(data_root, ICON_DIR, "train.jsonl")
        icon_data = read_jsonl(icon_path, limit_icon)
        for rec in icon_data:
            sid = rec.get("id")
            if isinstance(sid, str):
                pool[f"{ICON_DIR}:{sid}"] = {**rec, "dataset": ICON_DIR}
    if use_illus:
        illus_path = os.path.join(data_root, ILLUS_DIR, "train.jsonl")
        illus_data = read_jsonl(illus_path, limit_illus)
        for rec in illus_data:
            sid = rec.get("id")
            if isinstance(sid, str):
                pool[f"{ILLUS_DIR}:{sid}"] = {**rec, "dataset": ILLUS_DIR}
    return pool


def sample_negative(keys: List[str], avoid_key: str) -> Optional[str]:
    if not keys:
        return None
    for _ in range(5):
        k = random.choice(keys)
        if k != avoid_key:
            return k
    for k in keys:
        if k != avoid_key:
            return k
    return None

# 新增：按同一数据集筛选负样本

def sample_negative_same_dataset(keys: List[str], pool: Dict[str, Dict[str, Any]], dataset_name: str, avoid_key: str) -> Optional[str]:
    candidates = [k for k in keys if k != avoid_key and pool.get(k, {}).get("dataset") == dataset_name]
    if not candidates:
        return None
    return random.choice(candidates)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_train_items(img_root: str, pool: Dict[str, Dict[str, Any]], allow_missing_imgs: bool) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    items: List[Dict[str, str]] = []
    stats = {"found": 0, "missing": 0}
    keys = list(pool.keys())
    for key in keys:
        rec = pool[key]
        if rec.get("dataset") != ICON_DIR:
            continue
        sid = rec.get("id")
        svg = rec.get("svg")
        if not isinstance(sid, str) or not isinstance(svg, str):
            continue
        qry_img_abs = abs_png_path(img_root, ICON_DIR, sid)
        qry_img_rel = rel_png_path(ICON_DIR, sid)
        exists_qry = file_exists(qry_img_abs)
        stats["found" if exists_qry else "missing"] += 1
        if (not allow_missing_imgs) and (not exists_qry):
            continue
        pos_text = svg
        pos_img_abs = qry_img_abs
        pos_img_rel = qry_img_rel
        # 仅从同一数据集采样负样本
        neg_key = sample_negative_same_dataset(keys, pool, rec.get("dataset", ICON_DIR), avoid_key=key)
        if not neg_key:
            continue
        neg_rec = pool.get(neg_key)
        if not neg_rec:
            continue
        neg_sid = neg_rec.get("id")
        neg_svg = neg_rec.get("svg")
        neg_img_abs = abs_png_path(img_root, neg_rec.get("dataset", ICON_DIR), neg_sid) if isinstance(neg_sid, str) else ""
        neg_img_rel = rel_png_path(neg_rec.get("dataset", ICON_DIR), neg_sid) if isinstance(neg_sid, str) else ""
        exists_neg = file_exists(neg_img_abs)
        stats["found" if exists_neg else "missing"] += 1
        if (not allow_missing_imgs) and (not exists_neg):
            continue
        item = {
            "qry": PROMPT_QRY,
            "qry_image_path": qry_img_rel.replace("\\", "/"),
            "pos_text": pos_text,
            "pos_image_path": pos_img_rel.replace("\\", "/"),
            "neg_text": neg_svg if isinstance(neg_svg, str) else "",
            "neg_image_path": neg_img_rel.replace("\\", "/"),
        }
        items.append(item)
    return items, stats


def make_test_items(img_root: str, pool: Dict[str, Dict[str, Any]], allow_missing_imgs: bool) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    items: List[Dict[str, str]] = []
    stats = {"found": 0, "missing": 0}
    keys = list(pool.keys())
    for key in keys:
        rec = pool[key]
        if rec.get("dataset") != ICON_DIR:
            continue
        sid = rec.get("id")
        svg = rec.get("svg")
        if not isinstance(sid, str) or not isinstance(svg, str):
            continue
        img_abs = abs_png_path(img_root, ICON_DIR, sid)
        img_rel = rel_png_path(ICON_DIR, sid)
        exists_img = file_exists(img_abs)
        stats["found" if exists_img else "missing"] += 1
        if (not allow_missing_imgs) and (not exists_img):
            continue
        item = {
            "qry_text": PROMPT_QRY,
            "qry_img_path": img_rel.replace("\\", "/"),
            "tgt_text": svg,
            "tgt_img_path": img_rel.replace("\\", "/"),
        }
        items.append(item)
    return items, stats


def write_jsonl(path: str, items: List[Dict[str, str]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="构建OmniSVG多模态检索训练/测试集JSONL")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT, help="原始 JSONL 数据根目录（默认 datasets）")
    parser.add_argument("--img-root", default=IMG_ROOT_DEFAULT, help="PNG 图片根目录（默认 MMCoIR）")
    parser.add_argument("--use-icon", action="store_true", default=True, help="使用 MMSVG-Icon")
    parser.add_argument("--use-illus", action="store_true", default=True, help="使用 MMSVG-Illustration 作为负样本池")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="输出 split")
    parser.add_argument("--limit", type=int, default=1000, help="读取每个数据集的最多样本数（避免超大内存/IO）")
    parser.add_argument("--allow-missing-imgs", action="store_true", help="允许图片缺失仍生成路径（仅用于格式校验）")
    parser.add_argument("--out-dir", default=os.path.join("MMCoIR", "omnisvg_retrieval"), help="输出目录")

    args = parser.parse_args()

    pool = build_pool(
        data_root=args.data_root,
        use_icon=args.use_icon,
        use_illus=args.use_illus,
        limit_icon=args.limit,
        limit_illus=args.limit,
    )

    if not pool:
        print("[ERROR] 未能读取到任何样本。请确认 datasets/MMSVG-Icon 或 MMSVG-Illustration 的 train.jsonl 存在。")
        sys.exit(1)

    if args.split == "train":
        items, stats = make_train_items(args.img_root, pool, allow_missing_imgs=args.allow_missing_imgs)
        out_path = os.path.join(args.out_dir, "train_multimodal_retrieval.jsonl")
    else:
        items, stats = make_test_items(args.img_root, pool, allow_missing_imgs=args.allow_missing_imgs)
        out_path = os.path.join(args.out_dir, "test_multimodal_retrieval.jsonl")

    print(f"[INFO] 图片存在: {stats['found']}，缺失: {stats['missing']}")
    if not items:
        print("[WARN] 未生成任何条目。若图片路径在 <dataset>/imgs 下，请设置 --img-root 以正确检查存在性或启用 --allow-missing-imgs。")
    else:
        print(f"[OK] 生成条目数: {len(items)}")
    write_jsonl(out_path, items)
    print(f"[OK] 写入: {out_path}")


if __name__ == "__main__":
    main()