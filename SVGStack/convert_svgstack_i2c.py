#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 SVGStack 的 PNG 图片转换为 image-to-code (i2c) 检索格式 JSONL，并复制图片到统一目录。

数据来源：
- 默认从本地 `dataset/SVGStack/imgs/train` 与 `dataset/SVGStack/imgs/test` 扫描已存在 PNG 文件。
- 可选：从 HF 数据集 `starvector/svg-stack` 以流式方式读取 `Svg` 文本，用于作为目标代码（与 Filename 基名对齐）。

输出：
- 复制图片至：
  - 训练：MMCoIR-train/images/SVGStack/images/<name>.png
  - 测试：MMCoIR-test/images/SVGStack/images/<name>.png
- JSONL：
  - 训练：MMCoIR-train/SVGStack_i2c/train.jsonl
  - 测试：MMCoIR-test/SVGStack_i2c/test.jsonl

JSONL 字段（与 format_checker.py 一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

设计约定：
- 查询文本以图像 token 开头："<|image_1|>"；随后是指令："Please convert this image to svg code."。
- JSONL 中的图片路径使用 POSIX 风格相对路径："images/SVGStack/images/<name>.png"。
- 若无法找到对应的 Svg 文本：默认保留为空字符串（可用 --require-code 开启严格模式以跳过该样本）。

使用示例：
  python SVGStack/convert_svgstack_i2c.py \
    --train-src dataset/SVGStack/imgs/train \
    --test-src  dataset/SVGStack/imgs/test \
    --limit-train 100000 --limit-test 2000 \
    --overwrite-images --quiet
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasets import load_dataset


IMG_TOKEN = "<|image_1|>"
DEFAULT_PROMPT = "Please convert this image to svg code."
DATASET_NAME = "SVGStack"
HF_REPO_ID = "starvector/svg-stack"
HF_REVISION = "main"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_SRC = REPO_ROOT / "dataset" / "SVGStack" / "imgs" / "train"
DEFAULT_TEST_SRC = REPO_ROOT / "dataset" / "SVGStack" / "imgs" / "test"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DEFAULT_OUT_NAME = "SVGStack_i2c"


# ---------- utils ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_posix_rel(*parts: str) -> str:
    return "/".join(parts)


def list_pngs(src_dir: Path, limit: Optional[int] = None) -> List[Path]:
    if not src_dir.exists():
        return []
    files = sorted(src_dir.glob("*.png"))
    if limit is not None:
        files = files[: max(0, limit)]
    return files


def copy_image(src: Path, dst_dir: Path, overwrite: bool) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists() and not overwrite:
        return dst
    shutil.copy2(src, dst)
    return dst


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- build svg map from HF ----------

def build_svg_map(stems: Set[str], repo_id: str, revision: str, quiet: bool = False) -> Dict[str, str]:
    """从 HF 数据集按需要的文件基名(stem)构建映射：stem -> Svg 文本。
    为提升效率，流式遍历常见 splits，并在全部命中后提前退出。
    """
    if not stems:
        return {}
    svg_map: Dict[str, str] = {}
    target = set(stems)

    def _scan_split(split: str) -> None:
        nonlocal svg_map
        try:
            ds = load_dataset(repo_id, split=split, streaming=True, revision=revision)
        except Exception as e:
            if not quiet:
                print(f"[WARN] load_dataset split='{split}' failed: {e}")
            return
        for rec in ds:
            fn = rec.get("Filename") or rec.get("filename")
            if not isinstance(fn, str) or not fn:
                continue
            stem = Path(fn).stem
            if stem in target and stem not in svg_map:
                svg_text = rec.get("Svg") or rec.get("svg")
                if isinstance(svg_text, str) and svg_text:
                    svg_map[stem] = svg_text
                    if len(svg_map) >= len(target):
                        break

    for split in ("train", "test", "validation"):
        if len(svg_map) >= len(target):
            break
        _scan_split(split)

    if not quiet:
        print(f"[INFO] svg_map size={len(svg_map)} (requested {len(stems)})")
    return svg_map


# ---------- JSONL builders ----------

def to_train_item(rel_img: str, svg_code: Optional[str], prompt: str) -> Dict:
    return {
        "qry": f"{IMG_TOKEN}\n{prompt}",
        "qry_image_path": rel_img,
        "pos_text": svg_code or "",
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(rel_img: str, svg_code: Optional[str], prompt: str) -> Dict:
    return {
        "qry_text": f"{IMG_TOKEN}\n{prompt}",
        "qry_img_path": rel_img,
        "tgt_text": [svg_code or ""],
        "tgt_img_path": [""],
    }


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description="Convert SVGStack PNGs to i2c JSONL and copy images.")
    p.add_argument("--train-src", type=str, default=str(DEFAULT_TRAIN_SRC), help="源训练 PNG 目录")
    p.add_argument("--test-src", type=str, default=str(DEFAULT_TEST_SRC), help="源测试 PNG 目录")
    p.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="MMCoIR 训练根目录")
    p.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="MMCoIR 测试根目录")
    p.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME, help="JSONL 输出子目录名")
    p.add_argument("--limit-train", type=int, default=None, help="训练样本上限（默认全部）")
    p.add_argument("--limit-test", type=int, default=None, help="测试样本上限（默认全部）")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="查询指令文本")
    p.add_argument("--overwrite-images", action="store_true", help="若目标图片已存在则覆盖")
    p.add_argument("--quiet", action="store_true", help="静默模式")
    p.add_argument("--require-code", action="store_true", help="严格要求 Svg 代码存在，不存在则跳过该样本")
    p.add_argument("--fetch-svg", action="store_true", help="启用：从 HF 流式读取 Svg 以对齐文件名")
    p.add_argument("--repo-id", type=str, default=HF_REPO_ID, help="HF 数据集 repo id")
    p.add_argument("--revision", type=str, default=HF_REVISION, help="HF 数据集 revision")

    args = p.parse_args()

    train_src = Path(args.train_src).resolve()
    test_src = Path(args.test_src).resolve()
    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    out_name = args.out_name

    if not args.quiet:
        print(f"Train src: {train_src} | exists={train_src.exists()}")
        print(f"Test  src: {test_src}  | exists={test_src.exists()}")
        print(f"Train root: {train_root}")
        print(f"Test  root: {test_root}")
        print(f"Out name:   {out_name}")
        print(f"Limits:     train={args.limit_train}, test={args.limit_test}")
        print(f"Prompt:     {args.prompt}")
        print(f"Fetch Svg:  {args.fetch_svg} | require_code={args.require_code}")

    # 收集 PNG 文件
    train_pngs = list_pngs(train_src, limit=args.limit_train)
    test_pngs = list_pngs(test_src, limit=args.limit_test)

    # 目标图片目录（固定顶层 images 路径）
    train_images_dir = DEFAULT_TRAIN_ROOT / "images" / DATASET_NAME / "images"
    test_images_dir = DEFAULT_TEST_ROOT / "images" / DATASET_NAME / "images"
    ensure_dir(train_images_dir)
    ensure_dir(test_images_dir)

    # 复制图片
    for p in train_pngs:
        copy_image(p, train_images_dir, overwrite=args.overwrite_images)
    for p in test_pngs:
        copy_image(p, test_images_dir, overwrite=args.overwrite_images)

    # 构建 Svg 映射（可选）
    stem_train: Set[str] = {x.stem for x in train_pngs}
    stem_test: Set[str] = {x.stem for x in test_pngs}
    svg_map: Dict[str, str] = {}
    if args.fetch_svg:
        need_stems = stem_train | stem_test
        svg_map = build_svg_map(need_stems, args.repo_id, args.revision, quiet=args.quiet)

    # 构建 JSONL rows
    rows_train: List[Dict] = []
    rows_test: List[Dict] = []

    for p in train_pngs:
        stem = p.stem
        code = svg_map.get(stem)
        if args.require_code and not (isinstance(code, str) and code.strip()):
            continue
        rel_img = to_posix_rel("images", DATASET_NAME, "images", p.name)
        rows_train.append(to_train_item(rel_img, code, args.prompt))

    for p in test_pngs:
        stem = p.stem
        code = svg_map.get(stem)
        if args.require_code and not (isinstance(code, str) and code.strip()):
            continue
        rel_img = to_posix_rel("images", DATASET_NAME, "images", p.name)
        rows_test.append(to_test_item(rel_img, code, args.prompt))

    # 写入 JSONL
    out_train_jsonl = train_root / out_name / "train.jsonl"
    out_test_jsonl = test_root / out_name / "test.jsonl"
    if rows_train:
        write_jsonl(out_train_jsonl, rows_train)
        if not args.quiet:
            print(f"[SAVE] Train JSONL -> {out_train_jsonl} | rows={len(rows_train)}")
    else:
        if not args.quiet:
            print("[SAVE] Train JSONL skipped (no rows)")
    if rows_test:
        write_jsonl(out_test_jsonl, rows_test)
        if not args.quiet:
            print(f"[SAVE] Test JSONL  -> {out_test_jsonl} | rows={len(rows_test)}")
    else:
        if not args.quiet:
            print("[SAVE] Test JSONL skipped (no rows)")

    if not args.quiet:
        print("[DONE] SVGStack i2c conversion completed.")


if __name__ == "__main__":
    main()