#!/usr/bin/env python3
"""
将 Hugging Face 数据集 starvector/svg-stack 的 Svg 字段渲染为 PNG，
文件名使用 Filename 的基名，并保存到 `dataset/SVGStack/imgs/train/` 和 `dataset/SVGStack/imgs/test/`。

- train 集合转换 100000 个样本
- test 集合转换 2000 个样本（若数据集无 test split，则从 train 跳过前 100000 个后再取 2000 个样本）

依赖：
  pip install datasets cairosvg tqdm

使用示例：
  python SVGStack/convert_svgstack_svg_to_png.py \
    --outdir dataset/SVGStack/imgs \
    --train_count 100000 \
    --test_count 2000
"""

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset
from tqdm import tqdm
import cairosvg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render SVG to PNG from starvector/svg-stack dataset.")
    p.add_argument("--repo_id", type=str, default="starvector/svg-stack", help="HF dataset repo id")
    p.add_argument("--revision", type=str, default="main", help="Dataset revision/tag/branch")
    p.add_argument("--outdir", type=str, default=str(Path("dataset") / "SVGStack" / "imgs"), help="Output root directory")
    p.add_argument("--train_count", type=int, default=100_000, help="Number of train samples to convert")
    p.add_argument("--test_count", type=int, default=2_000, help="Number of test samples to convert")
    p.add_argument("--background", type=str, default=None, help="Optional background color, e.g. 'white' or '#FFFFFF'")
    return p.parse_args()


def svg_to_png_bytes(svg_text: str, background: Optional[str] = None) -> bytes:
    return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), background_color=background)


def convert_split(samples: Iterable, out_dir: Path, limit: int, background: Optional[str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_count = sum(1 for _ in out_dir.glob("*.png"))
    new_needed = max(0, limit - existing_count)
    if new_needed == 0:
        print(f"[INFO] {out_dir} already has {existing_count} PNGs; nothing to do.")
        return existing_count
    pbar = tqdm(total=new_needed, desc=f"Writing {out_dir.name}")
    new_saved = 0
    for ex in samples:
        if new_saved >= new_needed:
            break
        try:
            filename = ex.get("Filename")
            svg_text = ex.get("Svg")
            if not filename or not svg_text:
                continue
            png_name = f"{Path(filename).stem}.png"
            out_path = out_dir / png_name
            if out_path.exists():
                continue
            png_bytes = svg_to_png_bytes(svg_text, background)
            out_path.write_bytes(png_bytes)
            new_saved += 1
            pbar.update(1)
        except Exception:
            continue
    pbar.close()
    final_total = sum(1 for _ in out_dir.glob("*.png"))
    return final_total


def main() -> None:
    args = parse_args()

    out_root = Path(args.outdir)
    train_dir = out_root / "train"
    test_dir = out_root / "test"

    # 加载 train split（流式）
    print(f"[INFO] Loading '{args.repo_id}' (train, streaming=True, revision={args.revision})")
    ds_train = load_dataset(args.repo_id, split="train", streaming=True, revision=args.revision)

    # 优先尝试 test split；若不可用，稍后从 train 中跳过前 train_count 后再取 test_count
    ds_test = None
    try:
        print(f"[INFO] Loading '{args.repo_id}' (test, streaming=True, revision={args.revision})")
        ds_test = load_dataset(args.repo_id, split="test", streaming=True, revision=args.revision)
    except Exception:
        print("[WARN] Dataset has no 'test' split; will sample from 'train' after skipping train_count.")

    # 转换 train
    print(f"[INFO] Converting train -> {train_dir} (target={args.train_count})")
    train_total = convert_split(ds_train, train_dir, args.train_count, args.background)
    print(f"[INFO] Train total PNGs: {train_total} (target={args.train_count})")

    # 转换 test
    if ds_test is not None:
        print(f"[INFO] Converting test -> {test_dir} (target={args.test_count})")
        test_total = convert_split(ds_test, test_dir, args.test_count, args.background)
    else:
        print(f"[INFO] Converting pseudo-test from train (skip {args.train_count}) -> {test_dir} (target={args.test_count})")
        ds_train_2 = load_dataset(args.repo_id, split="train", streaming=True, revision=args.revision)
        # 从第 train_count 条开始迭代到结尾，convert_split 会自己填满 args.test_count
        test_iter = itertools.islice(ds_train_2, args.train_count, None)
        test_total = convert_split(test_iter, test_dir, args.test_count, args.background)
    print(f"[INFO] Test total PNGs: {test_total} (target={args.test_count})")

    print(
        f"[DONE] PNGs saved under: {out_root.resolve()}\n"
        f"  - train total: {train_total}\n"
        f"  - test  total: {test_total}"
    )


if __name__ == "__main__":
    main()