#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查原始 MMSVG-Icon / MMSVG-Illustration 的 PNG 图片是否存在（基于 train.jsonl 的 id 字段）。

默认读取：
- 数据根（--data-root）：MMCoIR
- 图片根（--img-root）：MMCoIR
- 期望结构：
  - <data-root>/<dataset>/train.jsonl       # 含字段 id, svg, ...
  - <img-root>/<dataset>/imgs/<id>.png       # 对应 PNG

可切换到 datasets 根：
- 例如：--data-root datasets --img-root MMCoIR

用法示例：
  python check_mmsvg_original_images.py           # 检查两套数据，默认根 MMCoIR
  python check_mmsvg_original_images.py --data-root datasets --img-root MMCoIR
  python check_mmsvg_original_images.py --datasets MMSVG-Icon MMSVG-Illustration
  python check_mmsvg_original_images.py --limit 10000 --max-report 30
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).parent
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_IMG_ROOT = REPO_ROOT / "datasets"
DEFAULT_DATASETS = ["MMSVG-Icon", "MMSVG-Illustration"]


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


def abs_png_path(img_root: Path, dataset: str, sample_id: str) -> Path:
    return img_root / dataset / "imgs" / f"{sample_id}.png"


def abs_svg_path(img_root: Path, dataset: str, sample_id: str) -> Path:
    return img_root / dataset / "svgs" / f"{sample_id}.svg"


def check_dataset(
    data_root: Path,
    img_root: Path,
    dataset: str,
    limit: Optional[int],
    max_report: int,
    check_kind: str = "any",
    max_exists_report: int = 20,
) -> Tuple[Dict[str, int], List[Tuple[int, str, Path, Path]], List[Tuple[int, str, Path, Path]]]:
    """
    逐行读取并检测，避免一次性加载全部样本，实现“边读边检”。
    """
    jsonl_path = data_root / dataset / "train.jsonl"
    stats = {"total": 0, "exists_any": 0, "missing": 0, "exists_png": 0, "exists_svg": 0, "exists_both": 0}
    missing_examples: List[Tuple[int, str, Path, Path]] = []  # (line_idx, sample_id, png_path, svg_path)
    exists_both_examples: List[Tuple[int, str, Path, Path]] = []  # (line_idx, sample_id, png_path, svg_path)

    if not jsonl_path.exists():
        print(f"[WARN] 文件不存在：{jsonl_path}")
        return stats, missing_examples, exists_both_examples

    processed = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # 非法 JSON 行，跳过但不计入 total
                continue
            stats["total"] += 1

            sid = rec.get("id")
            if not isinstance(sid, str) or not sid.strip():
                # 如果没有 id，跳过该条，不参与存在/缺失统计
                if limit is not None and stats["total"] >= limit:
                    break
                continue

            png_path = abs_png_path(img_root, dataset, sid)
            svg_path = abs_svg_path(img_root, dataset, sid)
            has_png = png_path.exists()
            has_svg = svg_path.exists()

            if has_png:
                stats["exists_png"] += 1
            if has_svg:
                stats["exists_svg"] += 1

            if has_png and has_svg:
                stats["exists_both"] += 1
                if len(exists_both_examples) < max_exists_report:
                    exists_both_examples.append((idx, sid, png_path, svg_path))

            if check_kind == "png":
                exists = has_png
            elif check_kind == "svg":
                exists = has_svg
            else:  # any
                exists = (has_png or has_svg)

            if exists:
                stats["exists_any"] += 1
            else:
                stats["missing"] += 1
                if len(missing_examples) < max_report:
                    missing_examples.append((idx, sid, png_path, svg_path))

            processed += 1
            if limit is not None and processed >= limit:
                break

    if stats["total"] == 0:
        print(f"[WARN] 无可读样本：{jsonl_path}（文件为空或无有效 JSON 行）")

    return stats, missing_examples, exists_both_examples


def main():
    parser = argparse.ArgumentParser(description="检查 MMSVG-Icon / MMSVG-Illustration 原始图片是否存在（PNG或SVG）")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="原始 JSONL 根目录（默认 datasets）")
    parser.add_argument("--img-root", type=str, default=str(DEFAULT_IMG_ROOT), help="原始图片根目录（默认 datasets）")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="要检查的数据集名称列表")
    parser.add_argument("--limit", type=int, default=None, help="最多检查的样本数（默认全部）")
    parser.add_argument("--max-report", type=int, default=20, help="最多打印的缺失样本条数")
    parser.add_argument("--check-kind", type=str, choices=["any", "png", "svg"], default="any", help="检查依据：any=PNG或SVG任一存在即视为存在；png=仅检查PNG；svg=仅检查SVG")
    parser.add_argument("--max-exists-report", type=int, default=20, help="最多打印的存在样本（PNG与SVG同时存在）条数")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    img_root = Path(args.img_root)

    print(f"Data root: {data_root} {'(exists)' if data_root.exists() else '(missing)'}")
    print(f"Img  root: {img_root} {'(exists)' if img_root.exists() else '(missing)'}")

    grand_total = {"total": 0, "exists": 0, "missing": 0}

    for ds in args.datasets:
        print(f"\n=== Dataset: {ds} ===")
        stats, missing, exist_both = check_dataset(
            data_root,
            img_root,
            ds,
            limit=args.limit,
            max_report=args.max_report,
            check_kind=args.check_kind,
            max_exists_report=args.max_exists_report,
        )
        if args.check_kind == "png":
            print(f"  total={stats['total']} exists_png={stats['exists_png']} missing={stats['missing']}")
        elif args.check_kind == "svg":
            print(f"  total={stats['total']} exists_svg={stats['exists_svg']} missing={stats['missing']}")
        else:
            print(f"  total={stats['total']} exists_any={stats['exists_any']} (png={stats['exists_png']}, svg={stats['exists_svg']}, both={stats['exists_both']}) missing={stats['missing']}")
        if exist_both:
            print(f"  Exists (both PNG & SVG present), showing up to {args.max_exists_report} samples:")
            for (line_idx, sid, png_path, svg_path) in exist_both:
                print(f"    - exist line#{line_idx} id='{sid}' -> png:{png_path} svg:{svg_path}")
        for (line_idx, sid, png_path, svg_path) in missing:
            print(f"  - missing line#{line_idx} id='{sid}' -> png:{png_path} svg:{svg_path}")
        grand_total["total"] += stats["total"]
        # 聚合为 any 的存在计数，便于整体汇总
        grand_total["exists"] += stats.get("exists_any", 0)
        grand_total["missing"] += stats["missing"]

    print("\n==== Summary ====")
    print(f"All datasets: total={grand_total['total']} exists={grand_total['exists']} missing={grand_total['missing']}")


if __name__ == "__main__":
    main()