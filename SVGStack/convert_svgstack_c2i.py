#!/usr/bin/env python3
"""
Convert SVGStack code (SVG) to image retrieval (c2i) format.

This script scans existing PNG images rendered from SVGStack under
`dataset/SVGStack/imgs/train` and `dataset/SVGStack/imgs/test`, copies them to
top-level images directories:
  - MMCoIR-train/images/SVGStack/images/
  - MMCoIR-test/images/SVGStack/images/

Then it builds JSONL files for code-to-image retrieval:
  - MMCoIR-train/SVGStack_c2i/train.jsonl
  - MMCoIR-test/SVGStack_c2i/test.jsonl

JSONL schema follows project conventions for c2i:
Train items:
  {
    "qry": "Please convert this svg code to image.\n<svg code>",
    "qry_image_path": "",
    "pos_text": "<|image_1|>",
    "pos_image_path": "images/SVGStack/images/<id>.png",
    "neg_text": "",
    "neg_image_path": ""
  }

Test items:
  {
    "qry_text": "Please convert this svg code to image.\n<svg code>",
    "qry_img_path": "",
    "tgt_text": ["<|image_1|>"],
    "tgt_img_path": ["images/SVGStack/images/<id>.png"]
  }

By default this script fetches SVG code from the Hugging Face dataset `starvector/svg-stack`
to populate the queries, and requires code presence (skipping samples without SVG).
You can disable fetching or code requirement via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMG_TOKEN = "<|image_1|>"
DEFAULT_PROMPT = "Please convert this svg code to image."

# Default local sources of PNGs (produced by convert_svgstack_svg_to_png.py)
DEFAULT_LOCAL_TRAIN_PNG_DIR = Path("dataset/SVGStack/imgs/train")
DEFAULT_LOCAL_TEST_PNG_DIR = Path("dataset/SVGStack/imgs/test")

# Top-level images roots used by this repo
DEFAULT_TRAIN_ROOT = Path("MMCoIR-train")
DEFAULT_TEST_ROOT = Path("MMCoIR-test")
DEFAULT_OUT_NAME = "SVGStack_c2i"

DATASET_NAME = "SVGStack"
IMAGES_SUBDIR = Path(f"images/{DATASET_NAME}/images")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_posix_rel(path: Path) -> str:
    return path.as_posix()


def list_pngs(src_dir: Path, limit: Optional[int] = None) -> List[Path]:
    if not src_dir.exists():
        return []
    files = sorted(src_dir.glob("*.png"))
    if limit is not None:
        files = files[:limit]
    return files


def copy_image(src: Path, dst_dir: Path, overwrite: bool = False) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists() and not overwrite:
        return dst
    shutil.copy2(src, dst)
    return dst


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_svg_map(fetch: bool, repo_id: str, revision: Optional[str]) -> Dict[str, str]:
    """
    Build a mapping from base filename (without extension) to SVG code text.
    When fetch=True, stream the HF dataset `starvector/svg-stack` to gather Svg field.
    """
    svg_map: Dict[str, str] = {}
    if not fetch:
        return svg_map
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"Failed to import datasets: {e}. Proceeding without SVG code.")
        return svg_map

    def _load_split(split: str):
        return load_dataset(
            repo_id,
            split=split,
            streaming=True,
            revision=revision,
        )

    # Try common split names
    splits = []
    try:
        splits.append(_load_split("train"))
    except Exception:
        pass
    try:
        splits.append(_load_split("test"))
    except Exception:
        pass
    try:
        splits.append(_load_split("validation"))
    except Exception:
        pass

    for ds in splits:
        for ex in ds:
            # Expect fields like Filename and Svg
            fname = ex.get("Filename") or ex.get("filename") or ex.get("name")
            svg = ex.get("Svg") or ex.get("svg") or ex.get("code")
            if not fname or not svg:
                continue
            base = Path(fname).stem
            # First write wins; avoid overriding if duplicate
            if base not in svg_map:
                svg_map[base] = svg
    return svg_map


def to_train_item(code_text: str, rel_img_path: str, prompt: str) -> Dict:
    return {
        "qry": f"{prompt}\n{code_text}",
        "qry_image_path": "",
        "pos_text": IMG_TOKEN,
        "pos_image_path": rel_img_path,
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(code_text: str, rel_img_path: str, prompt: str) -> Dict:
    return {
        "qry_text": f"{prompt}\n{code_text}",
        "qry_img_path": "",
        "tgt_text": [IMG_TOKEN],
        "tgt_img_path": [rel_img_path],
    }


def main():
    ap = argparse.ArgumentParser(description="Build SVGStack code-to-image (c2i) retrieval JSONL from local PNGs.")
    ap.add_argument("--train-src", type=Path, default=DEFAULT_LOCAL_TRAIN_PNG_DIR, help="Source dir of train PNGs.")
    ap.add_argument("--test-src", type=Path, default=DEFAULT_LOCAL_TEST_PNG_DIR, help="Source dir of test PNGs.")
    ap.add_argument("--train-limit", type=int, default=None, help="Limit number of train samples.")
    ap.add_argument("--test-limit", type=int, default=None, help="Limit number of test samples.")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Instruction prefix for queries.")
    ap.add_argument("--overwrite-images", action="store_true", help="Overwrite copied images if exist.")
    ap.add_argument("--fetch-svg", action="store_true", default=True, help="Fetch SVG code from HF dataset to build queries (default: on). Use --fetch-svg to keep enabled; disable via --no-fetch-svg if added.")
    ap.add_argument("--require-code", action="store_true", default=True, help="Skip samples if no SVG code available (default: on).")
    ap.add_argument("--repo-id", type=str, default="starvector/svg-stack", help="HF dataset repo id.")
    ap.add_argument("--revision", type=str, default=None, help="HF dataset revision/tag.")
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME, help="Output subdir name under MMCoIR-train/test.")
    args = ap.parse_args()

    # Discover local PNGs
    train_pngs = list_pngs(args.train_src, args.train_limit)
    test_pngs = list_pngs(args.test_src, args.test_limit)

    # Prepare top-level images dirs
    train_images_dir = DEFAULT_TRAIN_ROOT / IMAGES_SUBDIR
    test_images_dir = DEFAULT_TEST_ROOT / IMAGES_SUBDIR

    # Copy images and collect relative paths
    copied_train: List[Tuple[str, Path]] = []
    for p in train_pngs:
        dst = copy_image(p, train_images_dir, overwrite=args.overwrite_images)
        rel = IMAGES_SUBDIR / dst.name
        copied_train.append((p.stem, rel))

    copied_test: List[Tuple[str, Path]] = []
    for p in test_pngs:
        dst = copy_image(p, test_images_dir, overwrite=args.overwrite_images)
        rel = IMAGES_SUBDIR / dst.name
        copied_test.append((p.stem, rel))

    # Build SVG map if requested
    svg_map = build_svg_map(fetch=args.fetch_svg, repo_id=args.repo_id, revision=args.revision)

    # Build train JSONL rows
    train_rows: List[Dict] = []
    for base, rel in copied_train:
        code = svg_map.get(base)
        if args.require_code and not code:
            continue
        code = code or ""
        train_rows.append(to_train_item(code, to_posix_rel(rel), args.prompt))

    # Build test JSONL rows
    test_rows: List[Dict] = []
    for base, rel in copied_test:
        code = svg_map.get(base)
        if args.require_code and not code:
            continue
        code = code or ""
        test_rows.append(to_test_item(code, to_posix_rel(rel), args.prompt))

    # Write JSONL files
    train_out = DEFAULT_TRAIN_ROOT / args.out_name / "train.jsonl"
    test_out = DEFAULT_TEST_ROOT / args.out_name / "test.jsonl"
    write_jsonl(train_out, train_rows)
    write_jsonl(test_out, test_rows)

    print(f"Copied train images: {len(copied_train)}; rows written: {len(train_rows)} -> {train_out}")
    print(f"Copied test images:  {len(copied_test)}; rows written: {len(test_rows)}  -> {test_out}")
    print(f"Images dir (train): {train_images_dir}")
    print(f"Images dir (test):  {test_images_dir}")
    if args.fetch_svg:
        print(f"SVG map size: {len(svg_map)} (from {args.repo_id})")
    else:
        print("SVG fetching disabled; queries may be empty unless --require-code is set.")


if __name__ == "__main__":
    main()