"""
Convert WebSight (prepared by prepare_websight_subset.py) into Code-to-Image (c2i) retrieval JSONL.

Input:
- Arrow datasets prepared by WebSight/prepare_websight_subset.py
  Expected directories: datasets/WebSight/train, datasets/WebSight/test
  Each record should have at least: image (PIL.Image or bytes), raw_code (str)

Output:
- Copy images to unified directories under train/test roots:
  - train:  <train_root>/images/WebSight/images/<id>.png
  - test:   <test_root>/images/WebSight/images/<id>.png
- Generate JSONL with fields consistent with format_checker.py for c2i:
  - train fields:  qry, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path
    * qry: "Please convert this code to image." + code
    * qry_image_path: ""
    * pos_text: "<|image_1|>"
    * pos_image_path: relative path to copied image (POSIX style)
    * neg_text, neg_image_path: "" (not provided)
  - test fields:   qry_text, qry_img_path, tgt_text, tgt_img_path
    * qry_text: "Please convert this code to image." + code
    * qry_img_path: ""
    * tgt_text: ["<|image_1|>"]
    * tgt_img_path: [relative path to copied image (POSIX style)]

CLI:
  python WebSight/convert_websight_c2i.py \
    --train-root MMCoIR-train/WebSight_i2c \
    --test-root MMCoIR-test/WebSight_i2c \
    --input-train datasets/WebSight/train \
    --input-test datasets/WebSight/test \
    --limit-train 100 --limit-test 100 --overwrite-images

Notes:
- Paths in JSONL use POSIX style and start with "images/WebSight/images/..." so that
  format_checker.py can join them with provided --img-root.
- If your Arrow dataset uses a different field than 'raw_code', set --code-field.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


IMG_TOKEN = "<|image_1|>"
DEFAULT_PROMPT = "Please convert this code to image."
DATASET_NAME = "WebSight"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _posix_rel(sample_id: str) -> str:
    # Unified POSIX-style relative path under images/<Dataset>/images/
    return f"images/{DATASET_NAME}/images/{sample_id}.png".replace("\\", "/")


def _save_image(img: Image.Image, out_path: Path, overwrite: bool = False) -> bool:
    if out_path.exists() and not overwrite:
        return True
    try:
        img.save(out_path, format="PNG")
        return True
    except Exception as e:
        print(f"[WARN] Failed to save image to {out_path}: {e}")
        return False


def _extract_code(record: Dict[str, Any], code_field: str) -> Optional[str]:
    code = record.get(code_field)
    if code is None:
        return None
    if isinstance(code, (list, tuple)):
        # If arrow yields a list of strings, join them conservatively.
        try:
            return "\n".join(map(str, code))
        except Exception:
            return str(code)
    return str(code)


def _extract_image(record: Dict[str, Any]) -> Optional[Image.Image]:
    img = record.get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    # Some Arrow datasets may store bytes; try to open via PIL
    try:
        from io import BytesIO

        return Image.open(BytesIO(img)).convert("RGB")
    except Exception:
        return None


def build_train_item(sample_id: str, code: str, rel_img: str, prompt: str) -> Dict[str, Any]:
    return {
        "qry": f"{prompt}\n{code}",
        "qry_image_path": "",  # c2i: query has no image
        "pos_text": IMG_TOKEN,  # positive is image token aligned with pos_image_path
        "pos_image_path": rel_img,
        "neg_text": "",
        "neg_image_path": "",
    }


def build_test_item(sample_id: str, code: str, rel_img: str, prompt: str) -> Dict[str, Any]:
    return {
        "qry_text": f"{prompt}\n{code}",
        "qry_img_path": "",  # c2i: query has no image
        "tgt_text": [IMG_TOKEN],
        "tgt_img_path": [rel_img],
    }


def _iter_arrow_records(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load records from an Arrow dataset directory using pyarrow.dataset.
    Expects columns: 'image', 'raw_code'.
    """
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to read WebSight Arrow datasets. Please install it."
        ) from e

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset = ds.dataset(str(dataset_dir))
    # Read into Python rows; for large datasets consider streaming
    table = dataset.to_table()
    rows = table.to_pylist()
    return rows


def process_split(
    split_name: str,
    input_dir: Path,
    out_root: Path,
    limit: Optional[int],
    code_field: str,
    prompt: str,
    overwrite_images: bool,
    quiet: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    records = _iter_arrow_records(input_dir)
    if limit is not None:
        records = records[: max(0, int(limit))]

    images_dir = out_root / "images" / DATASET_NAME / "images"
    _ensure_dir(images_dir)

    jsonl_items: List[Dict[str, Any]] = []
    saved = 0

    for idx, rec in enumerate(records):
        code = _extract_code(rec, code_field)
        img = _extract_image(rec)
        if not code or img is None:
            if not quiet:
                print(f"[WARN] skip idx={idx} due to missing code or image")
            continue

        sample_id = f"websight_{split_name}_{idx:06d}"
        rel_img = _posix_rel(sample_id)
        out_img_path = images_dir / f"{sample_id}.png"

        if not _save_image(img, out_img_path, overwrite=overwrite_images):
            if not quiet:
                print(f"[WARN] failed to save image for idx={idx}")
            continue

        if split_name == "train":
            item = build_train_item(sample_id, code, rel_img, prompt)
        else:
            item = build_test_item(sample_id, code, rel_img, prompt)
        jsonl_items.append(item)
        saved += 1

    return jsonl_items, saved


def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert WebSight Arrow datasets to c2i JSONL format."
    )
    p.add_argument(
        "--input-train",
        type=str,
        default=str(Path("datasets") / "WebSight" / "train"),
        help="Arrow dataset directory for train",
    )
    p.add_argument(
        "--input-test",
        type=str,
        default=str(Path("datasets") / "WebSight" / "test"),
        help="Arrow dataset directory for test",
    )
    p.add_argument(
        "--train-root",
        type=str,
        default=str(Path("MMCoIR-train") / f"{DATASET_NAME}_c2i"),
        help="Output root for train JSONL and images",
    )
    p.add_argument(
        "--test-root",
        type=str,
        default=str(Path("MMCoIR-test") / f"{DATASET_NAME}_c2i"),
        help="Output root for test JSONL and images",
    )
    p.add_argument("--out-train", type=str, default="train.jsonl")
    p.add_argument("--out-test", type=str, default="test.jsonl")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument(
        "--code-field",
        type=str,
        default="raw_code",
        help="Field name in Arrow rows containing code",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Instruction prefix for the query text",
    )
    p.add_argument("--overwrite-images", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    _ensure_dir(train_root)
    _ensure_dir(test_root)

    # Process train
    train_items, train_saved = process_split(
        split_name="train",
        input_dir=Path(args.input_train),
        out_root=train_root,
        limit=args.limit_train,
        code_field=args.code_field,
        prompt=args.prompt,
        overwrite_images=args.overwrite_images,
        quiet=args.quiet,
    )
    write_jsonl(train_items, train_root / args.out_train)

    # Process test
    test_items, test_saved = process_split(
        split_name="test",
        input_dir=Path(args.input_test),
        out_root=test_root,
        limit=args.limit_test,
        code_field=args.code_field,
        prompt=args.prompt,
        overwrite_images=args.overwrite_images,
        quiet=args.quiet,
    )
    write_jsonl(test_items, test_root / args.out_test)

    if not args.quiet:
        print(
            f"Done. Train: {train_saved} items -> {train_root / args.out_train}\n"
            f"      Test: {test_saved} items -> {test_root / args.out_test}"
        )


if __name__ == "__main__":
    main()