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

import shutil
from io import BytesIO
from PIL import Image
from datasets import load_from_disk


IMG_TOKEN = "<|image_1|>"
DEFAULT_PROMPT = "Please convert this code to image."
DATASET_NAME = "WebSight"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _posix_rel(sample_id: str) -> str:
    # Unified POSIX-style relative path under images/<Dataset>/images/
    return f"images/{DATASET_NAME}/images/{sample_id}.png".replace("\\", "/")

def _extract_image_ref(img_entry: Any) -> Tuple[str, Any]:
    """Return (kind, value) for image entry. kind in {"path", "pil", "bytes", "unknown"}."""
    if isinstance(img_entry, dict):
        b = img_entry.get("bytes")
        p = img_entry.get("path")
        if b is not None:
            return "bytes", b
        if isinstance(p, str) and p:
            return "path", p
        return "unknown", img_entry
    if hasattr(img_entry, "convert"):
        return "pil", img_entry
    if isinstance(img_entry, (bytes, bytearray)):
        return "bytes", img_entry
    if isinstance(img_entry, str):
        return "path", img_entry
    return "unknown", img_entry

def _save_image(img_kind: str, img_value: Any, dst_png: Path, overwrite: bool = False) -> bool:
    try:
        if dst_png.exists() and not overwrite:
            return True
        _ensure_dir(dst_png.parent)
        if img_kind == "path":
            src = Path(str(img_value))
            if src.exists():
                if src.suffix.lower() == ".png":
                    shutil.copyfile(src, dst_png)
                else:
                    im = Image.open(src)
                    im.save(dst_png, format="PNG")
                return True
            return False
        elif img_kind == "pil":
            im: Image.Image = img_value
            im.save(dst_png, format="PNG")
            return True
        elif img_kind == "bytes":
            im = Image.open(BytesIO(img_value))
            im.save(dst_png, format="PNG")
            return True
        else:
            return False
    except Exception:
        return False


def _read_text_file(path: Path) -> Optional[str]:
    try:
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return None
    return None

def _extract_code(sample: Dict[str, Any], input_root: Path, code_field: str) -> Optional[str]:
    """Extract code text from sample. If value looks like a path (.raw/.html/.txt), try reading it."""
    candidates = [code_field, "raw", "raw_html", "code", "html", "svg", "text"]
    for key in candidates:
        if key not in sample:
            continue
        v = sample.get(key)
        if isinstance(v, str):
            suffix = Path(v).suffix.lower()
            if suffix in {".raw", ".html", ".htm", ".json", ".txt"}:
                p = Path(v)
                if not p.is_absolute():
                    # Try join with input_root/{train|test|""}
                    for sub in ("train", "test", ""):
                        pp = (input_root / sub / v).resolve()
                        txt = _read_text_file(pp)
                        if txt is not None:
                            return txt
                txt = _read_text_file(p)
                if txt is not None:
                    return txt
            else:
                return v
        elif isinstance(v, dict):
            p = v.get("path")
            if isinstance(p, str) and p:
                txt = _read_text_file(Path(p))
                if txt is not None:
                    return txt
            b = v.get("bytes")
            if isinstance(b, (bytes, bytearray)):
                try:
                    return b.decode("utf-8", errors="ignore")
                except Exception:
                    pass
    return None


def _extract_image(entry: Any) -> Tuple[str, Any]:
    return _extract_image_ref(entry)


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


def _load_split_with_hf(path: Path) -> Optional[Any]:
    try:
        return load_from_disk(str(path))
    except Exception as e:
        print(f"[WARN] 加载 {path.name} 失败: {e}")
        return None


def process_split(
    split_name: str,
    ds: Any,
    input_root: Path,
    out_root: Path,
    limit: Optional[int],
    code_field: str,
    prompt: str,
    overwrite_images: bool,
    quiet: bool,
    images_root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    if ds is None:
        return [], 0

    total = len(ds)
    n = min(limit or total, total)

    # Always save images under top-level MMCoIR-train/test images directory
    # Do not nest under dataset-specific JSONL output root
    base_root = images_root if images_root is not None else out_root
    images_dir = base_root / "images" / DATASET_NAME / "images"
    _ensure_dir(images_dir)

    jsonl_items: List[Dict[str, Any]] = []
    saved = 0

    for idx in range(n):
        rec = ds[idx]
        code = _extract_code(rec, input_root.parent if input_root.name in {"train", "test"} else input_root, code_field)
        if not isinstance(code, str) or not code.strip():
            if not quiet:
                print(f"[WARN] skip idx={idx} due to missing code")
            continue

        img_entry = rec.get("image") or rec.get("img") or rec.get("screenshot") or rec.get("rendered_image")
        kind, val = _extract_image(img_entry)
        if kind == "unknown":
            if not quiet:
                print(f"[WARN] skip idx={idx} due to unknown image entry")
            continue

        sample_id = f"websight_{split_name}_{idx:06d}"
        rel_img = _posix_rel(sample_id)
        out_img_path = images_dir / f"{sample_id}.png"

        if not _save_image(kind, val, out_img_path, overwrite=overwrite_images):
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
        description="Convert WebSight (HF saved Arrow) to c2i JSONL format."
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=str(Path("datasets") / "WebSight"),
        help="Root directory containing train|test splits saved by prepare_websight_subset.py",
    )
    p.add_argument(
        "--input-train",
        type=str,
        default=None,
        help="Optional: explicit train split directory (overrides --input-dir/train)",
    )
    p.add_argument(
        "--input-test",
        type=str,
        default=None,
        help="Optional: explicit test split directory (overrides --input-dir/test)",
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

    input_dir = Path(args.input_dir)
    train_dir = Path(args.input_train) if args.input_train else (input_dir / "train")
    test_dir = Path(args.input_test) if args.input_test else (input_dir / "test")

    # Load splits via HuggingFace datasets
    train_ds = _load_split_with_hf(train_dir)
    test_ds = _load_split_with_hf(test_dir)

    # Process train
    train_items, train_saved = process_split(
        split_name="train",
        ds=train_ds,
        input_root=train_dir,
        out_root=train_root,
        limit=args.limit_train,
        code_field=args.code_field,
        prompt=args.prompt,
        overwrite_images=args.overwrite_images,
        quiet=args.quiet,
        images_root=REPO_ROOT / "MMCoIR-train",
    )
    write_jsonl(train_items, train_root / args.out_train)

    # Process test
    test_items, test_saved = process_split(
        split_name="test",
        ds=test_ds,
        input_root=test_dir,
        out_root=test_root,
        limit=args.limit_test,
        code_field=args.code_field,
        prompt=args.prompt,
        overwrite_images=args.overwrite_images,
        quiet=args.quiet,
        images_root=REPO_ROOT / "MMCoIR-test",
    )
    write_jsonl(test_items, test_root / args.out_test)

    if not args.quiet:
        print(
            f"Done. Train: {train_saved} items -> {train_root / args.out_train}\n"
            f"      Test: {test_saved} items -> {test_root / args.out_test}"
        )


if __name__ == "__main__":
    main()