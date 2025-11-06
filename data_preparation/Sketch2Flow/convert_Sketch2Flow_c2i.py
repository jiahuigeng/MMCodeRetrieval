import argparse
import json
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from PIL import Image


"""
Convert ServiceNow/BigDocs-Sketch2Flow into code-to-image (c2i) JSONL.

Project-required fields (format_checker.py):
- Train: {"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- Test:  {"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

Design for c2i:
- Query uses the target code from `annotations` with a short instruction:
  "Please convert this code to image." followed by the code.
  Example: "Please convert this code to image.\n<code_here>"
- There is no query image; `qry_image_path` / `qry_img_path` are empty.
- Target is the image; `pos_image_path` / `tgt_img_path` point to copied PNG files.
- `pos_text` / `tgt_text` contain a single image token ("<|image_1|>").

Sampling defaults (with aliases):
- --train-samples == --train-count (default 20,000 from train+val)
- --test-samples  == --test-count  (default 2,000 from test)
"""


IMG_TOKEN = "<|image_1|>"
PROMPT = "Please convert this code to image."


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _code_from_annotations(ann: Any) -> Optional[str]:
    # Prefer first string; else JSON-serialize the first element or dict.
    if isinstance(ann, list) and ann:
        first = ann[0]
        if isinstance(first, str):
            s = first.strip()
            return s if s else None
        try:
            return json.dumps(first, ensure_ascii=False)
        except Exception:
            return None
    if isinstance(ann, str):
        s = ann.strip()
        return s if s else None
    if isinstance(ann, dict):
        try:
            return json.dumps(ann, ensure_ascii=False)
        except Exception:
            return None
    return None


def _extract_image_ref(img_entry: Any) -> Tuple[str, Any]:
    # Return (kind, value) where kind in {"path", "pil", "bytes", "unknown"}
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


def load_sketch2flow_parquet(data_root: Path) -> Dict[str, Any]:
    data_files = {
        "train": str(data_root / "train-*.parquet"),
        "validation": str(data_root / "val-*.parquet"),
        "test": str(data_root / "test-*.parquet"),
    }
    return load_dataset("parquet", data_files=data_files)


def collect_items(dset_split) -> List[Tuple[str, str, Any]]:
    # (identifier, code_text, image_entry)
    items: List[Tuple[str, str, Any]] = []
    for rec in dset_split:
        ident = rec.get("identifier")
        if not isinstance(ident, str) or not ident.strip():
            continue
        code = _code_from_annotations(rec.get("annotations"))
        if not code:
            continue
        imgs = rec.get("images")
        if not isinstance(imgs, list) or not imgs:
            continue
        items.append((ident.strip(), code, imgs[0]))
    return items


def sample_items(items: List[Any], k: int, seed: int) -> List[Any]:
    if k <= 0 or k >= len(items):
        return items
    rnd = random.Random(seed)
    return rnd.sample(items, k)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_train_rows(items: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ident, code, rel_img_path in items:
        qry = f"{PROMPT}\n{code}"
        rows.append(
            {
                "qry": qry,
                "qry_image_path": "",
                "pos_text": IMG_TOKEN,
                "pos_image_path": rel_img_path,
                "neg_text": "",
                "neg_image_path": "",
            }
        )
    return rows


def build_test_rows(items: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ident, code, rel_img_path in items:
        qry = f"{PROMPT}\n{code}"
        rows.append(
            {
                "qry_text": qry,
                "qry_img_path": "",
                "tgt_text": [IMG_TOKEN],
                "tgt_img_path": [rel_img_path],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert Sketch2Flow to c2i JSONL and copy images.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).parent.parent / "datasets" / "Sketch2Flow" / "data"),
        help="Path to local Sketch2Flow data directory containing parquet files.",
    )
    parser.add_argument(
        "--train-root",
        type=str,
        default=str(Path(__file__).parent.parent / "MMCoIR-train"),
        help="Root directory for training output (JSONL and images).",
    )
    parser.add_argument(
        "--test-root",
        type=str,
        default=str(Path(__file__).parent.parent / "MMCoIR-test"),
        help="Root directory for testing output (JSONL and images).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="Sketch2Flow_c2i",
        help="Subdirectory name for dataset under train/test roots.",
    )
    parser.add_argument(
        "--train-samples",
        "--train-count",
        type=int,
        default=20000,
        dest="train_samples",
        help="Samples from train+val combined. Alias: --train-count",
    )
    parser.add_argument(
        "--test-samples",
        "--test-count",
        type=int,
        default=2000,
        dest="test_samples",
        help="Samples from test split. Alias: --test-count",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite existing copied images if present.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    out_name = args.out_name

    # Load splits
    dsets = load_sketch2flow_parquet(data_root)
    train_items = collect_items(dsets["train"]) if "train" in dsets else []
    val_items = collect_items(dsets.get("validation", []))
    test_items = collect_items(dsets["test"]) if "test" in dsets else []

    trainval_items = train_items + val_items

    sampled_trainval = sample_items(trainval_items, args.train_samples, args.seed)
    sampled_test = sample_items(test_items, args.test_samples, args.seed)

    train_images_dir = train_root / "images" / "Sketch2Flow" / "images"
    test_images_dir = test_root / "images" / "Sketch2Flow" / "images"
    _ensure_dir(train_images_dir)
    _ensure_dir(test_images_dir)

    def process_split(items: List[Tuple[str, str, Any]], img_dst_dir: Path) -> List[Tuple[str, str, str]]:
        result: List[Tuple[str, str, str]] = []
        for ident, code, img_entry in items:
            kind, val = _extract_image_ref(img_entry)
            dst_png = img_dst_dir / f"{ident}.png"
            ok = _save_image(kind, val, dst_png, overwrite=args.overwrite_images)
            if not ok:
                continue
            rel_path = str(Path("images") / "Sketch2Flow" / "images" / f"{ident}.png")
            result.append((ident, code, rel_path))
        return result

    train_rows_input = process_split(sampled_trainval, train_images_dir)
    test_rows_input = process_split(sampled_test, test_images_dir)

    train_jsonl = train_root / out_name / "train.jsonl"
    test_jsonl = test_root / out_name / "test.jsonl"
    write_jsonl(train_jsonl, build_train_rows(train_rows_input))
    write_jsonl(test_jsonl, build_test_rows(test_rows_input))

    print(f"Wrote train JSONL to: {train_jsonl} (rows={len(train_rows_input)})")
    print(f"Wrote test JSONL to:  {test_jsonl} (rows={len(test_rows_input)})")


if __name__ == "__main__":
    main()