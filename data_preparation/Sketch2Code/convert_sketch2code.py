#!/usr/bin/env python3
"""
Convert Sketch2Code raw data into image-to-code (i2c) test JSONL.

Requirements:
- Input images under `datasets/Sketch2Code/sketches/`.
- Corresponding code under `datasets/Sketch2Code/webpages/<id>.html` (with optional fallback to folder `webpages/<id>/`).
- Output only test set:
  - JSONL -> `MMCoIR-test/Sketch2Code_i2c/test.jsonl`
  - Copy images -> `MMCoIR-test/images/Sketch2Code/images/<id>.png`
- JSONL fields (test): {"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}
- Image path in JSONL must start with `images/`.
- qry_text is image token + "Please convert this image to code.".
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


# ---------- Constants ----------
IMG_TOKEN = "<|image_1|>"
INSTR = "Please convert this image to code."

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKETCHES_DIR = REPO_ROOT / "datasets" / "Sketch2Code" / "sketches"
DEFAULT_WEBPAGES_DIR = REPO_ROOT / "datasets" / "Sketch2Code" / "webpages"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
OUT_NAME_DEFAULT = "Sketch2Code_i2c"
IMAGES_DST_BUCKET = DEFAULT_TEST_ROOT / "images" / "Sketch2Code" / "images"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXTS = {".html", ".htm", ".css", ".js", ".json", ".txt"}
MAX_TEXT_FILE_BYTES = 1024 * 1024  # 1 MiB safety cap per file


# ---------- Helpers ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def discover_images(sketches_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in IMAGE_EXTS:
        imgs.extend(sketches_dir.rglob(f"*{ext}"))
    return imgs


def read_text_file(fp: Path) -> Optional[str]:
    try:
        if fp.stat().st_size > MAX_TEXT_FILE_BYTES:
            return None
        with fp.open("r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with fp.open("r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def build_code_string(page_dir: Path) -> Optional[str]:
    if not page_dir.is_dir():
        return None
    mapping: Dict[str, str] = {}
    for fp in page_dir.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in TEXT_EXTS:
            continue
        txt = read_text_file(fp)
        if txt is None:
            continue
        rel_key = str(fp.relative_to(page_dir)).replace("\\", "/")
        mapping[rel_key] = txt
    if not mapping:
        return None
    try:
        return json.dumps(mapping, ensure_ascii=False)
    except Exception:
        return None


def build_code_string_for_ident(webpages_dir: Path, ident: str) -> Optional[str]:
    """Build code string for a given ident.
    First try single HTML file `webpages/<ident>.html`.
    If not found, fallback to a folder aggregator `webpages/<ident>/`.
    """
    # Prefer single HTML file
    html_fp = webpages_dir / f"{ident}.html"
    if html_fp.exists() and html_fp.is_file():
        html = read_text_file(html_fp)
        return html
    # Fallback: folder-based aggregation
    page_dir = webpages_dir / ident
    return build_code_string(page_dir)


def save_image_as_png(src_img: Path, dst_png: Path, overwrite: bool = False) -> bool:
    try:
        if dst_png.exists() and not overwrite:
            return True
        ensure_dir(dst_png.parent)
        if src_img.suffix.lower() == ".png":
            shutil.copy2(src_img, dst_png)
            return True
        im = Image.open(src_img)
        im.save(dst_png, format="PNG")
        return True
    except Exception:
        return False


def to_test_item(rel_img_path: str, code_str: str) -> Dict:
    return {
        "qry_text": f"{IMG_TOKEN}\n{INSTR}",
        "qry_img_path": rel_img_path,
        "tgt_text": [code_str],
        "tgt_img_path": [""],
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Convert Sketch2Code to image-to-code test JSONL.")
    parser.add_argument("--sketches-dir", type=str, default=str(DEFAULT_SKETCHES_DIR), help="Directory of input sketches.")
    parser.add_argument("--webpages-dir", type=str, default=str(DEFAULT_WEBPAGES_DIR), help="Directory of corresponding webpages (<id>.html or <id>/ folder).")
    parser.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="Root for MMCoIR-test output.")
    parser.add_argument("--out-name", type=str, default=OUT_NAME_DEFAULT, help="Subdirectory name under test root for JSONL.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of test samples.")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite copied/converted images if already exist.")

    args = parser.parse_args()

    sketches_dir = Path(args.sketches_dir)
    webpages_dir = Path(args.webpages_dir)
    test_root = Path(args.test_root)
    out_name = args.out_name

    if not sketches_dir.exists():
        raise FileNotFoundError(f"Sketches directory not found: {sketches_dir}")
    if not webpages_dir.exists():
        raise FileNotFoundError(f"Webpages directory not found: {webpages_dir}")

    # Discover image files
    img_files = discover_images(sketches_dir)
    if not img_files:
        print(f"[WARN] No images found under: {sketches_dir}")

    # Prepare output directories
    images_dst = Path(args.test_root) / "images" / "Sketch2Code" / "images"
    ensure_dir(images_dst)

    test_jsonl = test_root / out_name / "test.jsonl"
    ensure_dir(test_jsonl.parent)

    rows: List[Dict] = []
    n_skipped_no_code = 0
    n_skipped_img_fail = 0

    for img_path in img_files:
        stem = img_path.stem  # e.g., "1009_0"
        ident = stem.split("_")[0]  # e.g., "1009"
        code_str = build_code_string_for_ident(webpages_dir, ident)
        if not code_str:
            n_skipped_no_code += 1
            continue
        dst_png = images_dst / f"{stem}.png"
        ok = save_image_as_png(img_path, dst_png, overwrite=args.overwrite_images)
        if not ok:
            n_skipped_img_fail += 1
            continue
        # 使用统一的正斜杠路径，兼容 Linux/Ubuntu
        rel_img = f"images/Sketch2Code/images/{stem}.png"
        rows.append(to_test_item(rel_img, code_str))
        if args.limit and len(rows) >= args.limit:
            break

    # Write JSONL
    with test_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Sketch2Code i2c test conversion completed. rows={len(rows)}")
    print(f"Test JSONL:  {test_jsonl}")
    print(f"Test images: {images_dst}")
    if n_skipped_no_code:
        print(f"[WARN] Skipped (no HTML code file or empty): {n_skipped_no_code}")
    if n_skipped_img_fail:
        print(f"[WARN] Skipped (image convert/copy fail): {n_skipped_img_fail}")


if __name__ == "__main__":
    main()