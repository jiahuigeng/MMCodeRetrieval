#!/usr/bin/env python3
"""
Convert SVGs to PNGs for locally downloaded OmniSVG MMSVG datasets.
- Scans `datasets/<name>/svgs/*.svg` and renders to `datasets/<name>/imgs/*.png`.
- Keeps the same base filename (only extension changes to .png).
- Supports backend selection: cairosvg (Python), inkscape (CLI), magick (ImageMagick CLI).
- Prints a short summary similar to inspect_omnisvg.

Usage examples:
  # 默认转换所有文件（推荐）
  python convert_omnisvg.py --root datasets --dataset_names MMSVG-Illustration MMSVG-Icon
  
  # 仅转换少量文件做快速验证
  python convert_omnisvg.py --backend inkscape --max_count 100
  
  # 转换所有 SVG 并覆盖已有 PNG
  python convert_omnisvg.py --backend inkscape --overwrite
  
  # 使用 ImageMagick 后端
  python convert_omnisvg.py --backend magick

Notes:
- For `cairosvg` backend, you need Cairo library available on your system.
- For `inkscape` backend, ensure Inkscape is installed and on PATH.
- For `magick` backend, ensure ImageMagick is installed and `magick` is on PATH.
"""

import argparse
import json
import os
import re
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def summarize_dataset(root: Path, name: str, split: str = "train"):
    ds_dir = root / name
    print(f"\n== Dataset: {name} ==")
    print(f"Path: {ds_dir}")
    if not ds_dir.exists():
        print("Status: NOT FOUND")
        return

    svgs_dir = ds_dir / "svgs"
    imgs_dir = ds_dir / "imgs"
    print(f"svgs/: {'exists' if svgs_dir.exists() else 'missing'}")
    print(f"imgs/: {'exists' if imgs_dir.exists() else 'missing'}")
    if svgs_dir.exists():
        svg_files = list(svgs_dir.glob("*.svg"))
        print(f"svg count: {len(svg_files)}")
        for p in svg_files[:3]:
            print(f" - {p.name}")

    jsonl_path = ds_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None
        print(f"JSONL: {jsonl_path}")
        print(f"records: {total if total is not None else 'unknown'}")
    else:
        print("JSONL: missing")


def find_backend(preferred: str = "auto") -> Tuple[str, Optional[str]]:
    """Return (backend, path_or_none)."""
    preferred = preferred.lower()
    if preferred == "cairosvg":
        return "cairosvg", None
    if preferred == "inkscape":
        return "inkscape", shutil.which("inkscape")
    if preferred == "magick":
        return "magick", shutil.which("magick")

    # auto: try cairosvg import, else inkscape, else magick
    try:
        import cairosvg  # noqa: F401
        return "cairosvg", None
    except Exception:
        pass
    inkscape = shutil.which("inkscape")
    if inkscape:
        return "inkscape", inkscape
    magick = shutil.which("magick")
    if magick:
        return "magick", magick
    return "none", None


def convert_with_cairosvg(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg
        # Use bytestring to avoid file path encoding issues
        svg_bytes = svg_path.read_text(encoding="utf-8", errors="ignore").encode("utf-8")
        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(png_path))
        return True
    except Exception as e:
        eprint(f"cairosvg failed for {svg_path}: {e}")
        return False


def detect_inkscape_cli() -> str:
    """Return the correct CLI option to export PNG for the installed Inkscape."""
    exe = shutil.which("inkscape")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True)
        ver = out.stdout.lower()
        if "inkscape" in ver:
            # Inkscape >= 1.0 uses --export-type=png and --export-filename
            return "modern"
    except Exception:
        pass
    return "legacy"


def convert_with_inkscape(svg_path: Path, png_path: Path, exe_path: Optional[str] = None) -> bool:
    exe = exe_path or shutil.which("inkscape")
    if not exe:
        eprint("inkscape not found on PATH")
        return False
    mode = detect_inkscape_cli()
    try:
        if mode == "modern":
            cmd = [exe, "--export-type=png", f"--export-filename={png_path}", str(svg_path)]
        else:
            # Fallback old CLI (rare on modern systems)
            cmd = [exe, f"--export-png={png_path}", str(svg_path)]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            eprint(f"inkscape conversion failed: {res.stderr.decode(errors='ignore')}")
            return False
        return True
    except Exception as e:
        eprint(f"inkscape failed for {svg_path}: {e}")
        return False


def convert_with_magick(svg_path: Path, png_path: Path, exe_path: Optional[str] = None) -> bool:
    exe = exe_path or shutil.which("magick")
    if not exe:
        eprint("ImageMagick 'magick' not found on PATH")
        return False
    try:
        cmd = [exe, str(svg_path), str(png_path)]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            eprint(f"magick conversion failed: {res.stderr.decode(errors='ignore')}")
            return False
        return True
    except Exception as e:
        eprint(f"magick failed for {svg_path}: {e}")
        return False


def convert_svg_to_png(svg_path: Path, png_path: Path, backend: str = "auto", exe_path: Optional[str] = None) -> bool:
    if backend == "cairosvg":
        return convert_with_cairosvg(svg_path, png_path)
    if backend == "inkscape":
        return convert_with_inkscape(svg_path, png_path, exe_path)
    if backend == "magick":
        return convert_with_magick(svg_path, png_path, exe_path)
    # auto
    bk, path = find_backend("auto")
    if bk == "cairosvg":
        ok = convert_with_cairosvg(svg_path, png_path)
        if ok:
            return True
        # fallback
        bk = "inkscape"
        path = shutil.which("inkscape")
        if path:
            ok = convert_with_inkscape(svg_path, png_path, path)
            if ok:
                return True
        bk = "magick"
        path = shutil.which("magick")
        if path:
            ok = convert_with_magick(svg_path, png_path, path)
            if ok:
                return True
        return False
    elif bk == "inkscape":
        return convert_with_inkscape(svg_path, png_path, path)
    elif bk == "magick":
        return convert_with_magick(svg_path, png_path, path)
    else:
        eprint("No available backend found (cairosvg/inkscape/magick). Please install one.")
        return False


def process_dataset(root: Path, name: str, max_count: Optional[int], overwrite: bool, backend: str):
    ds_dir = root / name
    svgs_dir = ds_dir / "svgs"
    imgs_dir = ds_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    if not svgs_dir.exists():
        print(f"[Skip] {name} svgs/ not found: {svgs_dir}")
        return

    bk, bk_path = find_backend(backend)
    print(f"Backend: {bk}{' (' + bk_path + ')' if bk_path else ''}")

    svg_files = sorted(svgs_dir.glob("*.svg"))
    total = len(svg_files)
    print(f"Found {total} SVG files in {name}.")

    if bk == "none":
        print("No available backend found (cairosvg/inkscape/magick). Skipping conversion.")
        print("Install one of: pip install cairosvg (requires Cairo DLLs on Windows), Inkscape (add to PATH), or ImageMagick 'magick'.")
        return

    converted = 0
    failed = 0
    for i, svg_path in enumerate(svg_files, start=1):
        if max_count is not None and converted >= max_count:
            break
        png_path = imgs_dir / (svg_path.stem + ".png")
        if png_path.exists() and not overwrite:
            continue
        ok = convert_svg_to_png(svg_path, png_path, backend=bk, exe_path=bk_path)
        if ok:
            converted += 1
            if converted <= 3:
                print(f" + {svg_path.name} -> {png_path.name}")
        else:
            failed += 1
            if failed <= 3:
                print(f" - Failed: {svg_path.name}")
    print(f"Done {name}: converted={converted}, failed={failed}, skipped={total - converted - failed}")


def main():
    parser = argparse.ArgumentParser(description="Convert OmniSVG MMSVG svgs to imgs (png)")
    parser.add_argument("--root", type=str, default="datasets", help="Root directory containing datasets.")
    parser.add_argument("--dataset_names", nargs="+", default=["MMSVG-Illustration", "MMSVG-Icon"], help="Dataset subfolders under root.")
    parser.add_argument("--max_count", type=int, default=None, help="Max number of conversions per dataset (omit to convert all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cairosvg", "inkscape", "magick"], help="Conversion backend.")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Root: {root} {'(exists)' if root.exists() else '(missing)'}")
    for name in args.dataset_names:
        summarize_dataset(root, name)
        process_dataset(root, name, max_count=(args.max_count if (args.max_count is not None and args.max_count > 0) else None), overwrite=args.overwrite, backend=args.backend)


if __name__ == "__main__":
    main()