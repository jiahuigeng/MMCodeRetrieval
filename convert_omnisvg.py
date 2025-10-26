#!/usr/bin/env python3
"""
Convert SVGs to PNGs for locally downloaded OmniSVG MMSVG datasets.
- Scans `datasets/<name>/svgs/*.svg` and renders to `datasets/<name>/imgs/*.png`.
- Keeps the same base filename (only extension changes to .png).
- Supports backend selection: cairosvg (Python), inkscape (CLI), magick (ImageMagick CLI).
- Prints a short summary similar to inspect_omnisvg.

Usage examples:
  # 默认转换所有文件（推荐）
  python convert_omnisvg.py --root MMCoIR --dataset_names MMSVG-Illustration MMSVG-Icon
  
  # 仅转换少量文件做快速验证
  python convert_omnisvg.py --backend inkscape --max_count 100
  
  # 转换所有 SVG 并覆盖已有 PNG
  python convert_omnisvg.py --backend inkscape --overwrite
  
  # 使用 ImageMagick 后端
  python convert_omnisvg.py --backend magick

  # 仅统计待转换数量（不执行转换）
  python convert_omnisvg.py --root MMCoIR --dataset_names MMSVG-Icon --count_only
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
import multiprocessing
import queue


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


def convert_with_inkscape(svg_path: Path, png_path: Path, exe_path: Optional[str] = None, timeout: Optional[float] = None) -> bool:
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
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if res.returncode != 0:
            eprint(f"inkscape conversion failed: {res.stderr.decode(errors='ignore')}")
            return False
        return True
    except subprocess.TimeoutExpired:
        eprint(f"inkscape timeout ({timeout}s): {svg_path}")
        return False
    except Exception as e:
        eprint(f"inkscape failed for {svg_path}: {e}")
        return False


def convert_with_magick(svg_path: Path, png_path: Path, exe_path: Optional[str] = None, timeout: Optional[float] = None) -> bool:
    exe = exe_path or shutil.which("magick")
    if not exe:
        eprint("ImageMagick 'magick' not found on PATH")
        return False
    try:
        cmd = [exe, str(svg_path), str(png_path)]
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if res.returncode != 0:
            eprint(f"magick conversion failed: {res.stderr.decode(errors='ignore')}")
            return False
        return True
    except subprocess.TimeoutExpired:
        eprint(f"magick timeout ({timeout}s): {svg_path}")
        return False
    except Exception as e:
        eprint(f"magick failed for {svg_path}: {e}")
        return False


def _cairosvg_worker(svg_path_str: str, png_path_str: str, result_queue):
    try:
        import cairosvg
        with open(svg_path_str, 'rb') as f:
            data = f.read()
        cairosvg.svg2png(bytestring=data, write_to=png_path_str)
        result_queue.put(True)
    except Exception as e:
        try:
            result_queue.put(('error', str(e)))
        except Exception:
            pass


def convert_with_cairosvg_timeout(svg_path: Path, png_path: Path, timeout_sec: float) -> bool:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_cairosvg_worker, args=(str(svg_path), str(png_path), q))
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join()
        eprint(f"cairosvg timeout ({timeout_sec}s): {svg_path}")
        return False
    try:
        res = q.get_nowait()
    except queue.Empty:
        res = None
    if res is True:
        return True
    elif isinstance(res, tuple) and len(res) == 2 and res[0] == 'error':
        eprint(f"cairosvg failed for {svg_path}: {res[1]}")
        return False
    else:
        return False


def convert_svg_to_png(svg_path: Path, png_path: Path, backend: str, exe_path: Optional[str] = None, timeout: Optional[float] = None) -> bool:
    """Convert a single SVG to PNG using selected backend."""
    backend = backend.lower()
    if backend == 'auto':
        bk, path = find_backend('auto')
        if bk == 'cairosvg':
            return convert_with_cairosvg_timeout(svg_path, png_path, timeout) if timeout else convert_with_cairosvg(svg_path, png_path)
        elif bk == 'inkscape':
            return convert_with_inkscape(svg_path, png_path, exe_path=path, timeout=timeout)
        elif bk == 'magick':
            return convert_with_magick(svg_path, png_path, exe_path=path, timeout=timeout)
        else:
            eprint("No available backend found (cairosvg/inkscape/magick). Please install one.")
            return False
    elif backend == 'cairosvg':
        return convert_with_cairosvg_timeout(svg_path, png_path, timeout) if timeout else convert_with_cairosvg(svg_path, png_path)
    elif backend == 'inkscape':
        return convert_with_inkscape(svg_path, png_path, exe_path=exe_path, timeout=timeout)
    elif backend == 'magick':
        return convert_with_magick(svg_path, png_path, exe_path=exe_path, timeout=timeout)
    else:
        eprint(f"Unsupported backend: {backend}")
        return False


def process_dataset(root: Path, name: str, max_count: Optional[int], overwrite: bool, backend: str, print_names: bool, count_only: bool, progress: bool, timeout: Optional[float]) -> int:
    ds_dir = root / name
    svgs_dir = ds_dir / "svgs"
    imgs_dir = ds_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    if not svgs_dir.exists():
        print(f"[Skip] {name} svgs/ not found: {svgs_dir}")
        return 0

    svg_files = sorted(svgs_dir.glob("*.svg"))
    total = len(svg_files)
    print(f"Found {total} SVG files in {name}.")

    # Derive plan: set-diff when not overwriting
    if not overwrite:
        png_stems = {p.stem for p in imgs_dir.glob("*.png")}
        iter_paths = [p for p in svg_files if p.stem not in png_stems]
        missing_count = len(iter_paths)
        print(f"Missing PNGs estimated: {missing_count}")
    else:
        iter_paths = svg_files
        missing_count = len(iter_paths)
        print(f"Planned conversions with --overwrite: {missing_count}")

    if count_only:
        print(f"[CountOnly] {name}: planned_conversions={missing_count}")
        return missing_count

    # choose backend only when converting
    bk, bk_path = find_backend(backend)
    print(f"Backend: {bk}{' (' + bk_path + ')' if bk_path else ''}")

    if bk == "none":
        print("No available backend found (cairosvg/inkscape/magick). Skipping conversion.")
        print("Install one of: pip install cairosvg (requires Cairo DLLs on Windows), Inkscape (add to PATH), or ImageMagick 'magick'.")
        return 0

    converted = 0
    failed = 0
    planned_total = missing_count
    for i, svg_path in enumerate(iter_paths, start=1):
        if max_count is not None and converted >= max_count:
            break
        png_path = imgs_dir / (svg_path.stem + ".png")
        if not overwrite and png_path.exists():
            # Shouldn't happen because we prefiltered, but keep for safety.
            continue
        prefix = f"[{i}/{planned_total}] " if progress else ""
        if print_names:
            print(f"{prefix}convert: {svg_path.name} -> {png_path.name}")
        ok = convert_svg_to_png(svg_path, png_path, backend=bk, exe_path=bk_path, timeout=timeout)
        if ok:
            converted += 1
            if not print_names and converted <= 3:
                print(f" {prefix}+ {svg_path.name} -> {png_path.name}")
        else:
            failed += 1
            if not print_names and failed <= 3:
                print(f" {prefix}- Failed: {svg_path.name}")
    print(f"Done {name}: converted={converted}, failed={failed}, skipped={total - converted - failed}")
    return missing_count


def main():
    parser = argparse.ArgumentParser(description="Convert OmniSVG MMSVG svgs to imgs (png)")
    parser.add_argument("--root", type=str, default="MMCoIR", help="Root directory containing datasets.")
    parser.add_argument("--dataset_names", nargs="+", default=["MMSVG-Illustration", "MMSVG-Icon"], help="Dataset subfolders under root.")
    parser.add_argument("--max_count", type=int, default=None, help="Max number of conversions per dataset (omit to convert all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cairosvg", "inkscape", "magick"], help="Conversion backend.")
    parser.add_argument("--print_names", action="store_true", help="Print each processed filename during conversion.")
    parser.add_argument("--count_only", action="store_true", help="Only count planned conversions and skip actual conversion.")
    parser.add_argument("--progress", action="store_true", help="Print progress as [i/total] during conversion.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-file timeout seconds; skip if exceeded.")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Root: {root} {'(exists)' if root.exists() else '(missing)'}")
    total_planned = 0
    for name in args.dataset_names:
        summarize_dataset(root, name)
        planned = process_dataset(
            root,
            name,
            max_count=(args.max_count if (args.max_count is not None and args.max_count > 0) else None),
            overwrite=args.overwrite,
            backend=args.backend,
            print_names=args.print_names,
            count_only=args.count_only,
            progress=args.progress,
            timeout=args.timeout,
        )
        total_planned += planned
    if args.count_only:
        print(f"\n[Summary] Total planned conversions across datasets: {total_planned}")


if __name__ == "__main__":
    main()