#!/usr/bin/env python3
"""
Inspect locally downloaded OmniSVG MMSVG datasets and show a few samples.
- Expects data downloaded by `download_omnisvg.py` under `datasets/`.
- Summarizes folder structure, JSONL presence, and prints a few sample entries.

Usage:
  python inspect_omnisvg.py --root datasets --dataset_names MMSVG-Illustration MMSVG-Icon --samples 3
"""

import argparse
import json
import os
from pathlib import Path
import re
from typing import List, Optional


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(name)).strip("._") or "item"


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except Exception as e:
                    rec = {"_error": str(e), "_raw": line}
                records.append(rec)
                if limit is not None and len(records) >= limit:
                    break
    except FileNotFoundError:
        pass
    return records


def count_lines(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def choose_jsonl(ds_dir: Path, preferred_name: str) -> Optional[Path]:
    preferred = ds_dir / preferred_name
    if preferred.exists():
        return preferred
    for cand in ds_dir.glob("*.jsonl"):
        return cand
    return None


def print_dataset_summary(root: Path, name: str, samples: int = 3, split: str = "train"):
    ds_dir = root / name
    print(f"\n== Dataset: {name} ==")
    print(f"Path: {ds_dir}")
    if not ds_dir.exists():
        print("Status: NOT FOUND")
        return

    svgs_dir = ds_dir / "svgs"
    print(f"svgs/: {'exists' if svgs_dir.exists() else 'missing'}")
    if svgs_dir.exists():
        svg_files = list(svgs_dir.glob("*.svg"))
        print(f"svg count: {len(svg_files)}")
        for p in svg_files[:3]:
            print(f" - {p.name}")

    jsonl_path = choose_jsonl(ds_dir, f"{split}.jsonl")
    print(f"JSONL: {jsonl_path if jsonl_path else 'missing'}")
    if not jsonl_path:
        return

    total = count_lines(jsonl_path)
    print(f"records: {total if total is not None else 'unknown'}")

    recs = read_jsonl(jsonl_path, limit=samples)
    print(f"\nSamples (showing {len(recs)}):")
    for rec in recs:
        rid = str(rec.get("id", "N/A"))
        descr = rec.get("description", "")
        if isinstance(descr, str):
            descr_short = descr.strip().replace("\n", " ")[:120]
        else:
            descr_short = "N/A"
        svg = rec.get("svg", "")
        if isinstance(svg, str):
            svg_snip = svg.strip().replace("\n", " ")
            svg_short = svg_snip[:120] + ("..." if len(svg_snip) > 120 else "")
        else:
            svg_short = "N/A"
        svg_file = None
        if svgs_dir.exists():
            svg_file = svgs_dir / f"{sanitize_filename(rid)}.svg"

        print(f" - id: {rid}")
        print(f"   description: {descr_short}")
        print(f"   svg_snippet: {svg_short}")
        print(f"   svg_file: {svg_file if svg_file and svg_file.exists() else 'N/A'}")
        extra_keys = [k for k in rec.keys() if k not in ("id", "svg", "description", "dataset")]
        if extra_keys:
            print(f"   extra_keys: {', '.join(extra_keys[:8])}")


def main():
    parser = argparse.ArgumentParser(description="Inspect OmniSVG MMSVG datasets in a local folder.")
    parser.add_argument("--root", type=str, default="MMCoIR", help="Root directory containing downloaded datasets.")
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["MMSVG-Illustration", "MMSVG-Icon"],
        help="Subfolder names under --root (defaults match download_omnisvg.py).",
    )
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to print per dataset.")
    parser.add_argument("--split", type=str, default="train", help="Expected JSONL split name (default: train).")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Root: {root} {'(exists)' if root.exists() else '(missing)'}")
    for name in args.dataset_names:
        print_dataset_summary(root, name, samples=args.samples, split=args.split)


if __name__ == "__main__":
    main()