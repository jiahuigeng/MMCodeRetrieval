#!/usr/bin/env python3
"""
Inspect a parquet file in the 'useless' directory:
- Print basic metadata (rows, row groups)
- Print column names (keys)
- Try to display sample unique values for file/path-related columns
- Optionally list files present under the 'useless' directory

Usage:
  python inspect_useless_parquet.py --parquet_path useless/test-00000-of-00001.parquet --useless_dir useless --list_dir
"""

import argparse
import os
from pathlib import Path

# Optional dependencies
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

try:
    import pandas as pd
except Exception:
    pd = None


def find_parquet_files(directory: str):
    """Return a list of parquet file paths under the given directory."""
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.parquet'):
                results.append(os.path.join(root, f))
    return sorted(results)


def get_parquet_columns(parquet_path: str):
    """Return column names and basic metadata using pyarrow if available."""
    columns = []
    meta = {}
    if pq is not None:
        try:
            pf = pq.ParquetFile(parquet_path)
            columns = list(pf.schema.names)
            md = pf.metadata
            meta = {
                'num_rows': md.num_rows if md is not None else None,
                'num_row_groups': md.num_row_groups if md is not None else None,
            }
            return columns, meta
        except Exception as e:
            print(f"[WARN] pyarrow failed to read schema: {e}")
    # Fallback: use pandas to read a tiny sample
    if pd is not None:
        try:
            df = pd.read_parquet(parquet_path)
            columns = list(df.columns)
            meta = {
                'num_rows': len(df),
                'num_row_groups': None,
            }
            return columns, meta
        except Exception as e:
            print(f"[ERROR] pandas failed to read parquet: {e}")
    return columns, meta


def sample_file_columns(parquet_path: str, columns, limit: int = 50):
    """Sample unique values for columns likely containing file paths."""
    if pd is None:
        print("[WARN] pandas not available; skipping file column sampling.")
        return {}
    # Heuristics for file/path columns
    candidates = [
        c for c in columns if any(k in c.lower() for k in ['path', 'file', 'image', 'video', 'img'])
    ]
    samples = {}
    if not candidates:
        return samples
    try:
        df = pd.read_parquet(parquet_path, columns=candidates)
    except Exception as e:
        print(f"[WARN] pandas read_parquet failed for selected columns: {e}")
        return samples
    for c in candidates:
        try:
            # Dropna, cast to str, get unique
            vals = df[c].dropna().astype(str).unique().tolist()
            samples[c] = vals[:limit]
        except Exception as e:
            samples[c] = [f"[ERROR extracting values: {e}"]
    return samples


def list_useless_files(useless_dir: str, max_items: int = 200):
    """List files under useless_dir (relative paths)."""
    rels = []
    base = Path(useless_dir).resolve()
    for root, dirs, files in os.walk(base):
        for f in files:
            rel_path = Path(root).joinpath(f).resolve().relative_to(base)
            rels.append(str(rel_path))
            if len(rels) >= max_items:
                return rels
    return rels


def main():
    parser = argparse.ArgumentParser(description='Inspect parquet in useless directory')
    parser.add_argument('--parquet_path', type=str, default='useless/test-00000-of-00001.parquet',
                        help='Path to the parquet file to inspect')
    parser.add_argument('--useless_dir', type=str, default='useless',
                        help='Directory to list files from')
    parser.add_argument('--list_dir', action='store_true',
                        help='If set, list files under useless_dir')
    parser.add_argument('--max_list', type=int, default=200,
                        help='Max files to list from useless_dir')
    parser.add_argument('--sample_limit', type=int, default=50,
                        help='Max unique samples to show per file/path column')
    args = parser.parse_args()

    # Resolve parquet path if not exists
    parquet_path = args.parquet_path
    if not os.path.exists(parquet_path):
        print(f"[INFO] Provided parquet not found: {parquet_path}")
        print(f"[INFO] Searching for parquet files under: {args.useless_dir}")
        found = find_parquet_files(args.useless_dir)
        if found:
            parquet_path = found[0]
            print(f"[INFO] Using detected parquet: {parquet_path}")
        else:
            print("[ERROR] No parquet files found under useless directory.")
            return

    print(f"\n=== Parquet Metadata & Columns ===")
    print(f"Parquet: {parquet_path}")
    cols, meta = get_parquet_columns(parquet_path)
    if meta:
        print(f"Rows: {meta.get('num_rows')} | Row Groups: {meta.get('num_row_groups')}")
    print(f"Columns ({len(cols)}): {cols}")

    print(f"\n=== File/Path Columns: Sample Values ===")
    samples = sample_file_columns(parquet_path, cols, limit=args.sample_limit)
    if not samples:
        print("[INFO] No obvious file/path columns detected or sampling failed.")
    else:
        for c, vals in samples.items():
            print(f"- {c} (showing up to {args.sample_limit} uniques):")
            for v in vals:
                print(f"  {v}")

    if args.list_dir:
        print(f"\n=== Files under '{args.useless_dir}' (up to {args.max_list}) ===")
        items = list_useless_files(args.useless_dir, max_items=args.max_list)
        for itm in items:
            print(itm)
        print(f"Total listed: {len(items)}")


if __name__ == '__main__':
    main()