#!/usr/bin/env python3
"""
Simple conversion: Read a parquet and write selected columns to JSONL.
Default: converts useless/train_svg.parquet to useless/train_svg.jsonl
"""

import os
import argparse
import json
import pandas as pd


def to_serializable(record):
    # Ensure all fields are JSON serializable (e.g., numpy types -> python types)
    out = {}
    for k, v in record.items():
        if hasattr(v, "item"):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        out[k] = v
    return out


def convert_parquet_to_jsonl(parquet_path: str, output_path: str, include_columns=None):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if include_columns is None:
        include_columns = list(df.columns)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {col: row[col] for col in include_columns if col in df.columns}
            f.write(json.dumps(to_serializable(rec), ensure_ascii=False) + "\n")

    print(f"Converted {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a parquet file to JSONL")
    parser.add_argument("--input", default=os.path.join("useless", "train_svg.parquet"), help="Input parquet path")
    parser.add_argument("--output", default=os.path.join("useless", "train_svg.jsonl"), help="Output JSONL path")
    parser.add_argument("--columns", nargs="*", default=["id", "filename", "difficulty", "svg_code"], help="Columns to include")
    args = parser.parse_args()

    convert_parquet_to_jsonl(args.input, args.output, include_columns=args.columns)


if __name__ == "__main__":
    main()