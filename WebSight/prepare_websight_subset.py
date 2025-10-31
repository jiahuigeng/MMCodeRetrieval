#!/usr/bin/env python3
"""
Sample HuggingFaceM4/WebSight (subset v0.1, split=train) into local datasets/WebSight.

- Samples 100,000 for train and 2,000 for test (non-overlapping by default).
- Saves to disk using datasets' native save_to_disk for robust reuse:
  - datasets/WebSight/train
  - datasets/WebSight/test

Notes
- Default: materialize via `load_dataset` and shuffle/select (may read many shards).
- Optional streaming mode: only iterate needed samples without shuffling to avoid
  downloading/reading the full split. This minimizes data transfer, but selection
  is sequential (not random). Use a fixed `seed` only in non-streaming mode.
- Adjusts counts automatically if source split is smaller.
- Does not copy images; image features remain in the saved dataset.

Usage
  python WebSight/prepare_websight_subset.py

Advanced
  python WebSight/prepare_websight_subset.py \
    --subset v0.1 --split train --train-count 100000 --test-count 2000 \
    --output-dir datasets/WebSight --seed 42

Streaming (avoid full download, sequential sampling)
  python WebSight/prepare_websight_subset.py \
    --subset v0.1 --split train --train-count 100000 --test-count 2000 \
    --output-dir datasets/WebSight --streaming
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

from datasets import load_dataset, Dataset
from datasets.iterable_dataset import IterableDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample WebSight v0.1 train split to datasets/WebSight")
    parser.add_argument("--subset", type=str, default="v0.1", help="Dataset subset/config name (default v0.1)")
    parser.add_argument("--split", type=str, default="train", help="Source split to sample from (default train)")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of samples for local train")
    parser.add_argument("--test-count", type=int, default=2_000, help="Number of samples for local test")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed before splitting")
    parser.add_argument("--output-dir", type=str, default="datasets/WebSight", help="Target output directory")
    parser.add_argument("--repo-id", type=str, default="HuggingFaceM4/WebSight", help="HF dataset repo id")
    parser.add_argument("--token", type=str, default=None, help="HF access token; fallback to env (HUGGINGFACE_TOKEN/HF_TOKEN)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode to avoid full split download (sequential sampling, no shuffle)")
    return parser.parse_args()


def adjust_counts(total: int, train_cnt: int, test_cnt: int) -> Tuple[int, int]:
    train_cnt = min(train_cnt, total)
    remaining = max(total - train_cnt, 0)
    test_cnt = min(test_cnt, remaining)
    return train_cnt, test_cnt


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    print(f"Repo:      {args.repo_id}")
    print(f"Subset:    {args.subset}")
    print(f"Split:     {args.split}")
    print(f"Seed:      {args.seed}")
    print(f"Counts:    train={args.train_count}, test={args.test_count}")
    print(f"Output:    {out_dir}")
    print(f"Auth:      {'provided' if token else 'none'}")
    print(f"Streaming: {args.streaming}")

    # Load source split
    if args.streaming:
        ds_stream: IterableDataset = load_dataset(
            args.repo_id,
            args.subset,
            split=args.split,
            token=token,
            streaming=True,
        )
        # Sequentially take train_count, then test_count
        train_cnt, test_cnt = args.train_count, args.test_count
        train_buffer: List[Dict[str, Any]] = []
        test_buffer: List[Dict[str, Any]] = []

        print("Streaming and taking sequential samples without shuffle ...")
        for idx, ex in enumerate(ds_stream):
            if idx < train_cnt:
                train_buffer.append(ex)
            elif idx < train_cnt + test_cnt:
                test_buffer.append(ex)
            else:
                break

        print(f"Streamed: train_buffer={len(train_buffer)}, test_buffer={len(test_buffer)}")
        if len(train_buffer) < train_cnt:
            print(f"[WARN] Source smaller than requested train_count; using {len(train_buffer)}")
        if len(test_buffer) < test_cnt:
            print(f"[WARN] Source smaller than requested test_count; using {len(test_buffer)}")

        # Materialize to Arrow Datasets
        train_ds = Dataset.from_list(train_buffer)
        test_ds = Dataset.from_list(test_buffer) if len(test_buffer) > 0 else None
    else:
        ds: Dataset = load_dataset(
            args.repo_id,
            args.subset,
            split=args.split,
            token=token,
        )
        print(f"Loaded source split: size={len(ds)}")

        # Shuffle then split (non-streaming only)
        ds_shuffled = ds.shuffle(seed=args.seed)
        train_cnt, test_cnt = adjust_counts(len(ds_shuffled), args.train_count, args.test_count)
        print(f"Adjusted counts: train={train_cnt}, test={test_cnt}")

        train_ds = ds_shuffled.select(range(train_cnt))
        test_start = train_cnt
        test_ds = ds_shuffled.select(range(test_start, test_start + test_cnt)) if test_cnt > 0 else None

    # Save to disk (Arrow dataset)
    train_out = out_dir / "train"
    print(f"Saving train -> {train_out}")
    train_ds.save_to_disk(str(train_out))

    if test_ds is not None:
        test_out = out_dir / "test"
        print(f"Saving test  -> {test_out}")
        test_ds.save_to_disk(str(test_out))
    else:
        print("[WARN] Test count is 0; skipping test save.")

    # Basic summary
    cols = list(train_ds.column_names)
    print("\nSummary:")
    print(f"Columns: {cols}")
    print(f"Train size: {len(train_ds)}")
    print(f"Test  size: {len(test_ds) if test_ds is not None else 0}")
    print("Done.")


if __name__ == "__main__":
    main()