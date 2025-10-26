#!/usr/bin/env python3
"""
Download the SALT-NLP/Sketch2Code dataset from Hugging Face Hub.

Features
- Snapshots the entire dataset repo to a local folder using huggingface_hub.
- Optional token support (private/gated datasets), reads env vars automatically.
- Filtering via allow/ignore patterns to limit which paths are downloaded.
- Safe on Windows by disabling symlinks (uses real files).
- Prints a summary of downloaded files and total size.

Usage
  # Basic (download full dataset)
  python download_sketch2code.py --output_dir datasets/Sketch2Code

  # Specify repo id explicitly
  python download_sketch2code.py --repo_id SALT-NLP/Sketch2Code --output_dir datasets/Sketch2Code

  # Filter only certain subsets (depends on dataset structure)
  python download_sketch2code.py --allow train/** --allow validation/** --output_dir datasets/Sketch2Code

  # Ignore large or unnecessary paths
  python download_sketch2code.py --ignore examples/** --ignore docs/** --output_dir datasets/Sketch2Code

Prerequisites
  pip install huggingface_hub
  # optionally set token if required
  export HUGGINGFACE_TOKEN=hf_********************************
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def compute_dir_stats(path: Path) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                total_bytes += fp.stat().st_size
                total_files += 1
            except Exception:
                pass
    return total_files, total_bytes


def snapshot_download_hf(
    repo_id: str,
    output_dir: Path,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 8,
    resume_download: bool = True,
) -> None:
    from huggingface_hub import snapshot_download

    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,  # safer on Windows
        max_workers=max_workers,
        resume_download=resume_download,
    )
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    if token:
        # Prefer modern 'token' param; fall back to 'use_auth_token' for older versions
        kwargs["token"] = token

    try:
        snapshot_download(**kwargs)
    except TypeError:
        # Older huggingface_hub uses 'use_auth_token'
        if "token" in kwargs:
            tok = kwargs.pop("token")
            kwargs["use_auth_token"] = tok
        snapshot_download(**kwargs)



def main():
    parser = argparse.ArgumentParser(description="Download SALT-NLP/Sketch2Code from Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="SALT-NLP/Sketch2Code", help="HF dataset repo id")
    parser.add_argument("--output_dir", type=str, default="datasets/Sketch2Code", help="Local output directory")
    parser.add_argument("--revision", type=str, default=None, help="Repo revision (branch/tag/commit)")
    parser.add_argument("--allow", action="append", default=None, help="Allow patterns (repeatable), e.g., train/**")
    parser.add_argument("--ignore", action="append", default=None, help="Ignore patterns (repeatable), e.g., docs/**")
    parser.add_argument("--max_workers", type=int, default=8, help="Concurrent workers for download")
    parser.add_argument("--token", type=str, default=None, help="HF access token; if omitted, reads env var")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    print(f"Repo: {args.repo_id}")
    print(f"Output: {output_dir}")
    if args.revision:
        print(f"Revision: {args.revision}")
    if args.allow:
        print(f"Allow patterns: {args.allow}")
    if args.ignore:
        print(f"Ignore patterns: {args.ignore}")
    print(f"Token: {'provided' if token else 'none'}")
    print("Starting download snapshot...\n")

    try:
        snapshot_download_hf(
            repo_id=args.repo_id,
            output_dir=output_dir,
            revision=args.revision,
            token=token,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            max_workers=args.max_workers,
            resume_download=True,
        )
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        sys.exit(1)

    files, bytes_ = compute_dir_stats(output_dir)
    print("\nDownload completed.")
    print(f"Files: {files}")
    print(f"Size: {sizeof_fmt(bytes_)}")

    # Show a few top-level entries as a quick sanity check
    entries = sorted([p.name for p in output_dir.iterdir()]) if output_dir.exists() else []
    print(f"Top-level entries ({len(entries)}): {entries[:10]}")


if __name__ == "__main__":
    main()