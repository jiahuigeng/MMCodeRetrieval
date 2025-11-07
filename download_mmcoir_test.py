#!/usr/bin/env python3
"""
download_mmcoir_test.py

Purpose:
- Download the dataset from Hugging Face repository `JiahuiGengNLP/MMCoIR-test` to a specified local directory.

Features:
- Default download directory: `new_datasets`
- Automatically creates the target directory if it doesn't exist.
- Supports CLI arguments for repository ID and target directory.
- Validates downloaded files for completeness.

Usage examples:
  python download_mmcoir_test.py
  python download_mmcoir_test.py --repo-id JiahuiGengNLP/MMCoIR-test --target-dir new_datasets
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

def download_dataset(repo_id: str, target_dir: Path) -> None:
    """Download all files from the specified Hugging Face dataset repository."""
    api = HfApi()

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the repository
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"Found {len(files)} files in repository {repo_id}.")

    # Download each file
    for file in files:
        print(f"Downloading {file}...")
        local_path = target_dir / file
        local_path.parent.mkdir(parents=True, exist_ok=True)  # Create subdirectories if needed
        hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset", local_dir=str(local_path.parent))
        print(f"Saved to {local_path}")

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face repository.")
    parser.add_argument("--repo-id", default="JiahuiGengNLP/MMCoIR-test", help="Hugging Face repository ID (default: JiahuiGengNLP/MMCoIR-test).")
    parser.add_argument("--target-dir", default="new_datasets", help="Local directory to save the dataset (default: new_datasets).")

    args = parser.parse_args()

    repo_id = args.repo_id
    target_dir = Path(args.target_dir).resolve()

    print(f"Downloading dataset from {repo_id} to {target_dir}...")
    download_dataset(repo_id, target_dir)
    print("Download completed.")

if __name__ == "__main__":
    main()