#!/usr/bin/env python3
"""
download_mmcoir_test.py

功能:
- 从 Hugging Face 数据集仓库（默认: `JiahuiGengNLP/MMCoIR-test`）下载到本地指定目录。
- 可选只下载某个子目录（例如 `DiagramGenBenchmark_i2c`）。

特性:
- 默认目标目录: `new_datasets`（可通过参数覆盖）
- 自动创建目标目录与子目录
- 支持按子目录或通配模式过滤下载文件

用法示例:
  # 全量下载
  python download_mmcoir_test.py

  # 下载到指定目录
  python download_mmcoir_test.py --repo-id JiahuiGengNLP/MMCoIR-test --target-dir data/MMCoIR_test

  # 仅下载某个子目录（推荐本需求）
  python download_mmcoir_test.py --repo-id JiahuiGengNLP/MMCoIR-test --target-dir data/MMCoIR_test --subset DiagramGenBenchmark_i2c

  # 使用通配符过滤（多个模式用逗号分隔）
  python download_mmcoir_test.py --allow-patterns "DiagramGenBenchmark_i2c/*,DiagramGenBenchmark_i2c/**"
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
from fnmatch import fnmatch
from huggingface_hub import HfApi, hf_hub_download

def _filter_files(files: List[str], subset: Optional[str], allow_patterns: Optional[List[str]]) -> List[str]:
    """Filter files by subset prefix or glob patterns.

    Note: do NOT include the subset folder itself (e.g., 'DiagramGenBenchmark_i2c') as a file,
    only keep paths under it (e.g., 'DiagramGenBenchmark_i2c/...').
    """
    if subset:
        files = [f for f in files if f.startswith(f"{subset}/")]
    if allow_patterns:
        # Keep files that match ANY of the patterns
        allowed = []
        for f in files:
            if any(fnmatch(f, pat.strip()) for pat in allow_patterns):
                allowed.append(f)
        # If patterns were provided but none matched, fall back to original filtered list
        if allowed:
            files = allowed
    return files


def download_dataset(repo_id: str, target_dir: Path, subset: Optional[str] = None, allow_patterns: Optional[List[str]] = None) -> None:
    """Download files from the specified Hugging Face dataset repository.

    Args:
        repo_id: HF 数据集仓库名
        target_dir: 本地根目录
        subset: 仅下载该子目录（例如 'DiagramGenBenchmark_i2c'）
        allow_patterns: 允许的通配模式列表（如 ['DiagramGenBenchmark_i2c/*']）
    """
    api = HfApi()

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the repository
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"Found {len(files)} files in repository {repo_id}.")

    # Filter files
    files = _filter_files(files, subset=subset, allow_patterns=allow_patterns)
    print(f"Planned to download {len(files)} files after filtering.")

    # Download each file
    for file in files:
        print(f"Downloading {file}...")
        # Ensure the subdirectory exists under target_dir
        subdir = os.path.dirname(file)
        if subdir:
            (target_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Always set local_dir to the root target_dir, so hf_hub_download
        # will place the file at target_dir/<file>, preserving repo subfolders
        hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset", local_dir=str(target_dir))
        print(f"Saved to {target_dir / file}")

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face repository.")
    parser.add_argument("--repo-id", default="JiahuiGengNLP/MMCoIR-test", help="Hugging Face repository ID (default: JiahuiGengNLP/MMCoIR-test).")
    parser.add_argument("--target-dir", default="new_datasets", help="Local directory to save the dataset (default: new_datasets).")
    parser.add_argument("--subset", default=None, help="Only download a specific subfolder (e.g., DiagramGenBenchmark_i2c).")
    parser.add_argument("--allow-patterns", default=None, help="Comma-separated allow patterns, e.g., 'DiagramGenBenchmark_i2c/*,DiagramGenBenchmark_i2c/**'.")

    args = parser.parse_args()

    repo_id = args.repo_id
    target_dir = Path(args.target_dir).resolve()

    allow_patterns = args.allow_patterns.split(",") if args.allow_patterns else None
    print(f"Downloading dataset from {repo_id} to {target_dir}...")
    if args.subset:
        print(f"Subset filter: {args.subset}")
    if allow_patterns:
        print(f"Allow patterns: {allow_patterns}")
    download_dataset(repo_id, target_dir, subset=args.subset, allow_patterns=allow_patterns)
    print("Download completed.")

if __name__ == "__main__":
    main()