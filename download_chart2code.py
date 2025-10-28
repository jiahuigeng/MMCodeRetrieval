#!/usr/bin/env python3
"""
任务说明（请勿轻易改动）：
- 仅下载 Hugging Face 数据集 `xxxllz/Chart2Code-160k` 到 `datasets/chart2code_160k/`。
- 不做任何额外处理、过滤或复制操作。
"""

import os
import argparse

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("请先安装 huggingface_hub: pip install huggingface_hub")


def main():
    parser = argparse.ArgumentParser(description="下载 xxxllz/Chart2Code-160k 到 datasets/chart2code_160k")
    parser.add_argument("--repo-id", default="xxxllz/Chart2Code-160k", help="Hugging Face 数据集仓库 ID")
    parser.add_argument("--target-dir", default=os.path.join("datasets", "chart2code_160k"), help="下载目标目录")
    args = parser.parse_args()

    print(f"开始下载: repo_id={args.repo_id} -> {args.target_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.target_dir,
        local_dir_use_symlinks=False,
    )
    print("下载完成。")


if __name__ == "__main__":
    main()