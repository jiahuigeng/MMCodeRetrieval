#!/usr/bin/env python3
"""
下载 Hugging Face 上的 starvector/svg-stack 数据集到本地 datasets 目录。

功能
- 使用 huggingface_hub 的 snapshot_download 将数据集完整快照到本地文件夹。
- 支持可选的访问令牌（私有/门控数据集），可从环境变量读取。
- 在 Windows 上禁用符号链接，确保真实文件落地。

用法示例
  # 基础用法（下载到 datasets/SVGStack）
  python SVGStack/download_svgstack.py

  # 指定输出目录
  python SVGStack/download_svgstack.py --outdir datasets/SVGStack

  # 指定数据集与修订版本
  python SVGStack/download_svgstack.py --repo_id starvector/svg-stack --revision main

前置依赖
  pip install huggingface_hub
  # 若数据集需要令牌，可设置环境变量
  set HUGGINGFACE_TOKEN=hf_********************************   # Windows
  export HUGGINGFACE_TOKEN=hf_******************************** # Linux/macOS
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download starvector/svg-stack dataset to local 'datasets' folder.")
    p.add_argument("--repo_id", type=str, default="starvector/svg-stack", help="HF Hub dataset repo id")
    p.add_argument("--outdir", type=str, default=str(Path("datasets") / "SVGStack"), help="Local output directory")
    p.add_argument("--revision", type=str, default="main", help="Dataset revision/tag/branch")
    p.add_argument("--hf_token", type=str, default=None, help="Optional HF token, falls back to env HUGGINGFACE_TOKEN")
    p.add_argument("--workers", type=int, default=8, help="Max workers for download")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    token: Optional[str] = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    print(f"[INFO] Downloading '{args.repo_id}' to '{out_path.resolve()}' ...")
    try:
        local_snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(out_path),
            local_dir_use_symlinks=False,  # Windows 更安全
            revision=args.revision,
            token=token,
            max_workers=args.workers,
        )
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        raise

    print(f"[INFO] 下载完成，数据已保存到: {local_snapshot_path}")


if __name__ == "__main__":
    main()