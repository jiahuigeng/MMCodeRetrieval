#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Hugging Face 数据集 xxxllz/ChartEdit 到本地 datasets 目录。

示例用法:
    python download_chartedit.py
    python download_chartedit.py --output_dir ./datasets/ChartEdit
    python download_chartedit.py --use_mirror --mirror_url https://hf-mirror.com
    python download_chartedit.py --proxy http://127.0.0.1:7890

说明:
    - 默认下载到: datasets/ChartEdit
    - 使用 huggingface_hub.snapshot_download 直接拉取数据集仓库快照
    - 支持镜像源与代理配置，便于国内网络环境下载
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def try_import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except Exception as e:
        print("[ERROR] 未找到 huggingface_hub。请先安装：pip install huggingface_hub")
        print(f"详细错误：{e}")
        sys.exit(1)


def setup_hf_mirror(use_mirror: bool = True, mirror_url: str = "https://hf-mirror.com") -> None:
    """配置 HuggingFace 镜像源。

    Args:
        use_mirror: 是否启用镜像源
        mirror_url: 镜像地址
    """
    if use_mirror:
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info(f"使用 HuggingFace 镜像: {mirror_url}")
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        logger.info("使用官方 HuggingFace Hub")


def setup_proxy(proxy: Optional[str] = None) -> None:
    """配置 HTTP/HTTPS 代理。"""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        logger.info(f"使用代理: {proxy}")


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"


def compute_dir_stats(path: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if not path.exists():
        return files, total_bytes
    for p in path.rglob('*'):
        if p.is_file():
            files += 1
            try:
                total_bytes += p.stat().st_size
            except Exception:
                pass
    return files, total_bytes


def snapshot_download_dataset(
    repo_id: str,
    output_dir: Path,
    revision: str = "main",
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 4,
    resume_download: bool = True,
) -> None:
    """使用 snapshot_download 下载数据集快照到指定目录。"""
    snapshot_download = try_import_snapshot_download()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"开始下载数据集: {repo_id}")
    logger.info(f"输出目录: {output_dir.resolve()}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            revision=revision,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            resume_download=resume_download,
        )
    except Exception as e:
        logger.error(f"下载失败: {e}")
        sys.exit(1)

    files, bytes_ = compute_dir_stats(output_dir)
    logger.info("下载完成。")
    logger.info(f"文件数: {files}")
    logger.info(f"总大小: {sizeof_fmt(bytes_)}")
    entries = sorted([p.name for p in output_dir.iterdir()]) if output_dir.exists() else []
    logger.info(f"顶层条目({len(entries)}): {entries[:10]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 Hugging Face 数据集 xxxllz/ChartEdit 到 datasets/ChartEdit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", type=str, default="xxxllz/ChartEdit", help="HF 数据集仓库ID")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("datasets") / "ChartEdit"),
        help="下载输出目录",
    )
    parser.add_argument("--revision", type=str, default="main", help="仓库版本")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace访问令牌(如需)")
    parser.add_argument("--allow", nargs="*", default=None, help="允许下载的通配文件模式列表")
    parser.add_argument("--ignore", nargs="*", default=None, help="忽略下载的通配文件模式列表")
    parser.add_argument("--max_workers", type=int, default=4, help="并行下载线程数")
    parser.add_argument("--no_resume", action="store_true", help="禁用断点续传")

    # 镜像与代理
    parser.add_argument("--use_mirror", action="store_true", default=True, help="使用 HuggingFace 镜像")
    parser.add_argument("--no_mirror", action="store_true", help="禁用镜像，使用官方Hub")
    parser.add_argument("--mirror_url", type=str, default="https://hf-mirror.com", help="镜像地址")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP/HTTPS代理，如 http://127.0.0.1:7890")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 镜像 & 代理
    use_mirror = args.use_mirror and not args.no_mirror
    setup_hf_mirror(use_mirror=use_mirror, mirror_url=args.mirror_url)
    setup_proxy(proxy=args.proxy)

    output_dir = Path(args.output_dir)
    snapshot_download_dataset(
        repo_id=args.repo_id,
        output_dir=output_dir,
        revision=args.revision,
        token=args.token,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
        max_workers=args.max_workers,
        resume_download=(not args.no_resume),
    )


if __name__ == "__main__":
    main()