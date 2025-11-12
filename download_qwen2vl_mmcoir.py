#!/usr/bin/env python3
"""
下载 Hugging Face 模型脚本（专用：trumancai/qwen2vl_mmcoir）

功能：
- 从 Hugging Face Hub 下载模型仓库到指定本地目录（非仅缓存）。
- 支持镜像源（HF-Mirror）、代理、token、版本指定、文件过滤。

用法示例：
  python download_qwen2vl_mmcoir.py
  python download_qwen2vl_mmcoir.py --local-dir models/qwen2vl_mmcoir
  python download_qwen2vl_mmcoir.py --use-mirror --proxy http://127.0.0.1:7890
  python download_qwen2vl_mmcoir.py --allow-patterns "*.json" "*.safetensors"

注意：
- 如需私有模型或更高速下载，建议配置环境变量 HF_TOKEN 或使用 --token。
- 若网络受限，建议启用 --use-mirror 或设置代理。
"""

import os
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("[ERROR] 请先安装 huggingface_hub：pip install -U huggingface_hub")
    raise


DEFAULT_REPO_ID = "trumancai/qwen2vl_mmcoir"


def setup_env(use_mirror: bool = False, proxy: str | None = None, token: str | None = None):
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[INFO] 使用镜像源：https://hf-mirror.com")
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"[INFO] 使用代理：{proxy}")
    if token:
        os.environ["HF_TOKEN"] = token
        print("[INFO] 已设置 Hugging Face token")


def derive_dest_name(repo_id: str, prefix: str | None = "t") -> str:
    name = repo_id.split("/")[-1]
    if prefix:
        return f"{prefix}{name}"
    return name


def main():
    parser = argparse.ArgumentParser(description="Download HF model for Qwen2VL MMCoIR")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="模型仓库ID，例如 trumancai/Qwen2VL-2B-mmcoir-imageonly-lora8-len512-ckpt-400")
    parser.add_argument("--revision", default=None, help="分支/版本（默认仓库当前版本）")
    # 目录/命名策略：若显式提供 --local-dir 则优先使用；否则按 models_dir/dest_name 拼接，其中 dest_name 默认为 't'+仓库末段名
    parser.add_argument("--local-dir", default=None, help="显式指定本地保存目录（优先级最高）")
    parser.add_argument("--models-dir", default="models", help="模型保存的根目录（当未显式指定 local-dir 时使用）")
    parser.add_argument("--dest-prefix", default="", help="派生目标目录名的前缀，例如 't' => t<模型名>")
    parser.add_argument("--dest-name", default=None, help="派生的目标目录名（默认按 repo-id 末段加前缀），例如 tQwen2VL-2B-...")
    parser.add_argument("--cache-dir", default=None, help="缓存目录（可选）")
    parser.add_argument("--allow-patterns", nargs="*", default=None, help="允许下载的文件模式列表")
    parser.add_argument("--ignore-patterns", nargs="*", default=None, help="忽略的文件模式列表")
    parser.add_argument("--use-mirror", action="store_true", help="使用 HF-Mirror 镜像源")
    parser.add_argument("--proxy", default=None, help="HTTP(S) 代理地址，如 http://127.0.0.1:7890")
    parser.add_argument("--token", default=None, help="Hugging Face 访问令牌（可选）")

    args = parser.parse_args()

    # 解析本地保存目录
    if args.local_dir:
        local_dir = Path(args.local_dir).resolve()
        derived_from = "--local-dir"
    else:
        dest_name = args.dest_name or derive_dest_name(args.repo_id, args.dest_prefix)
        local_dir = Path(args.models_dir) / dest_name
        local_dir = local_dir.resolve()
        derived_from = f"{args.models_dir}/{dest_name}"
    local_dir.mkdir(parents=True, exist_ok=True)

    setup_env(use_mirror=args.use_mirror, proxy=args.proxy, token=args.token)

    print(f"[INFO] 下载模型：{args.repo_id}")
    print(f"[INFO] 保存路径：{local_dir} （来源：{derived_from}）")
    if args.revision:
        print(f"[INFO] 使用版本：{args.revision}")
    if args.allow_patterns:
        print(f"[INFO] 允许文件：{args.allow_patterns}")
    if args.ignore_patterns:
        print(f"[INFO] 忽略文件：{args.ignore_patterns}")

    # 执行下载到本地目录（复制真实文件，而不是仅缓存软链接）
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        resume_download=True,
        max_workers=4,
    )

    print("[DONE] 模型下载完成。")


if __name__ == "__main__":
    main()