import os
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 Hugging Face 数据集 nllg/datikz-v3 到项目根目录的 datasets 目录"
    )
    parser.add_argument(
        "--repo-id",
        default="nllg/datikz-v3",
        help="Hugging Face 数据集仓库 ID，默认：nllg/datikz-v3",
    )
    # 默认下载到 <repo_root>/datasets/datikz-v3
    repo_root = Path(__file__).resolve().parent.parent
    default_outdir = repo_root / "datasets" / "datikz-v3"
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
        help=f"本地保存目录（默认：{default_outdir}）",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选：指定分支/标签/commit 哈希",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="可选：Hugging Face 访问令牌（也可用环境变量 HF_TOKEN）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发下载线程数（默认：8）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 正在下载 '{args.repo_id}' 到 '{out_path}' ...")
    try:
        local_snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(out_path),
            local_dir_use_symlinks=False,  # Windows 更安全
            revision=args.revision,
            token=args.hf_token,
            max_workers=args.workers,
        )
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        raise

    print(f"[INFO] 下载完成，数据已保存到: {local_snapshot_path}")


if __name__ == "__main__":
    main()