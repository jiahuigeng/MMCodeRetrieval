import os
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download DiagramAgent/DiagramGenBenchmark dataset to local datasets directory")
    parser.add_argument(
        "--repo-id",
        default="DiagramAgent/DiagramGenBenchmark",
        help="Hugging Face dataset repo_id (default: DiagramAgent/DiagramGenBenchmark)",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).parent / "datasets" / "DiagramGenBenchmark"),
        help="Local output directory to store the dataset (default: ./datasets/DiagramGenBenchmark)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face access token (env HF_TOKEN also respected)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers (default: 8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading '{args.repo_id}' to '{out_path}' ...")
    try:
        local_snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(out_path),
            local_dir_use_symlinks=False,  # safer on Windows
            revision=args.revision,
            token=args.hf_token,
            max_workers=args.workers,
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        raise

    print(f"[INFO] Done. Dataset stored at: {local_snapshot_path}")


if __name__ == "__main__":
    main()