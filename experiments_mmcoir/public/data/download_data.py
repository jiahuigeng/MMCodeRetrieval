import argparse
import os
from typing import List, Optional, Tuple
from huggingface_hub import snapshot_download

TRAIN_REPO = "JiahuiGengNLP/MMCoIR-train"
TEST_REPO = "JiahuiGengNLP/MMCoIR-test"


def resolve_paths(data_dir_arg: Optional[str]) -> Tuple[str, str]:
    """Resolve local output directories for train and test under a base data dir.

    If --data-dir is not provided, base dir defaults to <repo_root>/data, where
    repo_root is calculated from this script's location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    base_data_dir = data_dir_arg or os.path.join(repo_root, "data")
    train_dir = os.path.join(base_data_dir, "MMCoIR_train")
    test_dir = os.path.join(base_data_dir, "MMCoIR_test")
    return train_dir, test_dir


def to_patterns(pattern_str: Optional[str]) -> Optional[List[str]]:
    if not pattern_str:
        return None
    return [p.strip() for p in pattern_str.split(",") if p.strip()]


def download_one(repo_id: str, out_dir: str, allow_patterns: Optional[List[str]],
                 token: Optional[str], repo_type: str, use_symlinks: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {repo_id} -> {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=out_dir,
        local_dir_use_symlinks=use_symlinks,
        token=token,
        allow_patterns=allow_patterns,
    )
    print(f"Done: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MMCoIR datasets from Hugging Face to local data directory"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download both train and test")
    group.add_argument("--target", choices=["train", "test"], help="Download only one dataset")

    parser.add_argument("--data-dir", type=str, default=None,
                        help="Base local data directory. Default: <repo_root>/data")
    parser.add_argument("--allow-patterns", type=str, default=None,
                        help="Comma-separated allow patterns to download a subset, e.g. 'ChartGen/*,PlantUML/*'")
    parser.add_argument("--repo-type", type=str, default="dataset", choices=["dataset", "model", "space"],
                        help="Repo type on Hugging Face. Default: dataset")
    parser.add_argument("--use-symlinks", action="store_true",
                        help="Use symlinks in local_dir (disabled by default; not recommended on Windows)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"),
                        help="Optional token override. If not set, uses cached login from huggingface-cli")

    args = parser.parse_args()

    allow_patterns = to_patterns(args.allow_patterns)
    train_dir, test_dir = resolve_paths(args.data_dir)

    print(f"Base data dir: {os.path.dirname(train_dir)}")
    print(f"Train out dir: {train_dir}")
    print(f"Test  out dir: {test_dir}")

    if args.all:
        download_one(TRAIN_REPO, train_dir, allow_patterns, args.hf_token, args.repo_type, args.use_symlinks)
        download_one(TEST_REPO, test_dir, allow_patterns, args.hf_token, args.repo_type, args.use_symlinks)
    else:
        if args.target == "train":
            download_one(TRAIN_REPO, train_dir, allow_patterns, args.hf_token, args.repo_type, args.use_symlinks)
        else:
            download_one(TEST_REPO, test_dir, allow_patterns, args.hf_token, args.repo_type, args.use_symlinks)

    print("All downloads completed.")


if __name__ == "__main__":
    main()