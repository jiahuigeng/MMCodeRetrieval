#!/usr/bin/env python3
"""
upload_mmcoir_images.py

Purpose:
- Given an images folder (e.g., MMCoIR-train/images/MMSVG-Icon/ or .../images/MMSVG-Icon/images/),
  compress the contained images directory into a .tar.gz and upload it to a Hugging Face dataset repo.

Defaults:
- Auto-detect target repo by path: if the path includes 'MMCoIR-train' -> JiahuiGengNLP/MMCoIR-train,
  if includes 'MMCoIR-test' -> JiahuiGengNLP/MMCoIR-test. Can be overridden with --repo-id.
- The dataset name is inferred from the path. If given .../images/<Dataset>/images, it uses <Dataset>.
- The tarball is named 'images.tar.gz' by default and uploaded to 'images/<Dataset>/images.tar.gz' in the repo (remote prefix defaults to 'images').
- You can optionally set a different remote prefix (e.g., '--remote-prefix assets') to change the upload path.

Notes:
- You must be logged in to Hugging Face (via huggingface-cli login or env HF_TOKEN).
- This script is inspired by existing split uploaders but focuses on images-only packing/upload.
- If a local 'images.tar.gz' exists and is readable (valid), the script skips compression and uploads it; if missing or corrupted, it re-packs from the 'images/' directory.

Usage examples:
  python upload_mmcoir_images.py --images-dir MMCoIR-train/images/MMSVG-Icon/
  python upload_mmcoir_images.py --images-dir MMCoIR-test/images/SVGStack/
  python upload_mmcoir_images.py --images-dir MMCoIR-train/images/MMSVG-Icon/images --force
  python upload_mmcoir_images.py --images-dir MMCoIR-train/images/MMSVG-Icon/ --repo-id JiahuiGengNLP/MMCoIR-train --dry-run
  python upload_mmcoir_images.py --images-dir MMCoIR-train/images/MMSVG-Icon/images --remote-prefix images --check

"""

import argparse
import os
from pathlib import Path
import tarfile
from typing import Optional

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError


DEFAULT_TRAIN_REPO = "JiahuiGengNLP/MMCoIR-train"
DEFAULT_TEST_REPO = "JiahuiGengNLP/MMCoIR-test"


def detect_repo_from_path(path: Path, train_repo: str, test_repo: str) -> str:
    parts = {p.lower() for p in path.parts}
    if "mmcoir-train" in parts:
        return train_repo
    if "mmcoir-test" in parts:
        return test_repo
    # Fallback: default to train repo
    return train_repo


def infer_dataset_name(images_dir: Path) -> str:
    # If the provided path ends with 'images', the dataset is the parent folder name.
    if images_dir.name.lower() == "images" and images_dir.parent.name:
        return images_dir.parent.name
    # If the provided path contains an 'images' child, use current folder name as dataset.
    if (images_dir / "images").exists():
        return images_dir.name
    # Otherwise, use the current folder name.
    return images_dir.name


def normalize_images_dir(images_dir: Path) -> Path:
    """Return the actual directory that contains image files to pack.

    Accept both of:
    - MMCoIR-<split>/images/<Dataset>/
    - MMCoIR-<split>/images/<Dataset>/images/
    If the given directory contains a child 'images', use that; else assume the directory itself is the images container.
    """
    if (images_dir / "images").exists() and (images_dir / "images").is_dir():
        return images_dir / "images"
    return images_dir


def is_tar_gz_valid(tar_path: Path) -> bool:
    """Return True if the .tar.gz exists, can be read, and has entries under 'images/'."""
    if not tar_path.exists() or not tar_path.is_file():
        return False
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            if not members:
                return False
            # Ensure the archive contains the expected 'images' root or entries under it
            has_images_root = any(m.name == "images" and m.isdir() for m in members)
            has_images_entries = any(m.name.startswith("images/") for m in members)
            return has_images_root or has_images_entries
    except (tarfile.ReadError, OSError):
        return False


def pack_images(images_dir: Path, tar_path: Path, force: bool = False) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if tar_path.exists():
        if force:
            tar_path.unlink()
        else:
            # If not forcing, keep existing tarball
            print(f"Tar already exists, skipping repack: {tar_path}")
            return

    actual_images_dir = normalize_images_dir(images_dir)
    if not actual_images_dir.exists() or not actual_images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {actual_images_dir}")

    # Create tar.gz with arcname 'images' for consistency
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(actual_images_dir, arcname="images")
    print(f"Packed images into: {tar_path}")


def ensure_repo(api: HfApi, repo_id: str, create_repo: bool = False) -> None:
    try:
        api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError as e:
        if create_repo:
            print(f"Repo not found. Creating dataset repo: {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        else:
            raise RuntimeError(f"Repo {repo_id} not found and --create-repo not set: {e}")


def upload_tar(api: HfApi, repo_id: str, local_tar: Path, remote_path: str, dry_run: bool = False, commit_prefix: Optional[str] = None) -> None:
    if dry_run:
        print(f"[Dry-run] Would upload {local_tar} to {repo_id}:{remote_path}")
        return

    commit_msg = (commit_prefix or "Upload") + f" {remote_path}"
    url = api.upload_file(
        path_or_fileobj=str(local_tar),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=remote_path,
        commit_message=commit_msg,
    )
    print(f"Uploaded: {url}")


def check_remote(api: HfApi, repo_id: str, remote_path: str) -> bool:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError:
        return False
    return remote_path in files


def main():
    parser = argparse.ArgumentParser(description="Pack a specified images folder to tar.gz and upload to Hugging Face dataset repo.")
    parser.add_argument("--images-dir", required=True, help="Path to images folder, e.g. MMCoIR-train/images/MMSVG-Icon/ or .../images/MMSVG-Icon/images/")
    parser.add_argument("--repo-id", default=None, help="Target HF dataset repo (overrides auto-detection).")
    parser.add_argument("--train-repo", default=DEFAULT_TRAIN_REPO, help="Default HF dataset repo for train.")
    parser.add_argument("--test-repo", default=DEFAULT_TEST_REPO, help="Default HF dataset repo for test.")
    parser.add_argument("--tar-name", default="images.tar.gz", help="Name of the tar.gz file to produce.")
    parser.add_argument("--force", action="store_true", help="Force re-pack even if tar already exists.")
    parser.add_argument("--create-repo", action="store_true", help="Create repo if missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without performing uploads.")
    parser.add_argument("--commit-prefix", default="Upload", help="Commit message prefix.")
    parser.add_argument("--check", action="store_true", help="Check if remote file exists after upload.")
    parser.add_argument("--remote-prefix", default="images", help="Optional path prefix in repo (default: 'images').")

    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Provided images-dir does not exist: {images_dir}")

    dataset = infer_dataset_name(images_dir)
    repo_id = args.repo_id or detect_repo_from_path(images_dir, args.train_repo, args.test_repo)

    # Local tarball location: place next to the provided folder (not inside 'images') for clarity
    local_tar = images_dir if images_dir.name.lower() != "images" else images_dir.parent
    local_tar = local_tar / args.tar_name

    # Decide whether to (re)pack based on existence and validity
    existing_valid = is_tar_gz_valid(local_tar)
    if args.force:
        print(f"--force specified; will re-pack {local_tar}.")
        pack_images(images_dir, local_tar, force=True)
    elif existing_valid:
        print(f"Existing tar is valid; skip compression: {local_tar}")
    else:
        if local_tar.exists():
            print(f"Existing tar invalid or corrupted; re-pack: {local_tar}")
        else:
            print(f"No tar found; will pack: {local_tar}")
        pack_images(images_dir, local_tar, force=True)

    api = HfApi()
    ensure_repo(api, repo_id, create_repo=args.create_repo)

    # Normalize remote prefix: allow 'images', 'images/', '/images/' etc.
    rp = args.remote_prefix.strip("/")
    remote_path = f"{rp + '/' if rp else ''}{dataset}/{args.tar_name}"
    upload_tar(api, repo_id, local_tar, remote_path, dry_run=args.dry_run, commit_prefix=args.commit_prefix)

    if args.check:
        exists = check_remote(api, repo_id, remote_path)
        print(f"Remote check for {repo_id}:{remote_path} -> {'FOUND' if exists else 'MISSING'}")


if __name__ == "__main__":
    main()