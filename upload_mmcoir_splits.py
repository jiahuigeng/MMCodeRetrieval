#!/usr/bin/env python3
"""
Upload MMCoIR-train and MMCoIR-test splits to Hugging Face dataset repos.
- Upload JSONL and images.tar.gz per dataset subfolder.
- Always upload local README.md from split root (MMCoIR-*/README.md).
- Skip uploading any subfolder that already exists remotely.

Usage examples:
  python upload_mmcoir_splits.py --create-repo --check
  python upload_mmcoir_splits.py --dataset ChartGen --dataset Chart2Code --dry-run
  python upload_mmcoir_splits.py --train-repo myorg/MMCoIR-train --test-repo myorg/MMCoIR-test

Auth: run `huggingface-cli login` or set HF_TOKEN env.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import tarfile
import tempfile

try:
    from huggingface_hub import HfApi, create_repo, list_repo_files
except Exception as e:
    print("[ERROR] huggingface_hub not installed:", e)
    print("pip install -U huggingface_hub")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DEFAULT_TRAIN_REPO = "JiahuiGengNLP/MMCoIR-train"
DEFAULT_TEST_REPO = "JiahuiGengNLP/MMCoIR-test"
DEFAULT_IMAGES_TAR = "images.tar.gz"


# --------------------------
# Utility helpers
# --------------------------
def ensure_repo(repo_id: str, create: bool) -> None:
    if not create:
        return
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[OK] 仓库可用: {repo_id}")
    except Exception as e:
        print(f"[WARN] 创建/确认仓库失败: {e}")


def detect_datasets(root: Path, split: str, specific: Optional[List[str]]) -> List[Path]:
    """Find all subdirectories that contain train.jsonl or test.jsonl."""
    key = "train.jsonl" if split == "train" else "test.jsonl"
    if specific:
        dirs = [(root / name) for name in specific]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    found = [d for d in dirs if (d / key).is_file()]
    return found


def pack_images(images_dir: Path, tar_path: Path, force: bool) -> bool:
    """Pack images directory into tar.gz if needed."""
    if not images_dir.is_dir():
        print(f"[WARN] images 目录不存在: {images_dir}")
        return False
    if tar_path.exists() and not force:
        print(f"[SKIP] 已存在打包文件: {tar_path}")
        return True
    try:
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, mode="w:gz") as tf:
            tf.add(images_dir, arcname="images")
        print(f"[OK] 打包完成: {tar_path}")
        return True
    except Exception as e:
        print(f"[FAIL] 打包失败: {tar_path} -> {e}")
        return False


def verify_local_files(folder: Path, names: List[str]) -> List[Tuple[str, Path]]:
    """Ensure local files exist."""
    mapping: List[Tuple[str, Path]] = []
    missing: List[Path] = []
    for n in names:
        p = folder / n
        if not p.is_file():
            missing.append(p)
        else:
            mapping.append((n, p))
    if missing:
        print("[ERROR] 以下本地文件缺失:")
        for m in missing:
            print(" -", m)
        sys.exit(1)
    return mapping


def upload_files(api: HfApi, repo_id: str, files: List[Tuple[str, Path]], prefix: str, commit_prefix: str, remote_files: List[str], dry_run: bool) -> None:
    """Upload files to repo unless they already exist."""
    for basename, path_obj in files:
        local_path = str(path_obj)
        path_in_repo = f"{prefix}/{basename}"

        if path_in_repo in remote_files:
            print(f"[SKIP] 远端已存在: {path_in_repo}")
            continue

        size = os.path.getsize(local_path) if os.path.exists(local_path) else -1
        print(f"[UPLOAD] {local_path} ({size} bytes) -> {repo_id}:{path_in_repo}")

        if dry_run:
            print(f"[DRY-RUN] 跳过实际上传: {path_in_repo}")
            continue

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                commit_message=f"{commit_prefix} {path_in_repo}",
            )
            print(f"[OK] 已上传: {path_in_repo}")
        except Exception as e:
            print(f"[FAIL] 上传失败: {path_in_repo} -> {e}")
            sys.exit(2)


def verify_remote(repo_id: str, expected_paths: List[str]) -> bool:
    """Check if uploaded files exist remotely."""
    print("[CHECK] 验证远端仓库文件...")
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[WARN] 无法列出远端文件: {e}")
        return False
    missing = [p for p in expected_paths if p not in files]
    if missing:
        print("[ERROR] 远端缺失以下路径:")
        for m in missing:
            print(" -", m)
        return False
    print("[OK] 远端校验通过")
    return True


def upload_local_readme(api: HfApi, repo_id: str, split_root: Path, commit_prefix: str, dry_run: bool):
    """Upload local README.md from split_root to the dataset repo root.
    Fails if README.md does not exist locally.
    """
    readme_path = split_root / "README.md"
    if not readme_path.is_file():
        print(f"[ERROR] 本地 README.md 不存在: {readme_path}")
        sys.exit(1)

    print(f"[UPLOAD] 上传本地 README.md -> {repo_id}:README.md ({readme_path})")

    if dry_run:
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                preview = f.read(500)
            print("[DRY-RUN] README.md 内容预览:")
            print(preview + ("..." if len(preview) == 500 else ""))
        except Exception as e:
            print(f"[DRY-RUN] 读取 README.md 失败: {e}")
        print("[DRY-RUN] 跳过实际上传 README.md")
        return

    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="README.md",
            commit_message=f"{commit_prefix} Update README.md",
        )
        print("[OK] 已上传本地 README.md")
    except Exception as e:
        print(f"[WARN] 上传 README.md 失败: {e}")


# dataset_script.py 上传逻辑已移除


# 生成 README 的逻辑已移除，改为上传本地 README.md


# --------------------------
# Main upload logic
# --------------------------
def process_split(api: HfApi, split_root: Path, repo_id: str, split: str, datasets: Optional[List[str]], images_tar_name: str, force_repack: bool, create_repo_flag: bool, check: bool, dry_run: bool, commit_prefix: str) -> None:
    ds_dirs = detect_datasets(split_root, split, datasets)
    if not ds_dirs:
        print(f"[WARN] 未在 {split_root} 发现 {split}.jsonl 所在的数据集子目录")
        return

    print(f"[INFO] 处理 {split_root} -> {repo_id}, 数据集数: {len(ds_dirs)}")
    ensure_repo(repo_id, create_repo_flag)

    # 始终上传本地 README.md
    upload_local_readme(api, repo_id, split_root, commit_prefix, dry_run)
    
    # 不再上传 dataset_script.py

    try:
        remote_files = list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        remote_files = []

    for d in ds_dirs:
        name = d.name
        key = "train.jsonl" if split == "train" else "test.jsonl"
        images_dir = d / "images"
        tar_path = d / images_tar_name

        # 如果该子目录已存在于远端，跳过整个数据集上传
        if any(f.startswith(f"{name}/") for f in remote_files):
            print(f"[SKIP] 远端已存在子目录: {name}/，跳过整个上传。")
            continue

        print(f"[DATASET] {name} ({split})")

        # 打包 images
        if not dry_run:
            pack_images(images_dir, tar_path, force=force_repack)
        else:
            print(f"[DRY-RUN] 打包跳过: {images_dir} -> {tar_path}")

        # 校验本地文件
        targets = [key, images_tar_name]
        mapping = verify_local_files(d, targets)

        # 上传
        upload_files(api, repo_id, mapping, prefix=name, commit_prefix=commit_prefix, remote_files=remote_files, dry_run=dry_run)

        # 上传后校验
        if check:
            exp = [f"{name}/{b}" for b, _ in mapping]
            verify_remote(repo_id, exp)


# --------------------------
# CLI entry
# --------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Upload MMCoIR-train/test JSONL + images.tar.gz to HF")
    ap.add_argument("--train-root", default=str(DEFAULT_TRAIN_ROOT), help="本地训练集根目录 (含各数据集子目录)")
    ap.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT), help="本地测试集根目录 (含各数据集子目录)")
    ap.add_argument("--train-repo", default=DEFAULT_TRAIN_REPO, help="训练集目标数据集仓库 ID")
    ap.add_argument("--test-repo", default=DEFAULT_TEST_REPO, help="测试集目标数据集仓库 ID")
    ap.add_argument("--dataset", action="append", default=[], help="限定数据集子目录名，可重复 (如 ChartGen, Chart2Code)")
    ap.add_argument("--images-tar", default=DEFAULT_IMAGES_TAR, help="打包文件名 (默认 images.tar.gz)")
    ap.add_argument("--force-repack", action="store_true", help="强制重新打包 images.tar.gz")
    ap.add_argument("--create-repo", action="store_true", help="若不存在则创建数据集仓库")
    ap.add_argument("--check", action="store_true", help="上传后校验远端是否存在文件")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划并检查本地，不实际上传")
    ap.add_argument("--commit-prefix", default="Add", help="提交信息前缀")
    args = ap.parse_args()

    api = HfApi()
    datasets = args.dataset if args.dataset else None

    # train split
    process_split(
        api=api,
        split_root=Path(args.train_root),
        repo_id=args.train_repo,
        split="train",
        datasets=datasets,
        images_tar_name=args.images_tar,
        force_repack=args.force_repack,
        create_repo_flag=args.create_repo,
        check=args.check,
        dry_run=args.dry_run,
        commit_prefix=args.commit_prefix,
    )

    # test split
    process_split(
        api=api,
        split_root=Path(args.test_root),
        repo_id=args.test_repo,
        split="test",
        datasets=datasets,
        images_tar_name=args.images_tar,
        force_repack=args.force_repack,
        create_repo_flag=args.create_repo,
        check=args.check,
        dry_run=args.dry_run,
        commit_prefix=args.commit_prefix,
    )

    print("[DONE] 上传流程完成。")


if __name__ == "__main__":
    main()