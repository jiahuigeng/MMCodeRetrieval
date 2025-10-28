#!/usr/bin/env python3
"""
Upload MMCoIR-train and MMCoIR-test splits to Hugging Face dataset repos.
- Upload only JSONL and images.tar.gz per dataset subfolder.
- Train -> repo JiahuiGengNLP/MMCoIR-train (default)
- Test  -> repo JiahuiGengNLP/MMCoIR-test  (default)
- Each dataset subdir is used as prefix in the target repo (e.g., ChartGen, Chart2Code).

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


def ensure_repo(repo_id: str, create: bool) -> None:
    if not create:
        return
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[OK] 仓库可用: {repo_id}")
    except Exception as e:
        print(f"[WARN] 创建/确认仓库失败: {e}")


def detect_datasets(root: Path, split: str, specific: Optional[List[str]]) -> List[Path]:
    if specific:
        dirs = [(root / name) for name in specific]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    key = "train.jsonl" if split == "train" else "test.jsonl"
    found = [d for d in dirs if (d / key).is_file()]
    return found


def pack_images(images_dir: Path, tar_path: Path, force: bool) -> bool:
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


def upload_files(api: HfApi, repo_id: str, local_folder: Path, files: List[Tuple[str, Path]], prefix: str, commit_prefix: str) -> None:
    for basename, path_obj in files:
        local_path = str(path_obj)
        path_in_repo = f"{prefix}/{basename}"
        try:
            size = os.path.getsize(local_path)
        except OSError:
            size = -1
        print(f"[UPLOAD] {local_path} ({size} bytes) -> {repo_id}:{path_in_repo}")
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


def process_split(api: HfApi, split_root: Path, repo_id: str, split: str, datasets: Optional[List[str]], images_tar_name: str, force_repack: bool, create_repo: bool, check: bool, dry_run: bool, commit_prefix: str) -> None:
    ds_dirs = detect_datasets(split_root, split, datasets)
    if not ds_dirs:
        print(f"[WARN] 未在 {split_root} 发现 {split}.jsonl 所在的数据集子目录")
        return
    print(f"[INFO] 处理 {split_root} -> {repo_id}, 数据集数: {len(ds_dirs)}")

    # optional create repo
    if create_repo:
        ensure_repo(repo_id, create=True)

    for d in ds_dirs:
        name = d.name
        key = "train.jsonl" if split == "train" else "test.jsonl"
        images_dir = d / "images"
        tar_path = d / images_tar_name

        print(f"[DATASET] {name} ({split})")
        # pack images
        if not dry_run:
            pack_images(images_dir, tar_path, force=force_repack)
        else:
            print(f"[DRY-RUN] 打包跳过: {images_dir} -> {tar_path}")

        # verify local files and build mapping
        targets = [key, images_tar_name]
        mapping = verify_local_files(d, targets)

        # upload
        if dry_run:
            for b, p in mapping:
                print(f"[DRY-RUN] 计划上传: {p} -> {repo_id}:{name}/{b}")
        else:
            upload_files(api, repo_id, d, mapping, prefix=name, commit_prefix=commit_prefix)

        # optional check
        if check:
            exp = [f"{name}/{b}" for b, _ in mapping]
            ok = verify_remote(repo_id, exp)
            if not ok:
                print(f"[WARN] 远端校验失败: {repo_id}::{name}")


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
        create_repo=args.create_repo,
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
        create_repo=args.create_repo,
        check=args.check,
        dry_run=args.dry_run,
        commit_prefix=args.commit_prefix,
    )

    print("[DONE] 上传流程完成。")


if __name__ == "__main__":
    main()