#!/usr/bin/env python3
"""
Upload chart2code files to Hugging Face dataset repo:
- Uploads only the following three files:
  - train.jsonl
  - test.jsonl
  - images.tar.gz
- Places them under `chart2code/` folder in the target dataset repo.
- Optionally creates the dataset repo if it does not exist.
- Optionally verifies the uploaded files exist on the remote.

Example:
  python upload_chart2code.py \
    --repo-id JiahuiGengNLP/MMCoIR \
    --local-root MMCoIR/chart2code \
    --create-repo \
    --check

Notes:
- Authentication: ensure you have logged in `huggingface-cli login`, or set HF_TOKEN env.
- This script uses huggingface_hub Python SDK (no need for Git/LFS).
"""

import os
import sys
import argparse
from typing import List, Tuple

try:
    from huggingface_hub import HfApi, create_repo, list_repo_files
except Exception as e:
    print("[ERROR] huggingface_hub not installed or import failed:", e)
    print("Please install: pip install -U huggingface_hub")
    sys.exit(1)


def ensure_repo(api: HfApi, repo_id: str, create: bool) -> None:
    if not create:
        return
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[OK] 数据集仓库可用: {repo_id}")
    except Exception as e:
        print(f"[WARN] 创建/确认仓库时遇到异常: {e}")


def verify_local_files(root: str, names: List[str]) -> List[Tuple[str, str]]:
    missing = []
    mapping: List[Tuple[str, str]] = []
    for name in names:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            missing.append(p)
        else:
            mapping.append((name, p))
    if missing:
        print("[ERROR] 以下本地文件缺失：")
        for m in missing:
            print(" -", m)
        sys.exit(1)
    return mapping


def upload_all(api: HfApi, repo_id: str, files: List[Tuple[str, str]], prefix: str = "chart2code", commit_prefix: str = "Add") -> None:
    failed: List[str] = []
    for basename, local_path in files:
        path_in_repo = f"{prefix}/{basename}"
        cm = f"{commit_prefix} {path_in_repo}"
        size = os.path.getsize(local_path)
        print(f"[UPLOAD] {local_path} ({size} bytes) -> {repo_id}:{path_in_repo}")
        try:
            HfApi().upload_file(
                path_or_fileobj=local_path,
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                commit_message=cm,
            )
            print(f"[OK] 已上传: {path_in_repo}")
        except Exception as e:
            print(f"[FAIL] 上传失败: {path_in_repo} -> {e}")
            failed.append(basename)
    if failed:
        print("[ERROR] 以下文件上传失败：", ", ".join(failed))
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
        print("[ERROR] 远端缺失以下路径：")
        for m in missing:
            print(" -", m)
        return False
    print("[OK] 远端校验通过，所有目标文件已存在。")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload chart2code files to Hugging Face dataset repo.")
    parser.add_argument("--repo-id", default="JiahuiGengNLP/MMCoIR", help="目标数据集仓库 ID")
    parser.add_argument("--local-root", default="MMCoIR/chart2code", help="本地文件根目录")
    parser.add_argument("--prefix", default="chart2code", help="仓库中的子目录前缀")
    parser.add_argument("--train", default="train.jsonl", help="训练集文件名")
    parser.add_argument("--test", default="test.jsonl", help="测试集文件名")
    parser.add_argument("--images-tar", default="images.tar.gz", help="图片打包文件名")
    parser.add_argument("--create-repo", action="store_true", help="若不存在则创建数据集仓库")
    parser.add_argument("--check", action="store_true", help="上传后校验远端是否存在文件")
    parser.add_argument("--dry-run", action="store_true", help="仅检查本地与打印计划，不实际上传")
    parser.add_argument("--commit-prefix", default="Add", help="提交信息前缀")
    args = parser.parse_args()

    api = HfApi()

    target_names = [args.train, args.test, args.images_tar]
    mapping = verify_local_files(args.local_root, target_names)

    print(f"[INFO] 目标仓库: {args.repo_id}")
    print(f"[INFO] 本地根目录: {args.local_root}")
    print(f"[INFO] 子目录前缀: {args.prefix}")
    for b, p in mapping:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        print(f" - {b}: {p} ({size} bytes)")

    if args.create_repo:
        ensure_repo(api, args.repo_id, create=True)

    expected_paths = [f"{args.prefix}/{b}" for b, _ in mapping]

    if args.dry_run:
        print("[DRY-RUN] 跳过上传，打印计划完成。")
        if args.check:
            verify_remote(args.repo_id, expected_paths)
        sys.exit(0)

    upload_all(api, args.repo_id, mapping, prefix=args.prefix, commit_prefix=args.commit_prefix)

    if args.check:
        ok = verify_remote(args.repo_id, expected_paths)
        if not ok:
            sys.exit(3)


if __name__ == "__main__":
    main()