#!/usr/bin/env python3
"""
递归上传 MMCoIR/chartgen 下的全部文件到 Hugging Face 数据集仓库。
- 递归上传目录内容，保持相对目录结构到远端 `chartgen/`。
- 可选：创建仓库、Dry-Run、上传后校验远端文件、忽略模式。
示例：
  python upload_chartgen.py --repo-id JiahuiGengNLP/MMCoIR --local-root MMCoIR/chartgen --create-repo --check
"""

import os
import sys
import argparse
from typing import List

try:
    from huggingface_hub import HfApi, create_repo, list_repo_files
except Exception as e:
    print("[ERROR] 需要安装 huggingface_hub:", e)
    print("pip install -U huggingface_hub")
    sys.exit(1)


def ensure_repo(repo_id: str, create: bool) -> None:
    if not create:
        return
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[OK] 数据集仓库可用: {repo_id}")
    except Exception as e:
        print(f"[WARN] 创建/确认仓库异常: {e}")


def collect_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        print(f"[ERROR] 本地目录不存在: {root}")
        sys.exit(1)
    rels: List[str] = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(r, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            rels.append(rel)
    if not rels:
        print(f"[ERROR] 目录为空，无可上传文件: {root}")
        sys.exit(1)
    return sorted(rels)


def main() -> None:
    ap = argparse.ArgumentParser(description="递归上传 chartgen 所有文件到 HF 数据集仓库")
    ap.add_argument("--repo-id", default="JiahuiGengNLP/MMCoIR", help="目标数据集仓库 ID")
    ap.add_argument("--local-root", default="MMCoIR/chartgen", help="本地根目录")
    ap.add_argument("--prefix", default="chartgen", help="远端子目录前缀")
    ap.add_argument("--create-repo", action="store_true", help="若不存在则创建数据集仓库")
    ap.add_argument("--check", action="store_true", help="上传后校验远端是否存在文件")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际上传")
    ap.add_argument("--commit-prefix", default="Add", help="提交信息前缀")
    ap.add_argument("--ignore", action="append", default=[], help="追加忽略模式（glob），可重复")
    args = ap.parse_args()

    api = HfApi()
    defaults_ignore = [".git/*", "__pycache__/*", ".ipynb_checkpoints/*"]
    ignore_patterns = defaults_ignore + args.ignore

    print(f"[INFO] 目标仓库: {args.repo_id}")
    print(f"[INFO] 本地根目录: {args.local_root}")
    print(f"[INFO] 子目录前缀: {args.prefix}")

    rels = collect_files(args.local_root)
    total_bytes = 0
    for rel in rels:
        p = os.path.join(args.local_root, rel.replace("/", os.sep))
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            pass
    print(f"[PLAN] 将上传 {len(rels)} 个文件，共约 {total_bytes} bytes")

    if args.dry_run:
        preview = rels[:20]
        for rel in preview:
            print(" -", rel)
        if len(rels) > len(preview):
            print(f" ... 其余 {len(rels)-len(preview)} 个文件省略")
        sys.exit(0)

    ensure_repo(args.repo_id, args.create_repo)

    cm = f"{args.commit_prefix} {args.prefix}/"
    print(f"[UPLOAD] {args.local_root} -> {args.repo_id}:{args.prefix}/ (递归)")
    try:
        api.upload_folder(
            path_dir=args.local_root,
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo=args.prefix,
            ignore_patterns=ignore_patterns,
            commit_message=cm,
        )
        print("[OK] 目录上传完成。")
    except Exception as e:
        print(f"[FAIL] 目录上传失败: {e}")
        sys.exit(2)

    if args.check:
        print("[CHECK] 验证远端仓库文件...")
        try:
            files = list_repo_files(args.repo_id, repo_type="dataset")
        except Exception as e:
            print(f"[WARN] 无法列出远端文件: {e}")
            sys.exit(3)
        expected = [f"{args.prefix}/{rel}" for rel in rels]
        missing = [p for p in expected if p not in files]
        if missing:
            print("[WARN] 远端缺失部分路径，示例：")
            for m in missing[:20]:
                print(" -", m)
            print(f"[WARN] 共缺失 {len(missing)} 项。请稍后重试或检查忽略模式。")
        else:
            print("[OK] 远端校验通过，所有目标文件已存在。")


if __name__ == "__main__":
    main()