#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChartGen-200K 数据集下载脚本

- 默认从 Hugging Face 下载数据集：SD122025/ChartGen-200K
- 将快照内容保存到本地：datasets/ChartGen-200K/
- 自动汇总文件结构（图片与 parquet 统计）
- 可选：尝试将 parquet 转为 JSONL（需要 pandas/pyarrow）

数据来源（参考）：
- Hugging Face 数据集页面：SD122025/ChartGen-200K
  https://huggingface.co/datasets/SD122025/ChartGen-200K

注意：
- Windows 下建议禁用符号链接复制（默认），避免权限问题。
- 首次下载会较慢，数据量较大，请确保磁盘空间充足。
"""
import argparse
import os
import sys
import json
from typing import List, Optional


def try_import_hf_hub():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except Exception as e:
        print("[ERROR] 未找到 huggingface_hub。请先安装：pip install huggingface_hub")
        print(f"详细错误：{e}")
        return None


def summarize_dir(root_dir: str) -> dict:
    img_exts = (".jpg", ".jpeg", ".png", ".webp")
    image_files = []
    parquet_files = []
    for r, _dirs, files in os.walk(root_dir):
        for fn in files:
            lf = fn.lower()
            fp = os.path.join(r, fn)
            if lf.endswith(img_exts):
                image_files.append(fp)
            elif lf.endswith(".parquet"):
                parquet_files.append(fp)
    summary = {
        "root": root_dir,
        "image_count": len(image_files),
        "parquet_count": len(parquet_files),
        "image_samples": image_files[:5],
        "parquet_samples": parquet_files[:5],
    }
    return summary


def print_summary(summary: dict):
    print("\n====== 下载内容汇总 ======")
    print(f"目标目录: {summary['root']}")
    print(f"图片数量: {summary['image_count']}")
    print(f"parquet 数量: {summary['parquet_count']}")
    if summary["image_samples"]:
        print("示例图片:")
        for p in summary["image_samples"]:
            print(" -", p)
    if summary["parquet_samples"]:
        print("示例 parquet:")
        for p in summary["parquet_samples"]:
            print(" -", p)
    print("========================\n")


def parquet_to_jsonl_if_possible(parquet_paths: List[str], out_dir: str, limit: Optional[int] = None):
    try:
        import pandas as pd
    except Exception as e:
        print("[WARN] 未安装 pandas，跳过 JSONL 生成。安装：pip install pandas pyarrow")
        print(f"详细错误：{e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    for pq in parquet_paths:
        base = os.path.splitext(os.path.basename(pq))[0]
        out_path = os.path.join(out_dir, f"{base}.jsonl")
        try:
            # 尝试读取常见列；若不存在则自动读取全部列
            wanted_cols = [
                "id", "code", "image", "image_path", "summary",
                "csv", "doctags", "question_answers"
            ]
            df = pd.read_parquet(pq)
            # 只保留存在的列
            cols = [c for c in wanted_cols if c in df.columns]
            if cols:
                df = df[cols]
            if limit is not None:
                df = df.head(limit)

            with open(out_path, "w", encoding="utf-8") as f:
                for rec in df.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[OK] 生成 JSONL: {out_path} (记录数: {len(df)})")
        except Exception as e:
            print(f"[WARN] 读取/转换失败: {pq} -> {out_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="下载 ChartGen-200K 数据集到本地并汇总结构")
    parser.add_argument("--repo-id", default="SD122025/ChartGen-200K", help="Hugging Face 数据集仓库 ID")
    parser.add_argument("--root", default="datasets", help="数据集根目录（将创建/使用该目录）")
    parser.add_argument("--dataset-name", default="ChartGen-200K", help="保存目录名（位于 root 下）")
    parser.add_argument("--allow", nargs="*", default=["train/*", "*.parquet"], help="允许下载的路径模式（可多项）")
    parser.add_argument("--no-symlinks", action="store_true", default=True, help="禁用符号链接复制（Windows 推荐）")
    parser.add_argument("--make-jsonl", action="store_true", help="尝试将 parquet 转为 JSONL（需 pandas/pyarrow）")
    parser.add_argument("--jsonl-limit", type=int, default=None, help="JSONL 生成的最大记录数（默认全部）")
    parser.add_argument("--token", default=None, help="可选：Hugging Face 访问令牌（若数据集受限）")

    args = parser.parse_args()

    target_dir = os.path.join(args.root, args.dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download = try_import_hf_hub()
    if snapshot_download is None:
        sys.exit(1)

    print("开始下载数据集快照...")
    try:
        # 下载整个数据集快照到目标目录
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=not args.no_symlinks,
            allow_patterns=args.allow if args.allow else None,
            token=args.token,
        )
        print(f"[OK] 已保存快照到: {target_dir}")
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        sys.exit(1)

    # 汇总结构
    summary = summarize_dir(target_dir)
    print_summary(summary)

    # 可选：生成 JSONL
    if args.make_jsonl:
        if summary.get("parquet_count", 0) == 0:
            print("[WARN] 未发现 parquet 文件，跳过 JSONL 生成。")
        else:
            parquet_to_jsonl_if_possible(summary["parquet_samples"] if summary["parquet_samples"] else [], out_dir=target_dir, limit=args.jsonl_limit)
            # 若只展示了样例，可扩展为全量：遍历所有 parquet_paths
            # 这里为避免巨大开销，仅示例生成。可根据需要改为使用 summary 中的全部 parquet 文件。

    print("完成。")


if __name__ == "__main__":
    main()