#!/usr/bin/env python3
"""
任务说明（t2c 版本，仅训练集）
- 将 datasets/ChartGen-200K 下的 Parquet 数据转换为文本到代码（text-to-code）训练集 JSONL。
- 随机采样：默认训练 100000（可通过参数调整）。
- 输出：
  - 训练集 JSONL -> `MMCoIR-train/ChartGen_t2c/train.jsonl`
- 本任务不涉及图片：所有 *_img_path / *_image_path 字段均为空字符串，占位一致；不会复制任何图片。

JSONL 字段约定（与项目格式一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}

字段映射（ChartGen-200K）：
- 训练：`qry` <- summary；`pos_text` <- code；其余 *_image_* 为空，neg_* 为空
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "ChartGen-200K" / "data"
TRAIN_ROOT_DEFAULT = REPO_ROOT / "MMCoIR-train" / "ChartGen_t2c"
TRAIN_SPLIT = "train"


def find_parquet_files(root: Path) -> List[Path]:
    parquet_files: List[Path] = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                parquet_files.append(Path(r) / fn)
    return sorted(parquet_files)


def to_train_item(summary: str, code: str) -> dict:
    return {
        "qry": str(summary),
        "qry_image_path": "",
        "pos_text": str(code),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ChartGen-200K text-to-code: generate train JSONL only (no images)"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--train-root", default=str(TRAIN_ROOT_DEFAULT), help="Output root for train JSONL")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of train samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    train_root = Path(args.train_root)

    print(f"[INFO] Scanning parquet under {input_root}")
    parquets = find_parquet_files(input_root)
    if not parquets:
        raise SystemExit(f"No parquet files found under {input_root}")
    print(f"[INFO] Found {len(parquets)} parquet files")

    train_candidates: List[Tuple[str, str]] = []  # (summary, code)

    for pq in parquets:
        print(f"[INFO] Reading {pq} ...")
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[WARN] Failed to read {pq}: {e}")
            continue

        cols = df.columns.tolist()
        if "image_path" not in cols:
            print(f"[WARN] 'image_path' missing in {pq}; skip")
            continue
        keep_cols = [c for c in ["image_path", "summary", "code"] if c in cols]
        if "summary" not in keep_cols or "code" not in keep_cols:
            print(f"[WARN] 'summary' or 'code' missing in {pq}; skip")
            continue
        df = df[keep_cols]

        for _idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    continue
                # 处理 summary/code 的缺失与 NaN/None 情况，确保落盘时为字符串且非空
                summary_raw = row.get("summary", None)
                code_raw = row.get("code", None)
                # pandas NaN 检测
                try:
                    import pandas as _pd
                    is_summary_nan = _pd.isna(summary_raw)
                    is_code_nan = _pd.isna(code_raw)
                except Exception:
                    is_summary_nan = False
                    is_code_nan = False
                if summary_raw is None or is_summary_nan:
                    continue
                if code_raw is None or is_code_nan:
                    continue
                summary = str(summary_raw).strip()
                # 排除字符串形式的 "None"/"nan"
                if not summary or summary.lower() in ("none", "nan"):
                    continue
                code = str(code_raw)

                # 仅处理训练集
                if f"{TRAIN_SPLIT}/" in image_path:
                    train_candidates.append((summary, code))
            except Exception as e:
                print(f"[WARN] Row parse error in {pq}: {e}")
                continue

    print(f"[INFO] Candidates: train={len(train_candidates)}")

    # Random sample
    random.seed(args.seed)
    train_n = min(args.train_count, len(train_candidates))
    train_sample = random.sample(train_candidates, train_n)

    # Convert to JSON items
    train_items = [to_train_item(summary=s, code=c) for (s, c) in train_sample]

    # Save JSONL（目录后缀为 _t2c，文件名无后缀）
    train_json = train_root / "train.jsonl"
    train_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving train ({len(train_items)}) -> {train_json}")
    with open(train_json, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # 不再生成测试集 JSONL

    print("[DONE] ChartGen t2c train-only conversion completed.")
    print(f"Train JSONL: {train_json}")


if __name__ == "__main__":
    main()