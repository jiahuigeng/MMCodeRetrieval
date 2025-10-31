#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 OmniSVG 的 MMSVG-Icon / MMSVG-Illustration 数据集转换为“文本到代码(t2c)”训练/测试集。

输出：
- 训练 JSONL -> MMCoIR-train/<dataset>_t2c/train.jsonl
- 测试 JSONL -> MMCoIR-test/<dataset>_t2c/test.jsonl

图片路径：
- 本任务不涉及图片作为查询或目标，JSONL 中 `qry_img_path` 与 `tgt_img_path` 均为空。

JSONL 字段（与 format_checker.py 一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

字段含义（t2c）：
- qry_text / qry: 固定文本 prompt + 原始数据集的 `description` 字段内容
- qry_img_path / qry_image_path: 空字符串 ""
- tgt_text / pos_text: 目标代码（来自 `svg` 字段）
- tgt_img_path / pos_image_path: 空字符串 ""

数据来源：datasets/<dataset>/train.jsonl，要求包含 `id`, `svg`, `description` 字段。
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).parent
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DEFAULT_DATASETS = ["MMSVG-Icon", "MMSVG-Illustration"]

PROMPT_T2C = (
    "You are a helpful assistant that writes code from text descriptions.\n"
)


# ---------- IO helpers ----------

def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            data.append(obj)
            if limit is not None and len(data) >= limit:
                break
    return data


# ---------- JSONL builders ----------

def to_train_item(description: str, svg_code: str) -> Dict[str, Any]:
    return {
        "qry": f"{PROMPT_T2C}{str(description)}",
        "qry_image_path": "",
        "pos_text": str(svg_code),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(description: str, svg_code: str) -> Dict[str, Any]:
    return {
        "qry_text": f"{PROMPT_T2C}{str(description)}",
        "qry_img_path": "",
        "tgt_text": [str(svg_code)],
        "tgt_img_path": [""]
    }


# ---------- main logic ----------

def process_dataset(
    dataset: str,
    data_root: Path,
    train_root: Path,
    test_root: Path,
    train_count: int,
    test_count: int,
    seed: int,
) -> None:
    print(f"\n== Dataset (t2c): {dataset} ==")
    # 找到 datasets/<dataset>/train.jsonl（若不存在则回退为第一个 *.jsonl）
    ds_dir = data_root / dataset
    jsonl_path = ds_dir / "train.jsonl"
    if not jsonl_path.exists():
        cand = None
        for p in ds_dir.glob("*.jsonl"):
            cand = p
            break
        jsonl_path = cand if cand else jsonl_path
    if not jsonl_path or not jsonl_path.exists():
        print(f"[WARN] JSONL missing: {jsonl_path}")
        return

    print(f"[INFO] Reading JSONL: {jsonl_path}")
    records = read_jsonl(jsonl_path)

    # 收集 (id, description, svg)
    triples_all: List[Tuple[str, str, str]] = []  # (id, description, svg_code)
    for rec in records:
        sid = rec.get("id")
        description = rec.get("description")
        svg_code = rec.get("svg")
        if isinstance(sid, (str, int)) and isinstance(description, str) and isinstance(svg_code, str):
            triples_all.append((str(sid), description, svg_code))

    if not triples_all:
        print("[WARN] No usable records (need id, description, svg); skipping.")
        return

    total = len(triples_all)
    print(f"[INFO] Available records with id+description+svg: {total}")

    # 固定随机划分
    rnd = random.Random(seed)
    rnd.shuffle(triples_all)

    train_n = min(train_count, total)
    test_n = min(test_count, max(0, total - train_n))
    train_sel = triples_all[:train_n]
    test_sel = triples_all[train_n:train_n + test_n]
    print(f"[PLAN] Train: {len(train_sel)} | Test: {len(test_sel)}")

    # 构造 JSONL 项
    train_items = [to_train_item(desc, svg) for _sid, desc, svg in train_sel]
    test_items = [to_test_item(desc, svg) for _sid, desc, svg in test_sel]

    # 保存到 _t2c 目录
    out_test_json = test_root / f"{dataset}_t2c" / "test.jsonl"
    out_test_json.parent.mkdir(parents=True, exist_ok=True)
    out_train_json = None
    if train_n > 0:
        out_train_json = train_root / f"{dataset}_t2c" / "train.jsonl"
        out_train_json.parent.mkdir(parents=True, exist_ok=True)

    if out_train_json is not None:
        print(f"[SAVE] Train JSONL -> {out_train_json}")
        with out_train_json.open("w", encoding="utf-8") as f:
            for it in train_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    else:
        print("[SAVE] Skipped train JSONL (train-count=0)")

    print(f"[SAVE] Test JSONL  -> {out_test_json}")
    with out_test_json.open("w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print("[DONE] OmniSVG t2c split completed.")
    if out_train_json is not None:
        print(f"Train JSONL: {out_train_json}")
    else:
        print("Train JSONL: (skipped)")
    print(f"Test JSONL:  {out_test_json}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert MMSVG datasets to t2c train/test JSONL")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Input datasets root (JSONL under <dataset>/train.jsonl)")
    parser.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="Output root for train (<root>/<dataset>_t2c/train.jsonl)")
    parser.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="Output root for test (<root>/<dataset>_t2c/test.jsonl)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset names to process")
    parser.add_argument("--train-count", type=int, default=100_000, help="Number of train samples per dataset")
    parser.add_argument("--test-count", type=int, default=2_000, help="Number of test samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    print(f"Data root: {data_root} {'(exists)' if data_root.exists() else '(missing)'}")
    print(f"Train root: {train_root}")
    print(f"Test root:  {test_root}")

    for ds in args.datasets:
        process_dataset(
            dataset=ds,
            data_root=data_root,
            train_root=train_root,
            test_root=test_root,
            train_count=args.train_count,
            test_count=args.test_count,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()