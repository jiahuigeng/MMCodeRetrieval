#!/usr/bin/env python3
"""
将 SD122025/ChartGen-200K 转换为基于 QA 的多模态检索格式，
在同一目标文件夹下生成 train_qa.jsonl 与 test_qa.jsonl。

- Input: 本地数据位于 datasets/ChartGen-200K （通过 download_chartgen.py 下载）
- Output: JSONL 文件位于 MMCoIR/chartgen/{train_qa.jsonl, test_qa.jsonl}
- Images: 仅写出相对路径，不复制图片，不校验图片存在性。

构建规则：
- 仅处理 train split 行；从训练集中按顺序抽取前 N=2000 张图片作为测试集，其余作为训练集。
- 每张图片从 question_answers（或嵌套的 utterances）里提取成对的问答（user -> agent），优先选择靠后的两个问答对。
- 对每个问答对构建一个样本：
  - 训练集字段：
    - qry: "<|image_1|>\n" + question
    - qry_image_path: 相对路径 "chartgen/train/images/<filename>"
    - pos_text: answer
    - pos_image_path: ""
    - neg_text: ""
    - neg_image_path: ""
  - 测试集字段：
    - qry_text: "<|image_1|>\n" + question
    - qry_img_path: 相对路径 "chartgen/train/images/<filename>"
    - tgt_text: [answer]
    - tgt_img_path: [""]
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "ChartGen-200K"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MMCoIR" / "chartgen"
CHARTGEN_PREFIX = "chartgen"
IMAGES_SUBDIR = "images"
TRAIN_SUBDIR = "train"
IMAGE_TOKEN = "<|image_1|>"


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_images_dir = out_dir / TRAIN_SUBDIR / IMAGES_SUBDIR
    train_images_dir.mkdir(parents=True, exist_ok=True)
    # 不复制图片，仅确保路径存在性（可选）
    return out_dir, train_images_dir


def find_parquet_files(root: Path) -> List[Path]:
    parquet_files: List[Path] = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                parquet_files.append(Path(r) / fn)
    return sorted(parquet_files)


def parse_qa_segments(qa_cell: object) -> List[Dict[str, str]]:
    """将 question_answers 单元解析为 segment 列表。
    支持：
    - list[dict] 直接为对话切片
    - dict 包含 key 'utterances' 或 'dialogs' 为列表
    - str 为 JSON 字符串，上述两种之一
    返回统一结构：[{"speaker": ..., "text": ...}, ...]
    """
    try:
        qa_obj = json.loads(qa_cell) if isinstance(qa_cell, str) else qa_cell
    except Exception:
        qa_obj = qa_cell

    segments: List[Dict[str, str]] = []
    if isinstance(qa_obj, list):
        for seg in qa_obj:
            sp = (seg.get("speaker") or seg.get("role") or "").lower()
            txt = seg.get("text") or seg.get("content") or ""
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    txt = ""
            segments.append({"speaker": sp, "text": txt})
    elif isinstance(qa_obj, dict):
        cand = None
        if isinstance(qa_obj.get("utterances"), list):
            cand = qa_obj.get("utterances")
        elif isinstance(qa_obj.get("dialogs"), list):
            cand = qa_obj.get("dialogs")
        if cand is not None:
            for seg in cand:
                sp = (seg.get("speaker") or seg.get("role") or "").lower()
                txt = seg.get("text") or seg.get("content") or ""
                if not isinstance(txt, str):
                    try:
                        txt = str(txt)
                    except Exception:
                        txt = ""
                segments.append({"speaker": sp, "text": txt})
    return segments


def extract_qa_pairs(qa_cell: object) -> List[Tuple[str, str]]:
    """从 QA 单元中提取成对的 (question, answer)。
    逻辑：遇到 user 记为待匹配的 question，遇到 agent/assistant 则与最近的 question 匹配成对。
    """
    segs = parse_qa_segments(qa_cell)
    pairs: List[Tuple[str, str]] = []
    pending_q: Optional[str] = None
    for seg in segs:
        sp = seg.get("speaker", "").lower()
        txt = seg.get("text", "")
        if not isinstance(txt, str):
            try:
                txt = str(txt)
            except Exception:
                txt = ""
        if sp == "user":
            if txt:
                pending_q = txt
        elif sp in ("agent", "assistant"):
            if pending_q and txt:
                pairs.append((pending_q, txt))
                pending_q = None
    return pairs


def to_train_item(question: str, answer: str, rel_img_path: str) -> dict:
    return {
        "qry": f"{IMAGE_TOKEN}\n{question}",
        "qry_image_path": rel_img_path,
        "pos_text": str(answer),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(question: str, answer: str, rel_img_path: str) -> dict:
    return {
        "qry_text": f"{IMAGE_TOKEN}\n{question}",
        "qry_img_path": rel_img_path,
        "tgt_text": [str(answer)],
        "tgt_img_path": [""],
    }


def convert_chartgen_qa(
    input_root: Path,
    output_dir: Path,
    test_count: int = 2000,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    pairs_per_image: int = 2,
):
    out_dir, train_images_dir = ensure_dirs(output_dir)
    parquet_files = find_parquet_files(input_root)
    if not parquet_files:
        print(f"[ERROR] No parquet files found under {input_root}")
        return

    print(f"[INFO] Found {len(parquet_files)} parquet files")

    # 每张图片对应一组训练样本（若可用则为若干条）
    examples_per_image: List[List[dict]] = []

    for pq in parquet_files:
        print(f"[INFO] Reading {pq} ...")
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[WARN] Failed to read {pq}: {e}")
            continue

        cols = df.columns.tolist()
        if "image_path" not in cols:
            print(f"[WARN] 'image_path' missing in {pq}; skipping this file")
            continue
        qa_col = None
        for cand in ["question_answers", "qa", "utterances", "dialogs"]:
            if cand in cols:
                qa_col = cand
                break
        if qa_col is None:
            print(f"[WARN] QA column missing in {pq}; expected one of question_answers/qa/utterances/dialogs")
            continue

        df = df[["image_path", qa_col]]

        for idx, row in df.iterrows():
            try:
                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    continue
                # 仅使用 train split；原 test split 不参与生成
                split = "train" if "train/" in image_path else None
                if split is None:
                    continue

                qa_pairs = extract_qa_pairs(row.get(qa_col))
                if not qa_pairs:
                    continue
                # 选择靠后的若干问答对
                use_pairs = qa_pairs[-pairs_per_image:] if len(qa_pairs) >= pairs_per_image else qa_pairs

                basename = Path(image_path).name
                rel_img_path = f"{CHARTGEN_PREFIX}/{TRAIN_SUBDIR}/{IMAGES_SUBDIR}/{basename}"

                group_items: List[dict] = []
                for q, a in use_pairs:
                    group_items.append(to_train_item(q, a, rel_img_path))
                if group_items:
                    examples_per_image.append(group_items)
            except Exception as e:
                print(f"[WARN] Failed processing row {idx} in {pq}: {e}")
                continue

    # 按图片切分：前 test_count 张图片的样本作为测试集，其余作为训练集
    total_images = len(examples_per_image)
    selected_for_test = examples_per_image[:test_count]
    remaining_for_train = examples_per_image[test_count:]

    # 将训练样本映射为测试格式
    test_items: List[dict] = []
    for group in selected_for_test:
        for it in group:
            test_items.append({
                "qry_text": it["qry"],
                "qry_img_path": it["qry_image_path"],
                "tgt_text": [it["pos_text"]],
                "tgt_img_path": [""],
            })

    train_items: List[dict] = [it for group in remaining_for_train for it in group]

    print(f"[INFO] Built image groups: {total_images}; test images: {len(selected_for_test)} -> test items: {len(test_items)}; remaining train images: {len(remaining_for_train)} -> train items: {len(train_items)}")

    # 可选限制
    if limit_train is not None:
        train_items = train_items[:limit_train]
    if limit_test is not None:
        test_items = test_items[:limit_test]

    # 保存输出
    train_out = out_dir / "train_qa.jsonl"
    test_out = out_dir / "test_qa.jsonl"

    print(f"[INFO] Saving train set ({len(train_items)}) to {train_out}")
    with open(train_out, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Saving test set ({len(test_items)}) to {test_out}")
    with open(test_out, "w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] QA conversion completed.")
    print(f"Train images directory (for path reference): {train_images_dir}")
    print(f"Train QA JSONL: {train_out}")
    print(f"Test QA JSONL: {test_out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert ChartGen-200K QA to MMCoIR JSONL (test carved from train; per image take last 2 QA pairs; paths use chartgen/train/images)"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Input dataset root (local snapshot)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory under MMCoIR/chartgen")
    parser.add_argument("--test-count", type=int, default=2000, help="Number of images to carve out from train as test (default 2000)")
    parser.add_argument("--pairs-per-image", type=int, default=2, help="Number of QA pairs to take per image (use last K)")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional limit for train items")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional limit for test items")
    args = parser.parse_args()

    convert_chartgen_qa(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        test_count=args.test_count,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        pairs_per_image=args.pairs_per_image,
    )


if __name__ == "__main__":
    main()