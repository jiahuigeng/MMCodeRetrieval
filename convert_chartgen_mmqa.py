#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ChartGen-200K 转换为多模态代码检索训练/测试集（MM2Code 风格）。

- 训练集（split=train）：从 question_answers 中提取 user 文本作为查询（前缀加 <|image_1|>），agent 文本作为 pos_text。
- 测试集（split=test）：同样提取 user -> qry_text，agent -> tgt_text。
- 图片路径：相对路径，包含 chartgen/train/images/<filename> 或 chartgen/test/images/<filename>。
- 不进行图片拷贝与存在性检查，仅输出路径字符串。

输入支持 .jsonl 或 .parquet（parquet 需要 pandas/pyarrow）。
默认递归扫描目录 datasets/ChartGen-200K 里的 train/test 文件。

输出位置默认在 MMCoIR/chartgen 下生成 train_mmqa.jsonl 与 test_mmqa.jsonl。

使用示例：
- 读取目录（自动识别 train/test）：
  python convert_chartgen_mmqa.py --input-dir datasets/ChartGen-200K --output-dir MMCoIR/chartgen
- 读取单个文件（parquet）：
  python convert_chartgen_mmqa.py --input datasets/ChartGen-200K/data/train-00000-of-00001.parquet --output-dir MMCoIR/chartgen
- 读取单个文件（jsonl）：
  python convert_chartgen_mmqa.py --input datasets/ChartGen-200K/train-0000.jsonl --output-dir MMCoIR/chartgen
"""

import os
import sys
import json
import argparse
from typing import Optional, Tuple, List, Dict

CHARTGEN_DIRNAME = "chartgen"
IMG_SUBDIR = "images"
IMAGE_TOKEN = "<|image_1|>"


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
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


def read_parquet(path: str, limit: Optional[int] = None) -> List[Dict]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        raise RuntimeError("读取 parquet 需要 pandas/pyarrow，请先安装：pip install pandas pyarrow")
    df = pd.read_parquet(path)
    records: List[Dict] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def detect_split_from_path(image_path: str) -> Optional[str]:
    p = image_path.lower()
    if "/train/" in p or "\\train\\" in p:
        return "train"
    if "/test/" in p or "\\test\\" in p:
        return "test"
    return None


def make_rel_img_path(image_path: str, split: str) -> str:
    base = os.path.basename(image_path)
    return f"{CHARTGEN_DIRNAME}/{split}/{IMG_SUBDIR}/{base}"


def extract_user_agent(qa: Optional[object], prefer_last_agent: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """从 question_answers 列提取 user 与 agent 文本。
    支持 list[dict] 或 JSON 字符串；优先取第一个 user 与第一个/最后一个 agent。
    """
    if qa is None:
        return None, None
    user_text: Optional[str] = None
    agent_text: Optional[str] = None
    # 如果是字符串，尝试当作 JSON 解析
    try:
        qa_obj = json.loads(qa) if isinstance(qa, str) else qa
    except Exception:
        qa_obj = qa

    if isinstance(qa_obj, list):
        agent_candidates: List[str] = []
        for seg in qa_obj:
            sp = (seg.get("speaker") or seg.get("role") or "").lower()
            txt = seg.get("text") or seg.get("content") or ""
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    txt = ""
            if sp == "user" and user_text is None and txt:
                user_text = txt
            if sp in ("agent", "assistant") and txt:
                agent_candidates.append(txt)
        if agent_candidates:
            agent_text = agent_candidates[-1] if prefer_last_agent else agent_candidates[0]
    return user_text, agent_text


def convert_records(records: List[Dict], split_hint: Optional[str], prefer_last_agent: bool) -> Tuple[List[Dict], List[Dict]]:
    """将原始记录转换为训练/测试输出条目。
    返回： (train_items, test_items)
    """
    train_out: List[Dict] = []
    test_out: List[Dict] = []
    for rec in records:
        image_path = rec.get("image_path") or rec.get("image") or ""
        if not isinstance(image_path, str) or not image_path:
            # 某些 parquet 可能仅包含 image bytes；本脚本不处理图片字节，直接跳过
            continue
        split = split_hint or detect_split_from_path(image_path) or "train"
        rel_img = make_rel_img_path(image_path, split)
        user_text, agent_text = extract_user_agent(rec.get("question_answers"), prefer_last_agent=prefer_last_agent)
        if not user_text or not agent_text:
            # 缺少必要文本则跳过
            continue
        qry_with_token = f"{IMAGE_TOKEN}\n{user_text}"
        if split == "train":
            train_out.append({
                "qry": qry_with_token,
                "qry_image_path": rel_img,
                "pos_text": agent_text,
                "pos_image_path": "",
                "neg_text": "",
                "neg_image_path": ""
            })
        else:
            test_out.append({
                "qry_text": qry_with_token,
                "qry_img_path": rel_img,
                "tgt_text": [agent_text],
                "tgt_img_path": [""]
            })
    return train_out, test_out


def find_input_files(input_dir: str) -> Tuple[List[str], List[str]]:
    """递归扫描目录，返回 (train_files, test_files)，支持 .jsonl 与 .parquet。"""
    train_files: List[str] = []
    test_files: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            lower = fn.lower()
            if lower.endswith(".jsonl") or lower.endswith(".parquet"):
                full = os.path.join(root, fn)
                if "train" in lower:
                    train_files.append(full)
                elif "test" in lower:
                    test_files.append(full)
    return sorted(train_files), sorted(test_files)


def save_jsonl(items: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="将 ChartGen question_answers 转为 MM2Code 风格训练/测试 JSONL")
    parser.add_argument("--input", help="输入文件（.jsonl 或 .parquet）", default=None)
    parser.add_argument("--input-dir", help="输入目录（递归扫描 train/test 文件）", default="datasets/ChartGen-200K")
    parser.add_argument("--output-dir", help="输出根目录", default=os.path.join("MMCoIR", "chartgen"))
    parser.add_argument("--limit", type=int, default=None, help="每个文件最多读取的样本数")
    parser.add_argument("--prefer-last-agent", action="store_true", help="若有多轮回复，使用最后一个 agent 作为正样本")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集输入文件
    file_list: List[Tuple[str, Optional[str]]] = []
    if args.input:
        hint = "train" if "train" in os.path.basename(args.input).lower() else ("test" if "test" in os.path.basename(args.input).lower() else None)
        file_list.append((args.input, hint))
    else:
        trains, tests = find_input_files(args.input_dir)
        for p in trains:
            file_list.append((p, "train"))
        for p in tests:
            file_list.append((p, "test"))

    train_all: List[Dict] = []
    test_all: List[Dict] = []
    for path, split_hint in file_list:
        print(f"[INFO] 读取: {path} (split hint: {split_hint})")
        if path.lower().endswith(".jsonl"):
            recs = read_jsonl(path, limit=args.limit)
        else:
            recs = read_parquet(path, limit=args.limit)
        train_out, test_out = convert_records(recs, split_hint, prefer_last_agent=args.prefer_last_agent)
        print(f"[OK] 转换完成: train {len(train_out)} / test {len(test_out)}")
        train_all.extend(train_out)
        test_all.extend(test_out)

    train_path = os.path.join(args.output_dir, "train_mmqa.jsonl")
    test_path = os.path.join(args.output_dir, "test_mmqa.jsonl")
    save_jsonl(train_all, train_path)
    save_jsonl(test_all, test_path)

    print("\n==== 汇总 ====")
    print(f"训练集样本数: {len(train_all)} -> {train_path}")
    print(f"测试集样本数: {len(test_all)} -> {test_path}")
    print("注意：输出的图片路径为相对路径，如 'chartgen/train/images/<filename>'；脚本不拷贝/校验图片。")
    print("训练/评估时请将 image_root 设为 'MMCoIR'，并确保图片已放置到 'MMCoIR/chartgen/train/images/' 与 'MMCoIR/chartgen/test/images/'。")


if __name__ == "__main__":
    main()