#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 SD122025/ChartGen-200K 转换为基于 QA 的多模态检索格式，
在同一目标文件夹下生成 train_qa.jsonl 与 test_qa.jsonl。

- Input: 本地数据快照位于 datasets/ChartGen-200K（通过 download_chartgen.py 下载）或单文件 .jsonl/.parquet
- Output: JSONL 文件位于 MMCoIR/chartgen/{train_qa.jsonl, test_qa.jsonl}
- Images: 期望位于输出目录的 train/images 与 test/images，相对路径指向这些目录；脚本不复制、不校验图片存在性。

训练集 JSONL 字段（与规范一致）:
- qry: 带图像 token 的查询文本（<|image_1|> + 问句）
- qry_image_path: 查询图片相对路径（chartgen/train/images/<filename>）
- pos_text: 正样本文本（答案文本）
- pos_image_path: 正样本图片路径（本任务无 -> 空字符串）
- neg_text: 负样本文本（无 -> 空字符串）
- neg_image_path: 负样本图片路径（无 -> 空字符串）

测试集 JSONL 字段（与规范一致）:
- qry_text: 带图像 token 的查询文本（<|image_1|> + 问句）
- qry_img_path: 查询图片相对路径（chartgen/<split>/images/<filename>）
- tgt_text: 目标文本列表（List[str]，答案文本）
- tgt_img_path: 目标图片路径列表（List[str]；本任务无目标图像 -> [""]）

额外说明：原始 test split 可参与生成，但若未显式标注，则从 image_path 自动检测 split；默认仅使用每张图最后 K=2 组 (question, answer)。
"""

import os
import sys
import json
import argparse
from typing import Optional, Tuple, List, Dict, Any

try:
    import pandas as pd
except Exception:
    pd = None

CHARTGEN_DIRNAME = "chartgen"
IMG_SUBDIR = "images"
IMAGE_TOKEN = "<|image_1|>"

DEFAULT_INPUT_DIR = os.path.join("datasets", "ChartGen-200K")
DEFAULT_OUTPUT_DIR = os.path.join("MMCoIR", "chartgen")

# --- 路径与 split ---
def detect_split_from_path(image_path: str) -> Optional[str]:
    p = str(image_path).lower()
    if "/train/" in p or "\\train\\" in p:
        return "train"
    if "/test/" in p or "\\test\\" in p:
        return "test"
    return None


def make_rel_img_path(image_path: str, split: str) -> str:
    base = os.path.basename(str(image_path))
    return f"{CHARTGEN_DIRNAME}/{split}/{IMG_SUBDIR}/{base}"


# --- 文件读写 ---
def save_jsonl(items: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    return records


def read_parquet(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if pd is None:
        print(f"[WARN] pandas 未安装，无法读取 parquet: {path}")
        return []
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] 读取 parquet 失败: {path}: {e}")
        return []
    out: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        if limit is not None and len(out) >= limit:
            break
        rec = {}
        # 尝试标准字段
        for k in ["image_path", "image", "question_answers", "qa_cell", "utterances", "dialog"]:
            if k in df.columns:
                rec[k] = row.get(k, None)
        # 兜底：若存在 image 列是 dict（包含 path/bytes），保留原结构
        out.append(rec)
    return out


# --- QA 解析 ---
def _norm_key(k: Any) -> str:
    return str(k).strip().lower().replace(" ", "")


def _norm_speaker(s: Any) -> str:
    return str(s).strip().lower()


def _extract_pairs_from_segments(segments: List[Dict[str, Any]], debug: bool = False) -> List[Tuple[str, str]]:
    q_roles = {"user", "human"}
    a_roles = {"agent", "assistant", "bot"}
    pairs: List[Tuple[str, str]] = []
    pending_q: Optional[str] = None

    for i, seg in enumerate(segments):
        speaker = _norm_speaker(seg.get("speaker", seg.get("role", "")))
        text = seg.get("text")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if debug:
            print(f"[DBG] seg[{i}] speaker={speaker} text={text[:60]}")
        if speaker in q_roles:
            pending_q = text
        elif speaker in a_roles and pending_q:
            pairs.append((pending_q, text))
            pending_q = None
        else:
            # 其他角色忽略
            pass
    return pairs


def parse_question_answers(raw: Any, debug: bool = False) -> List[Tuple[str, str]]:
    """将多种 question_answers 表达形式解析为 (question, answer) 对列表。"""
    if raw is None:
        return []
    # 字符串 -> 可能是 JSON
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except Exception:
            return []
        raw = obj
    # 字典 -> 优先找 utterances/dialog/question_answers/qa_cell
    if isinstance(raw, dict):
        keys = { _norm_key(k): k for k in raw.keys() }
        for key_norm in ["utterances", "dialog", "question_answers", "qacell"]:
            if key_norm in keys:
                segments = raw[keys[key_norm]]
                if isinstance(segments, list):
                    return _extract_pairs_from_segments(segments, debug=debug)
        # 单问答键（不常见），尝试直接拼一对
        q = raw.get("question")
        a = raw.get("answer")
        if isinstance(q, str) and isinstance(a, str):
            return [(q, a)]
        return []
    # 列表 -> 视为 utterances 列表
    if isinstance(raw, list):
        return _extract_pairs_from_segments(raw, debug=debug)
    return []


# --- 转换逻辑 ---
def convert_records(records: List[Dict[str, Any]], split_hint: Optional[str], use_last_k: int, debug: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_out: List[Dict[str, Any]] = []
    test_out: List[Dict[str, Any]] = []
    for rec in records:
        image_path = rec.get("image_path") or rec.get("image") or ""
        if not isinstance(image_path, str) or not image_path:
            # 某些 parquet 可能仅包含 image bytes；本脚本不处理图片字节，直接跳过
            if debug:
                print("[DBG] 跳过记录：无有效 image_path")
            continue
        split = split_hint or detect_split_from_path(image_path) or "train"
        rel_img = make_rel_img_path(image_path, split)
        qa_raw = rec.get("question_answers") or rec.get("qa_cell") or rec.get("utterances") or rec.get("dialog")
        pairs = parse_question_answers(qa_raw, debug=debug)
        if not pairs:
            if debug:
                print("[DBG] 未解析到 QA 对，跳过该条")
            continue
        if use_last_k and use_last_k > 0:
            pairs = pairs[-use_last_k:]
        for (q_text, a_text) in pairs:
            q_text = str(q_text).strip()
            a_text = str(a_text).strip()
            if not q_text or not a_text:
                continue
            qry_with_token = f"{IMAGE_TOKEN}\n{q_text}"
            if split == "train":
                train_out.append({
                    "qry": qry_with_token,
                    "qry_image_path": rel_img,
                    "pos_text": a_text,
                    "pos_image_path": "",
                    "neg_text": "",
                    "neg_image_path": ""
                })
            else:
                test_out.append({
                    "qry_text": qry_with_token,
                    "qry_img_path": rel_img,
                    "tgt_text": [a_text],
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
                else:
                    # 未标注，默认按 train 处理
                    train_files.append(full)
    return sorted(train_files), sorted(test_files)


def main():
    parser = argparse.ArgumentParser(description="将 ChartGen question_answers 转为 QA 检索风格训练/测试 JSONL")
    parser.add_argument("--input", help="输入文件（.jsonl 或 .parquet）", default=None)
    parser.add_argument("--input-dir", help="输入目录（递归扫描 train/test 文件）", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", help="输出根目录", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="每个文件最多读取的样本数")
    parser.add_argument("--use-last-k", type=int, default=2, help="每张图使用末尾 K 组 QA 对")
    parser.add_argument("--debug", action="store_true", help="打印解析调试信息")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    file_list: List[Tuple[str, Optional[str]]] = []
    if args.input:
        base = os.path.basename(args.input).lower()
        hint = "train" if "train" in base else ("test" if "test" in base else None)
        file_list.append((args.input, hint))
    else:
        trains, tests = find_input_files(args.input_dir)
        for p in trains:
            file_list.append((p, "train"))
        for p in tests:
            file_list.append((p, "test"))

    train_all: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []
    for path, split_hint in file_list:
        print(f"[INFO] 读取: {path} (split hint: {split_hint})")
        if path.lower().endswith(".jsonl"):
            recs = read_jsonl(path, limit=args.limit)
        else:
            recs = read_parquet(path, limit=args.limit)
        train_out, test_out = convert_records(recs, split_hint, use_last_k=args.use_last_k, debug=args.debug)
        print(f"[OK] 转换完成: train {len(train_out)} / test {len(test_out)}")
        train_all.extend(train_out)
        test_all.extend(test_out)

    train_path = os.path.join(args.output_dir, "train_qa.jsonl")
    test_path = os.path.join(args.output_dir, "test_qa.jsonl")
    save_jsonl(train_all, train_path)
    save_jsonl(test_all, test_path)

    print("\n==== 汇总 ====")
    print(f"训练集样本数: {len(train_all)} -> {train_path}")
    print(f"测试集样本数: {len(test_all)} -> {test_path}")
    print("注意：输出的图片路径为相对路径，如 'chartgen/train/images/<filename>'；脚本不拷贝/校验图片。")
    print("训练/评估时请将 image_root 设为 'MMCoIR'，并确保图片已放置到 'MMCoIR/chartgen/train/images/' 与 'MMCoIR/chartgen/test/images/'。")


if __name__ == "__main__":
    main()