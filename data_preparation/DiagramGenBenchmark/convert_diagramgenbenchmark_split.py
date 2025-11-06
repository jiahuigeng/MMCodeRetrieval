#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 DiagramAgent/DiagramGenBenchmark 的 DiagramCoding 子集转换为 MMCoIR 测试集格式。

输入：
- 默认从 `datasets/DiagramGenBenchmark/DiagramCoding.json` 读取（可通过参数指定）。
- 图片默认位于 `datasets/DiagramGenBenchmark/images/`（可通过 `--src-images-dir` 指定）。

输出：
- 生成测试集到 `MMCoIR-test/DiagramGenBenchmark_i2c/test.jsonl`。
- JSONL 字段遵循规范：
  - `qry_text`: 查询文本（包含 `<|image_1|>` 前缀）
  - `qry_img_path`: 相对图片路径（`images/DiagramGenBenchmark/images/<filename>`）
  - `tgt_text`: 目标文本列表（代码字符串）
  - `tgt_img_path`: 目标图片路径列表（本任务无 -> [""]）

图片复制（默认开启）：
- 实际保存到 `MMCoIR-test/images/DiagramGenBenchmark/images/`；
- JSONL 中写入的路径以 `images/DiagramGenBenchmark/images/<filename>` 开始；
- 使用 `--no-copy-images` 关闭复制；`--overwrite-images` 覆盖已存在的图片。

示例用法：
  python DiagramGenBenchmark/convert_diagramgenbenchmark_split.py \
    --input-json datasets/DiagramGenBenchmark/DiagramCoding.json \
    --out-dir MMCoIR-test/DiagramGenBenchmark_i2c \
    --src-images-dir datasets/DiagramGenBenchmark/images

可选参数：
- `--limit N`           仅处理前 N 条样本
- `--src-images-dir`    源图片目录（默认 datasets/DiagramGenBenchmark/images）
- `--overwrite-images`  覆盖已存在的目标图片
- `--no-copy-images`    不复制图片（默认复制）
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


DATASET_NAME = "DiagramGenBenchmark"
IMG_SUBDIR = "images"
REL_IMG_PREFIX = "images/DiagramGenBenchmark/images"
IMAGE_TOKEN = "<|image_1|>"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> List[Dict[str, Any]]:
    print(f"[LOAD] {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # 允许 dict 形式（用 data/items/records 包裹）
        for key in ("data", "items", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    else:
        return []


def normalize_image_token(text: str) -> str:
    if text is None:
        text = ""
    # 移除可能的占位符
    t = str(text).replace("<image>", "").strip()
    # 统一加入 IMAGE_TOKEN 换行
    return f"{IMAGE_TOKEN}\n{t}" if not t.startswith(IMAGE_TOKEN) else t


def extract_first_image(sample: Dict[str, Any]) -> Optional[str]:
    img = sample.get("images")
    if isinstance(img, list) and img:
        img0 = img[0]
        if isinstance(img0, str):
            return img0
    if isinstance(img, str):
        return img
    # 兼容其他字段名
    for key in ("image", "img", "picture"):
        v = sample.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def image_basename(img_path: str) -> str:
    # 支持形如 "./images/<id>.png"、"images/<id>.png" 或绝对路径
    base = os.path.basename(img_path)
    # 若包含类似 "images/<id>.png"，直接取最后一段
    return base


def make_rel_img_path(basename: str) -> str:
    # 以 images/DiagramGenBenchmark/images 为前缀
    return f"{REL_IMG_PREFIX}/{basename}"


# 图片复制默认开启；可通过 --no-copy-images 关闭


def to_test_item(sample: Dict[str, Any], rel_img: str) -> Dict[str, Any]:
    q_text = sample.get("query") or sample.get("instruction") or ""
    a_text = sample.get("reference_answer") or sample.get("answer") or sample.get("code") or ""
    return {
        "qry_text": normalize_image_token(q_text),
        "qry_img_path": rel_img,
        "tgt_text": [str(a_text)],
        "tgt_img_path": [""]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DiagramGenBenchmark DiagramCoding subset to MMCoIR test JSONL")
    parser.add_argument("--input-json", type=str, default=str(Path("datasets")/DATASET_NAME/"DiagramCoding.json"), help="输入 JSON（DiagramCoding.json）路径")
    parser.add_argument("--out-dir", type=str, default=str(Path("MMCoIR-test")/f"{DATASET_NAME}_i2c"), help="输出目录（将生成 test.jsonl）")
    parser.add_argument("--limit", type=int, default=None, help="仅转换前 N 条样本")
    parser.add_argument("--no-copy-images", action="store_true", help="不复制源图片到 qry_image_path 指定的目标目录")
    parser.add_argument("--src-images-dir", type=str, default=str(Path("datasets")/DATASET_NAME/IMG_SUBDIR), help="源图片目录（默认 datasets/DiagramGenBenchmark/images）")
    parser.add_argument("--overwrite-images", action="store_true", help="覆盖已存在的目标图片")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / "test.jsonl"

    ensure_dir(out_dir)

    # 读取数据
    try:
        samples = read_json(input_json)
    except FileNotFoundError:
        print(f"[ERROR] 未找到输入文件: {input_json}")
        return
    except Exception as e:
        print(f"[ERROR] 读取输入失败: {e}")
        return

    total = len(samples)
    if args.limit is not None:
        samples = samples[:args.limit]
    print(f"[INFO] 总样本: {total}，本次处理: {len(samples)}")

    test_items: List[Dict[str, Any]] = []
    # 复制相关目录
    src_images_dir = Path(args.src_images_dir)
    # 实际复制目标在 MMCoIR-test 下，但 JSON 中写入相对路径以 images/ 开头
    dest_images_dir = Path("MMCoIR-test") / REL_IMG_PREFIX
    copy_images = not args.no_copy_images
    if copy_images:
        ensure_dir(dest_images_dir)
        if not src_images_dir.exists():
            print(f"[WARN] 源图片目录不存在: {src_images_dir}")
    copy_ok, copy_skip, copy_missing, copy_error = 0, 0, 0, 0

    # 转换并复制图片
    for i, sp in enumerate(samples, 1):
        img_p = extract_first_image(sp)
        if not img_p:
            # 无图片则跳过
            continue
        basename = image_basename(img_p)
        rel_img = make_rel_img_path(basename)
        test_items.append(to_test_item(sp, rel_img))

        # 可选复制图片到 qry_image_path 指定的目录
        if copy_images:
            src_img = src_images_dir / basename
            dest_img = dest_images_dir / basename
            try:
                if not src_img.exists():
                    copy_missing += 1
                else:
                    if dest_img.exists() and not args.overwrite_images:
                        copy_skip += 1
                    else:
                        shutil.copy2(src_img, dest_img)
                        copy_ok += 1
            except Exception as e:
                copy_error += 1
                print(f"[ERROR] 复制失败 {src_img} -> {dest_img}: {e}")

        if i % 2000 == 0:
            print(f"  [proc] {i}/{len(samples)}")

    # 保存 JSONL
    print(f"[SAVE] Test JSONL -> {out_jsonl}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] DiagramGenBenchmark DiagramCoding -> MMCoIR test 完成")
    print(f"Test JSONL:  {out_jsonl}")
    if copy_images:
        print("[COPY] 图片复制统计:")
        print(f"  成功复制: {copy_ok}")
        print(f"  已存在跳过: {copy_skip}")
        print(f"  源缺失: {copy_missing}")
        print(f"  失败: {copy_error}")
        print(f"目标目录: {dest_images_dir}")
    else:
        print("[INFO] 未启用复制；请确保图片已位于 MMCoIR-test/images/DiagramGenBenchmark/images 下")

    # 打印一个示例
    if test_items:
        print("\n=== 测试样本示例 ===")
        print(json.dumps(test_items[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()