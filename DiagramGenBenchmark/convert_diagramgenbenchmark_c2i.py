#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 DiagramAgent/DiagramGenBenchmark 的 DiagramCoding 子集转换为“代码到图片搜索(c2i)”测试集。

输出：
- 生成到 `MMCoIR-test/DiagramGenBenchmark_c2i/test.jsonl`。
- 图片复制到 `MMCoIR-test/images/DiagramGenBenchmark/images/`（默认开启，可关闭）；JSONL 中路径以 `images/DiagramGenBenchmark/images/<filename>` 记录。
- JSONL 字段：
  - `qry_text`: 文本查询，内容为 "Please convert this code to image:" + 代码（无 image token）。
  - `qry_img_path`: 始终为空字符串 ""。
  - `tgt_text`: 仅包含一个 image token（`<|image_1|>`）。
  - `tgt_img_path`: 列表，含目标图片路径（`images/DiagramGenBenchmark/images/<filename>`）。

示例用法：
  python DiagramGenBenchmark/convert_diagramgenbenchmark_c2i.py \
    --input-json datasets/DiagramGenBenchmark/DiagramCoding.json \
    --out-dir MMCoIR-test/DiagramGenBenchmark_c2i

可选参数：
- `--limit N`              仅处理前 N 条样本
- `--no-copy-images`       不复制图片到目标目录
- `--src-images-dir PATH`  源图片目录（默认 datasets/DiagramGenBenchmark/images）
- `--overwrite-images`     覆盖已存在的目标图片
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional


DATASET_NAME = "DiagramGenBenchmark"               # 原项目名用于图片目录
OUT_DATASET_NAME = "DiagramGenBenchmark_c2i"        # 输出数据集目录名
IMG_SUBDIR = "images"
OUT_ROOT = "MMCoIR-test"                            # JSONL 输出根目录
DEST_IMG_PREFIX = "images/DiagramGenBenchmark/images"  # 目标图片前缀（与原项目名一致）
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
        for key in ("data", "items", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    else:
        return []


def extract_first_image(sample: Dict[str, Any]) -> Optional[str]:
    img = sample.get("images")
    if isinstance(img, list) and img:
        img0 = img[0]
        if isinstance(img0, str):
            return img0
    if isinstance(img, str):
        return img
    for key in ("image", "img", "picture"):
        v = sample.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def image_basename(img_path: str) -> str:
    return os.path.basename(img_path)


def make_tgt_image_path(basename: str) -> str:
    return f"{DEST_IMG_PREFIX}/{basename}"


def get_code_text(sample: Dict[str, Any]) -> Optional[str]:
    # 代码来源字段：reference_answer / answer / code
    v = sample.get("reference_answer") or sample.get("answer") or sample.get("code")
    if v is None:
        return None
    return str(v)


def build_qry_text(code_str: str) -> str:
    return f"Please convert this code to image:\n{code_str}".strip()


def to_test_item(sample: Dict[str, Any], tgt_img_path: str, code_str: str) -> Dict[str, Any]:
    return {
        "qry_text": build_qry_text(code_str),
        "qry_img_path": "",
        "tgt_text": [IMAGE_TOKEN],
        "tgt_img_path": [tgt_img_path],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DiagramGenBenchmark DiagramCoding subset to code-to-image (c2i) test JSONL")
    parser.add_argument("--input-json", type=str, default=str(Path("datasets")/DATASET_NAME/"DiagramCoding.json"), help="输入 JSON（DiagramCoding.json）路径")
    parser.add_argument("--out-dir", type=str, default=str(Path(OUT_ROOT)/OUT_DATASET_NAME), help="输出目录（将生成 test.jsonl）")
    parser.add_argument("--limit", type=int, default=None, help="仅转换前 N 条样本")
    parser.add_argument("--no-copy-images", action="store_true", help="不复制源图片到目标目录")
    parser.add_argument("--src-images-dir", type=str, default=str(Path("datasets")/DATASET_NAME/IMG_SUBDIR), help="源图片目录（默认 datasets/DiagramGenBenchmark/images）")
    parser.add_argument("--overwrite-images", action="store_true", help="覆盖已存在的目标图片")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / "test.jsonl"
    ensure_dir(out_dir)

    # 复制相关目录
    src_images_dir = Path(args.src_images_dir)
    # 实际复制目标在 MMCoIR-test 下，但 JSON 中写入相对路径以 images/ 开头
    dest_images_dir = Path(OUT_ROOT) / DEST_IMG_PREFIX
    copy_images = not args.no_copy_images
    if copy_images:
        ensure_dir(dest_images_dir)
        if not src_images_dir.exists():
            print(f"[WARN] 源图片目录不存在: {src_images_dir}")

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
    copy_ok, copy_skip, copy_missing, copy_error = 0, 0, 0, 0
    drop_missing_code, drop_missing_image = 0, 0

    for i, sp in enumerate(samples, 1):
        img_p = extract_first_image(sp)
        code_str = get_code_text(sp)

        if not code_str:
            drop_missing_code += 1
            continue
        if not img_p:
            drop_missing_image += 1
            continue

        basename = image_basename(img_p)
        tgt_img_path = make_tgt_image_path(basename)

        test_items.append(to_test_item(sp, tgt_img_path, code_str))

        # 复制图片到 MMCoIR-test/images/DiagramGenBenchmark/images
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

    print("[DONE] DiagramGenBenchmark DiagramCoding -> C2I 测试集 完成")
    print(f"Test JSONL:  {out_jsonl}")
    print(f"[INFO] 丢弃样本 - 无代码: {drop_missing_code}, 无图片: {drop_missing_image}")
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