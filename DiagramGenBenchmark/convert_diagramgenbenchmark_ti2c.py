#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 DiagramAgent/DiagramGenBenchmark 的 DiagramEditing 子集转换为“文本+图片到代码搜索(ti2c)”测试集。

输出：
- 生成到 `MMCoIR-test/DiagramGenBenchmark_ti2c/test.jsonl`。
- 图片复制到 `MMCoIR-test/images/DiagramGenBenchmark/images/`（默认开启，可关闭）；JSONL 中图片路径以 `images/DiagramGenBenchmark/images/<filename>` 记录。
- JSONL 字段（共 4 个 key）：
  - `qry_text`: 来自 `query` 字段的文本，前置一个 image token（`<|image_1|>`）。
  - `qry_img_path`: 相对图片路径（`images/DiagramGenBenchmark/images/<filename>`）。
  - `tgt_text`: 列表，包含来自 `reference_answer` 的目标代码字符串。
  - `tgt_img_path`: 列表，始终为 [""]（本任务无目标图片）。

说明：
- 原数据的 `images` 字段为仅含一个元素的列表，脚本使用其中第一个元素作为查询图片。
- 复制功能默认开启；关闭复制时，请确保图片已在 `MMCoIR-test/images/DiagramGenBenchmark/images/` 下。

示例用法：
  python DiagramGenBenchmark/convert_diagramgenbenchmark_ti2c.py \
    --input-json datasets/DiagramGenBenchmark/DiagramEditing.json \
    --out-dir MMCoIR-test/DiagramGenBenchmark_ti2c \
    --src-images-dir datasets/DiagramGenBenchmark/images

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


DATASET_NAME = "DiagramGenBenchmark"
SUBSET_JSON = "DiagramEditing.json"
OUT_DATASET_NAME = "DiagramGenBenchmark_ti2c"
OUT_ROOT = "MMCoIR-test"  # JSONL 输出根目录
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
        for key in ("data", "items", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    else:
        return []


def normalize_image_token(text: str) -> str:
    if text is None:
        text = ""
    t = str(text).replace("<image>", "").strip()
    return f"{IMAGE_TOKEN}\n{t}" if not t.startswith(IMAGE_TOKEN) else t


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


def make_rel_img_path(basename: str) -> str:
    return f"{REL_IMG_PREFIX}/{basename}"


def get_query_text(sample: Dict[str, Any]) -> Optional[str]:
    v = sample.get("query") or sample.get("instruction")
    if v is None:
        return None
    return str(v)


def get_reference_code(sample: Dict[str, Any]) -> Optional[str]:
    v = sample.get("reference_answer") or sample.get("answer") or sample.get("code")
    if v is None:
        return None
    return str(v)


def to_test_item(q_text: str, rel_img: str, code_str: str) -> Dict[str, Any]:
    return {
        "qry_text": normalize_image_token(q_text),
        "qry_img_path": rel_img,
        "tgt_text": [code_str],
        "tgt_img_path": [""]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DiagramGenBenchmark DiagramEditing subset to text+image-to-code (ti2c) test JSONL")
    parser.add_argument("--input-json", type=str, default=str(Path("datasets")/DATASET_NAME/SUBSET_JSON), help="输入 JSON（DiagramEditing.json）路径")
    parser.add_argument("--out-dir", type=str, default=str(Path(OUT_ROOT)/OUT_DATASET_NAME), help="输出目录（将生成 test.jsonl）")
    parser.add_argument("--limit", type=int, default=None, help="仅转换前 N 条样本")
    parser.add_argument("--no-copy-images", action="store_true", help="不复制源图片到目标目录")
    parser.add_argument("--src-images-dir", type=str, default=str(Path("datasets")/DATASET_NAME/"images"), help="源图片目录（默认 datasets/DiagramGenBenchmark/images）")
    parser.add_argument("--overwrite-images", action="store_true", help="覆盖已存在的目标图片")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / "test.jsonl"
    ensure_dir(out_dir)

    # 复制相关目录（物理保存到 MMCoIR-test/images/...，JSON 写入以 images/... 开头）
    src_images_dir = Path(args.src_images_dir)
    dest_images_dir = Path(OUT_ROOT) / REL_IMG_PREFIX
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
    drop_missing_q, drop_missing_img, drop_missing_code = 0, 0, 0

    for i, sp in enumerate(samples, 1):
        q_text = get_query_text(sp)
        img_p = extract_first_image(sp)
        code_str = get_reference_code(sp)

        if not q_text:
            drop_missing_q += 1
            continue
        if not img_p:
            drop_missing_img += 1
            continue
        if not code_str:
            drop_missing_code += 1
            continue

        basename = image_basename(img_p)
        rel_img = make_rel_img_path(basename)

        test_items.append(to_test_item(q_text, rel_img, code_str))

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

    print("[DONE] DiagramGenBenchmark DiagramEditing -> TI2C 测试集 完成")
    print(f"Test JSONL:  {out_jsonl}")
    print(f"[INFO] 丢弃样本 - 无 query: {drop_missing_q}, 无 images: {drop_missing_img}, 无 reference_answer/code: {drop_missing_code}")
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