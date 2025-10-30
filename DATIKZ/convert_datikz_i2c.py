#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 nllg/datikz-v3 数据集转换为 MMCoIR 测试集（image-to-code, i2c）。

输入假设：
- 读取一个包含两列的文件：`image` 和 `code`（支持 .jsonl 或 .parquet）。
- 图片源目录默认位于 `datasets/datikz-v3/images/`（可通过参数指定）。

输出：
- 生成到 `MMCoIR-test/DATIKZ_i2c/test.jsonl`。
- 图片复制到 `MMCoIR-test/images/DATIKZ/images/`（默认开启，可关闭）；JSONL 中图片路径以 `images/DATIKZ/images/<filename>` 记录。
- JSONL 字段（共 4 个 key，与其他测试集一致）：
  - `qry_text`: 固定文本，包含 image token：`<|image_1|>\nPlease convert this image to code.`
  - `qry_img_path`: 相对图片路径（`images/DATIKZ/images/<filename>`）。
  - `tgt_text`: 列表，包含目标代码字符串。
  - `tgt_img_path`: 列表，始终为 [""]（本任务无目标图片）。

示例用法：
  python DATIKZ/convert_datikz_i2c.py \
    --input-file datasets/datikz-v3/test.jsonl \
    --out-dir MMCoIR-test/DATIKZ_i2c \
    --src-images-dir datasets/datikz-v3/images

可选参数：
- `--limit N`              仅处理前 N 条样本
- `--no-copy-images`       不复制图片到目标目录
- `--src-images-dir PATH`  源图片目录（默认 datasets/datikz-v3/images）
- `--overwrite-images`     覆盖已存在的目标图片
- `--image-token TOKEN`    配置查询文本中的图片 token（默认 `<|image_1|>`）
- `--image-col NAME`       输入文件中图片列名（默认 `image`）
- `--code-col NAME`        输入文件中代码列名（默认 `code`）
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List


REL_IMG_PREFIX = "images/DATIKZ/images"
DEFAULT_IMAGE_TOKEN = "<|image_1|>"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_input_file(path: Path, image_col: str, code_col: str) -> List[Dict[str, Any]]:
    """读取 .jsonl 或 .parquet，返回 {image: str, code: str} 列表。
    """
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    rows: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # 兼容纯 JSON（非逐行）
                    try:
                        data = json.loads(line)
                        obj = data
                    except Exception:
                        continue
                if image_col in obj and code_col in obj:
                    rows.append({"image": str(obj[image_col]), "code": str(obj[code_col])})
    elif suffix in (".parquet", ".pq"):
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError("读取 parquet 需要 pandas 依赖，请先安装：pip install pandas pyarrow")
        df = pd.read_parquet(path)
        if image_col not in df.columns or code_col not in df.columns:
            raise KeyError(f"输入文件缺少列: {image_col} 或 {code_col}; 现有列: {list(df.columns)}")
        for _, row in df.iterrows():
            rows.append({"image": str(row[image_col]), "code": str(row[code_col])})
    else:
        raise ValueError(f"不支持的输入文件类型: {suffix}（支持 .jsonl / .parquet）")

    return rows


def image_basename(img_path: str) -> str:
    return os.path.basename(img_path)


def make_rel_img_path(basename: str) -> str:
    return f"{REL_IMG_PREFIX}/{basename}"


def to_test_item(image_token: str, rel_img: str, code_str: str) -> Dict[str, Any]:
    qry = f"{image_token}\nPlease convert this image to code."
    return {
        "qry_text": qry,
        "qry_img_path": rel_img,
        "tgt_text": [code_str],
        "tgt_img_path": [""]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DATIKZ (nllg/datikz-v3) to MMCoIR i2c test JSONL")
    parser.add_argument("--input-file", type=str, default=str(Path("datasets")/"datikz-v3"/"test.jsonl"), help="输入文件路径（支持 .jsonl / .parquet），需包含 image 与 code 列")
    parser.add_argument("--out-dir", type=str, default=str(Path("MMCoIR-test")/"DATIKZ_i2c"), help="输出目录（将生成 test.jsonl）")
    parser.add_argument("--limit", type=int, default=None, help="仅转换前 N 条样本")
    parser.add_argument("--no-copy-images", action="store_true", help="不复制源图片到目标目录")
    parser.add_argument("--src-images-dir", type=str, default=str(Path("datasets")/"datikz-v3"/"images"), help="源图片目录（默认 datasets/datikz-v3/images）")
    parser.add_argument("--overwrite-images", action="store_true", help="覆盖已存在的目标图片")
    parser.add_argument("--image-token", type=str, default=DEFAULT_IMAGE_TOKEN, help="查询文本中的图片 token（默认 <|image_1|>）")
    parser.add_argument("--image-col", type=str, default="image", help="输入文件中图片列名（默认 image）")
    parser.add_argument("--code-col", type=str, default="code", help="输入文件中奖励代码列名（默认 code）")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / "test.jsonl"
    ensure_dir(out_dir)

    # 复制目录（物理保存到 MMCoIR-test/images/DATIKZ/images；JSON 写入以 images/DATIKZ/images/<filename>）
    src_images_dir = Path(args.src_images_dir)
    dest_images_dir = Path("MMCoIR-test") / REL_IMG_PREFIX
    copy_images = not args.no_copy_images
    if copy_images:
        ensure_dir(dest_images_dir)
        if not src_images_dir.exists():
            print(f"[WARN] 源图片目录不存在: {src_images_dir}")

    # 读取输入
    try:
        rows = read_input_file(input_file, args.image_col, args.code_col)
    except Exception as e:
        print(f"[ERROR] 读取输入失败: {e}")
        return

    total = len(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"[INFO] 总样本: {total}，本次处理: {len(rows)}")

    test_items: List[Dict[str, Any]] = []
    copy_ok, copy_skip, copy_missing, copy_error = 0, 0, 0, 0

    for i, r in enumerate(rows, 1):
        img_p = r.get("image")
        code_str = r.get("code")
        if not img_p or not code_str:
            continue

        basename = image_basename(img_p)
        rel_img = make_rel_img_path(basename)
        test_items.append(to_test_item(args.image_token, rel_img, str(code_str)))

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
            print(f"  [proc] {i}/{len(rows)}")

    # 保存 JSONL
    print(f"[SAVE] Test JSONL -> {out_jsonl}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] DATIKZ -> i2c 测试集 完成")
    print(f"Test JSONL:  {out_jsonl}")
    if copy_images:
        print("[COPY] 图片复制统计:")
        print(f"  成功复制: {copy_ok}")
        print(f"  已存在跳过: {copy_skip}")
        print(f"  源缺失: {copy_missing}")
        print(f"  失败: {copy_error}")
        print(f"目标目录: {dest_images_dir}")
    else:
        print("[INFO] 未启用复制；请确保图片已位于 MMCoIR-test/images/DATIKZ/images 下")

    if test_items:
        print("\n=== 测试样本示例 ===")
        print(json.dumps(test_items[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()