#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 nllg/datikz-v3 数据集转换为 MMCoIR i2c（image-to-code）训练/测试集。

输入结构：
- 在 `datasets/datikz-v3/data/` 下存在多个 `.parquet` 文件；可通过文件名判断分割（例如：`train-00000-of-000xx.parquet`、`test-00000-of-000xx.parquet`）。
- 若确有 `train/` 与 `test/` 子目录也会优先使用，否则按文件名自动分类。

采样：
- 训练集随机采样 100,000 条（可配），测试集随机采样 2,000 条（可配），支持 `--seed` 重现。

输出：
- 写入 `MMCoIR-train/DATIKZ_i2c/train.jsonl` 与 `MMCoIR-test/DATIKZ_i2c/test.jsonl`。
- 图片分别复制到：
  - `MMCoIR-train/images/DATIKZ_i2c/images/`
  - `MMCoIR-test/images/DATIKZ_i2c/images/`
- JSONL 字段（4 个 key，与规范一致）：
  - `qry_text`: 固定文本，包含 image token：`<|image_1|>\nPlease convert this image to code.`
  - `qry_img_path`: 相对图片路径（`images/DATIKZ_i2c/images/<filename>`）
  - `tgt_text`: 目标代码列表（单元素）
  - `tgt_img_path`: 目标图片路径列表（本任务无 -> [""]）

示例：
  python DATIKZ/convert_datikz_i2c.py \
    --dataset-root datasets/datikz-v3 \
    --train-limit 100000 --test-limit 2000 --seed 42

可选参数：
- `--dataset-root PATH`   数据集根目录（默认 datasets/datikz-v3）
- `--data-subdir NAME`    data 子目录名（默认 data）
- `--train-subdir NAME`   训练子目录名（默认 train）
- `--test-subdir NAME`    测试子目录名（默认 test）
- `--image-token TOKEN`   查询文本图片 token（默认 `<|image_1|>`）
- `--image-col NAME`      图片列名（默认 image）
- `--code-col NAME`       代码列名（默认 code）
- `--train-limit N`       训练采样大小（默认 100000）
- `--test-limit N`        测试采样大小（默认 2000）
- `--seed N`              随机种子（默认 42）
- `--no-copy-images`      不复制图片到目标目录
- `--overwrite-images`    覆盖已存在的目标图片
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple
import re


REL_IMG_PREFIX = "images/DATIKZ_i2c/images"
DEFAULT_IMAGE_TOKEN = "<|image_1|>"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_parquet_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.rglob("*.parquet")])


def categorize_by_filename(files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """根据文件名将 parquet 分类为 (train_files, test_files, unknown_files)。"""
    train_files: List[Path] = []
    test_files: List[Path] = []
    unknown_files: List[Path] = []
    # 模式：按分词边界匹配 train/test/val
    pat_train = re.compile(r"(?:^|[._\-/\\])train(?:[._\-/\\]|$)", re.IGNORECASE)
    pat_test = re.compile(r"(?:^|[._\-/\\])test(?:[._\-/\\]|$)", re.IGNORECASE)
    pat_val  = re.compile(r"(?:^|[._\-/\\])(val|valid|validation)(?:[._\-/\\]|$)", re.IGNORECASE)
    for fp in files:
        name = fp.name.lower()
        if pat_train.search(name):
            train_files.append(fp)
        elif pat_test.search(name) or pat_val.search(name):
            test_files.append(fp)
        else:
            unknown_files.append(fp)
    return train_files, test_files, unknown_files


def iter_parquet_rows(files: List[Path], image_col: str, code_col: str) -> Iterable[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        raise RuntimeError("读取 parquet 需要 pandas 依赖，请先安装：pip install pandas pyarrow")
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"[WARN] 读取失败，跳过 {fp}: {e}")
            continue
        if image_col not in df.columns or code_col not in df.columns:
            print(f"[WARN] 缺少列 {image_col}/{code_col}，跳过 {fp}；现有列: {list(df.columns)}")
            continue
        for _, row in df.iterrows():
            img = row[image_col]
            code = row[code_col]
            if img is None or code is None:
                continue
            yield {"image": str(img), "code": str(code)}


def reservoir_sample(iterable: Iterable[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    sample: List[Dict[str, Any]] = []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def image_basename(img_path: str) -> str:
    return os.path.basename(img_path)


def make_rel_img_path(basename: str) -> str:
    return f"{REL_IMG_PREFIX}/{basename}"


def to_item(image_token: str, rel_img: str, code_str: str) -> Dict[str, Any]:
    qry = f"{image_token}\nPlease convert this image to code."
    return {
        "qry_text": qry,
        "qry_img_path": rel_img,
        "tgt_text": [code_str],
        "tgt_img_path": [""]
    }


def resolve_src_image(dataset_root: Path, src_images_dir: Path, image_value: str) -> Path:
    """尽可能解析图片的真实路径：优先尝试 dataset_root 下的相对路径，其次回退到 src_images_dir/basename。"""
    iv = image_value.strip()
    p = Path(iv)
    # 若是绝对路径
    if p.is_absolute() and p.exists():
        return p
    # 若是相对路径，尝试拼接 dataset_root
    cand = dataset_root / iv
    if cand.exists():
        return cand
    # 尝试 data 子目录前缀
    cand2 = dataset_root / "data" / iv
    if cand2.exists():
        return cand2
    # 回退到 src_images_dir / basename
    base = os.path.basename(iv)
    cand3 = src_images_dir / base
    return cand3


def main():
    parser = argparse.ArgumentParser(description="Convert DATIKZ (nllg/datikz-v3) to MMCoIR i2c train/test JSONL with sampling")
    parser.add_argument("--dataset-root", type=str, default=str(Path("datasets")/"datikz-v3"), help="数据集根目录（默认 datasets/datikz-v3）")
    parser.add_argument("--data-subdir", type=str, default="data", help="数据子目录名（默认 data）")
    parser.add_argument("--train-subdir", type=str, default="train", help="训练子目录名（默认 train）")
    parser.add_argument("--test-subdir", type=str, default="test", help="测试子目录名（默认 test）")
    parser.add_argument("--image-token", type=str, default=DEFAULT_IMAGE_TOKEN, help="查询文本中的图片 token（默认 <|image_1|>）")
    parser.add_argument("--image-col", type=str, default="image", help="图片列名（默认 image）")
    parser.add_argument("--code-col", type=str, default="code", help="代码列名（默认 code）")
    parser.add_argument("--train-limit", type=int, default=100000, help="训练采样大小（默认 100000）")
    parser.add_argument("--test-limit", type=int, default=2000, help="测试采样大小（默认 2000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--no-copy-images", action="store_true", help="不复制源图片到目标目录")
    parser.add_argument("--overwrite-images", action="store_true", help="覆盖已存在的目标图片")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    data_dir = dataset_root / args.data_subdir
    train_dir = data_dir / args.train_subdir
    test_dir = data_dir / args.test_subdir

    # 列出 parquet 文件（优先使用 train/test 子目录；否则按文件名自动分类）
    train_files = list_parquet_files(train_dir)
    test_files = list_parquet_files(test_dir)
    if not train_files and not test_files:
        all_files = list_parquet_files(data_dir)
        if not all_files:
            print(f"[ERROR] 在数据目录未找到任何 parquet 文件: {data_dir}")
            return
        train_files, test_files, unknown_files = categorize_by_filename(all_files)
        if unknown_files:
            print(f"[WARN] 存在无法识别分割的文件 {len(unknown_files)} 个，示例: {unknown_files[:3]}")

    # 采样
    train_rows = reservoir_sample(iter_parquet_rows(train_files, args.image_col, args.code_col), args.train_limit, args.seed)
    test_rows = reservoir_sample(iter_parquet_rows(test_files, args.image_col, args.code_col), args.test_limit, args.seed)

    print(f"[INFO] 训练采样: {len(train_rows)} / 目标 {args.train_limit}；测试采样: {len(test_rows)} / 目标 {args.test_limit}")

    # 输出目录
    out_train_dir = Path("MMCoIR-train") / "DATIKZ_i2c"
    out_test_dir = Path("MMCoIR-test") / "DATIKZ_i2c"
    ensure_dir(out_train_dir)
    ensure_dir(out_test_dir)
    out_train_jsonl = out_train_dir / "train.jsonl"
    out_test_jsonl = out_test_dir / "test.jsonl"

    # 复制目录目标
    dest_train_images_dir = Path("MMCoIR-train") / REL_IMG_PREFIX
    dest_test_images_dir = Path("MMCoIR-test") / REL_IMG_PREFIX
    copy_images = not args.no_copy_images
    if copy_images:
        ensure_dir(dest_train_images_dir)
        ensure_dir(dest_test_images_dir)

    # 源图片目录灵活解析：优先 dataset_root 相对路径，回退到 dataset_root/images
    src_images_dir_default = dataset_root / "images"

    # 写训练集
    train_items: List[Dict[str, Any]] = []
    t_copy_ok, t_copy_skip, t_copy_missing, t_copy_error = 0, 0, 0, 0
    for i, r in enumerate(train_rows, 1):
        img_val = r.get("image")
        code_str = r.get("code")
        if not img_val or not code_str:
            continue
        base = image_basename(img_val)
        rel_img = make_rel_img_path(base)
        train_items.append(to_item(args.image_token, rel_img, str(code_str)))
        if copy_images:
            src_img = resolve_src_image(dataset_root, src_images_dir_default, img_val)
            dest_img = dest_train_images_dir / base
            try:
                if not src_img.exists():
                    t_copy_missing += 1
                else:
                    if dest_img.exists() and not args.overwrite_images:
                        t_copy_skip += 1
                    else:
                        shutil.copy2(src_img, dest_img)
                        t_copy_ok += 1
            except Exception as e:
                t_copy_error += 1
                print(f"[ERROR] 训练复制失败 {src_img} -> {dest_img}: {e}")
        if i % 20000 == 0:
            print(f"  [train proc] {i}/{len(train_rows)}")

    # 写测试集
    test_items: List[Dict[str, Any]] = []
    v_copy_ok, v_copy_skip, v_copy_missing, v_copy_error = 0, 0, 0, 0
    for i, r in enumerate(test_rows, 1):
        img_val = r.get("image")
        code_str = r.get("code")
        if not img_val or not code_str:
            continue
        base = image_basename(img_val)
        rel_img = make_rel_img_path(base)
        test_items.append(to_item(args.image_token, rel_img, str(code_str)))
        if copy_images:
            src_img = resolve_src_image(dataset_root, src_images_dir_default, img_val)
            dest_img = dest_test_images_dir / base
            try:
                if not src_img.exists():
                    v_copy_missing += 1
                else:
                    if dest_img.exists() and not args.overwrite_images:
                        v_copy_skip += 1
                    else:
                        shutil.copy2(src_img, dest_img)
                        v_copy_ok += 1
            except Exception as e:
                v_copy_error += 1
                print(f"[ERROR] 测试复制失败 {src_img} -> {dest_img}: {e}")
        if i % 2000 == 0:
            print(f"  [test proc] {i}/{len(test_rows)}")

    # 保存 JSONL
    print(f"[SAVE] Train JSONL -> {out_train_jsonl}")
    with out_train_jsonl.open("w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[SAVE] Test JSONL -> {out_test_jsonl}")
    with out_test_jsonl.open("w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] DATIKZ -> i2c 训练/测试集 完成")
    print(f"Train JSONL: {out_train_jsonl}")
    print(f"Test  JSONL: {out_test_jsonl}")
    if copy_images:
        print("[COPY] 训练图片复制统计:")
        print(f"  成功复制: {t_copy_ok}")
        print(f"  已存在跳过: {t_copy_skip}")
        print(f"  源缺失: {t_copy_missing}")
        print(f"  失败: {t_copy_error}")
        print(f"训练目标目录: {dest_train_images_dir}")
        print("[COPY] 测试图片复制统计:")
        print(f"  成功复制: {v_copy_ok}")
        print(f"  已存在跳过: {v_copy_skip}")
        print(f"  源缺失: {v_copy_missing}")
        print(f"  失败: {v_copy_error}")
        print(f"测试目标目录: {dest_test_images_dir}")
    else:
        print("[INFO] 未启用复制；请确保图片分别位于 MMCoIR-train/images/DATIKZ_i2c/images 与 MMCoIR-test/images/DATIKZ_i2c/images 下")

    # 打印一个示例
    if train_items:
        print("\n=== 训练样本示例 ===")
        print(json.dumps(train_items[0], ensure_ascii=False, indent=2))
    if test_items:
        print("\n=== 测试样本示例 ===")
        print(json.dumps(test_items[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()