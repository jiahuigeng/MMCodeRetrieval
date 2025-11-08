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
- 写入 `MMCoIR-train/DATIKZ_i2c/train.jsonl` 与 `MMCoIR-test/DATIKZ_i2c/test.jsonl`（可通过 `--out-name` 自定义）。
- 图片分别复制到：
  - `MMCoIR-train/images/DATIKZ/images/`
  - `MMCoIR-test/images/DATIKZ/images/`
- JSONL 字段（4 个 key，与规范一致）：
  - `qry_text`: 固定文本，包含 image token：`<|image_1|>\nPlease convert this image to code.`
  - `qry_img_path`: 相对图片路径（`images/DATIKZ/images/<filename>`）
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
from typing import Dict, Any, List, Iterable, Tuple, Optional
import re
import hashlib
import io


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
            # 保留原始图像值（可能是 str 路径、dict 包含 bytes/path、PIL.Image、或字节），供后续灵活处理
            yield {"image": img, "code": str(code)}


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


def image_basename_from_value(image_value: Any, fallback_name: str) -> str:
    """从不同类型的 image 值生成稳定的文件名。

    - str: 视为路径，使用 basename
    - dict: 若含 'path' 使用其 basename；若含 'bytes' 使用 sha1 生成名
    - bytes: 使用 sha1 生成名
    - PIL.Image: 使用 fallback_name
    - 其他: 使用 fallback_name
    """
    try:
        # str 路径
        if isinstance(image_value, str):
            return os.path.basename(image_value)
        # dict 结构（HF Image 常见）
        if isinstance(image_value, dict):
            if 'path' in image_value and image_value['path']:
                return os.path.basename(str(image_value['path']))
            if 'bytes' in image_value and image_value['bytes']:
                b = image_value['bytes']
                h = hashlib.sha1(b).hexdigest()[:16]
                return f"img_{h}.png"
        # 原始字节
        if isinstance(image_value, (bytes, bytearray)):
            h = hashlib.sha1(bytes(image_value)).hexdigest()[:16]
            return f"img_{h}.png"
        # PIL.Image
        try:
            from PIL.Image import Image as PILImage
            if isinstance(image_value, PILImage):
                return fallback_name
        except Exception:
            pass
    except Exception:
        pass
    return fallback_name


def make_rel_img_path(rel_prefix: str, basename: str) -> str:
    return f"{rel_prefix}/{basename}"


def to_train_item(image_token: str, rel_img: str, code_str: str) -> Dict[str, Any]:
    qry = f"{image_token}\nPlease convert this image to code."
    return {
        "qry": qry,
        "qry_image_path": rel_img,
        "pos_text": str(code_str),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(image_token: str, rel_img: str, code_str: str) -> Dict[str, Any]:
    qry = f"{image_token}\nPlease convert this image to code."
    return {
        "qry_text": qry,
        "qry_img_path": rel_img,
        "tgt_text": [str(code_str)],
        "tgt_img_path": [""]
    }


def extract_image_payload(dataset_root: Path, src_images_dir: Path, image_value: Any) -> Tuple[Optional[Path], Optional[bytes]]:
    """提取图像来源：返回 (src_path, img_bytes)。

    - 若有可用路径（绝对或相对存在），返回 src_path
    - 若包含字节内容（dict['bytes'] 或 bytes），返回 img_bytes
    - 若为 PIL.Image，转存为 PNG 字节返回 img_bytes
    """
    # str 路径情况
    if isinstance(image_value, str):
        iv = image_value.strip()
        p = Path(iv)
        if p.is_absolute() and p.exists():
            return p, None
        cand = dataset_root / iv
        if cand.exists():
            return cand, None
        cand2 = dataset_root / "data" / iv
        if cand2.exists():
            return cand2, None
        base = os.path.basename(iv)
        cand3 = src_images_dir / base
        if cand3.exists():
            return cand3, None
        # 找不到路径，视为无路径
        return None, None

    # dict（HF Image Feature 常见）
    if isinstance(image_value, dict):
        # 优先 path
        if 'path' in image_value and image_value['path']:
            try:
                p = Path(str(image_value['path']))
                if p.is_absolute() and p.exists():
                    return p, None
                cand = dataset_root / p
                if cand.exists():
                    return cand, None
            except Exception:
                pass
        # 其次 bytes
        if 'bytes' in image_value and image_value['bytes']:
            try:
                b = image_value['bytes']
                if isinstance(b, (bytes, bytearray)):
                    return None, bytes(b)
            except Exception:
                pass
        return None, None

    # 原始字节
    if isinstance(image_value, (bytes, bytearray)):
        return None, bytes(image_value)

    # PIL.Image
    try:
        from PIL import Image as PILImageModule
        from PIL.Image import Image as PILImage
        if isinstance(image_value, PILImage):
            buf = io.BytesIO()
            image_value.save(buf, format='PNG')
            return None, buf.getvalue()
    except Exception:
        pass

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Convert DATIKZ (nllg/datikz-v3) to MMCoIR i2c train/test JSONL with sampling")
    parser.add_argument("--dataset-root", type=str, default=str(Path("datasets")/"datikz-v3"), help="数据集根目录（默认 datasets/datikz-v3）")
    parser.add_argument("--data-subdir", type=str, default="data", help="数据子目录名（默认 data）")
    parser.add_argument("--train-subdir", type=str, default="train", help="训练子目录名（默认 train）")
    parser.add_argument("--test-subdir", type=str, default="test", help="测试子目录名（默认 test）")
    parser.add_argument("--image-token", type=str, default=DEFAULT_IMAGE_TOKEN, help="查询文本中的图片 token（默认 <|image_1|>）")
    parser.add_argument("--out-name", type=str, default="DATIKZ_i2c", help="JSONL 输出数据集文件夹名（默认 DATIKZ_i2c）")
    parser.add_argument("--images-name", type=str, default="DATIKZ", help="图片目录与 qry_img_path 的数据集名（默认 DATIKZ）")
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
    out_train_dir = Path("MMCoIR-train") / args.out_name
    out_test_dir = Path("MMCoIR-test") / args.out_name
    ensure_dir(out_train_dir)
    ensure_dir(out_test_dir)
    out_train_jsonl = out_train_dir / "train.jsonl"
    out_test_jsonl = out_test_dir / "test.jsonl"

    # 复制目录目标
    rel_img_prefix = f"images/{args.images_name}/images"
    dest_train_images_dir = Path("MMCoIR-train") / rel_img_prefix
    dest_test_images_dir = Path("MMCoIR-test") / rel_img_prefix
    copy_images = not args.no_copy_images
    if copy_images:
        ensure_dir(dest_train_images_dir)
        ensure_dir(dest_test_images_dir)

    # 源图片目录灵活解析：优先 dataset_root 相对路径，回退到 dataset_root/images
    src_images_dir_default = dataset_root / "images"

    # 写训练集
    train_items: List[Dict[str, Any]] = []
    t_copy_ok, t_copy_skip, t_copy_missing, t_copy_error = 0, 0, 0, 0
    t_write_ok, t_write_skip = 0, 0
    for i, r in enumerate(train_rows, 1):
        img_val = r.get("image")
        code_str = r.get("code")
        if not img_val or not code_str:
            continue
        base = image_basename_from_value(img_val, f"datikz_train_{i:07d}.png")
        rel_img = make_rel_img_path(rel_img_prefix, base)
        train_items.append(to_train_item(args.image_token, rel_img, str(code_str)))
        if copy_images:
            src_path, img_bytes = extract_image_payload(dataset_root, src_images_dir_default, img_val)
            dest_img = dest_train_images_dir / base
            try:
                if src_path and src_path.exists():
                    if dest_img.exists() and not args.overwrite_images:
                        t_copy_skip += 1
                    else:
                        shutil.copy2(src_path, dest_img)
                        t_copy_ok += 1
                elif img_bytes:
                    if dest_img.exists() and not args.overwrite_images:
                        t_write_skip += 1
                    else:
                        with open(dest_img, 'wb') as f:
                            f.write(img_bytes)
                        t_write_ok += 1
                else:
                    t_copy_missing += 1
            except Exception as e:
                t_copy_error += 1
                print(f"[ERROR] 训练写入/复制失败 -> {dest_img}: {e}")
        if i % 20000 == 0:
            print(f"  [train proc] {i}/{len(train_rows)}")

    # 写测试集
    test_items: List[Dict[str, Any]] = []
    v_copy_ok, v_copy_skip, v_copy_missing, v_copy_error = 0, 0, 0, 0
    v_write_ok, v_write_skip = 0, 0
    for i, r in enumerate(test_rows, 1):
        img_val = r.get("image")
        code_str = r.get("code")
        if not img_val or not code_str:
            continue
        base = image_basename_from_value(img_val, f"datikz_test_{i:07d}.png")
        rel_img = make_rel_img_path(rel_img_prefix, base)
        test_items.append(to_test_item(args.image_token, rel_img, str(code_str)))
        if copy_images:
            src_path, img_bytes = extract_image_payload(dataset_root, src_images_dir_default, img_val)
            dest_img = dest_test_images_dir / base
            try:
                if src_path and src_path.exists():
                    if dest_img.exists() and not args.overwrite_images:
                        v_copy_skip += 1
                    else:
                        shutil.copy2(src_path, dest_img)
                        v_copy_ok += 1
                elif img_bytes:
                    if dest_img.exists() and not args.overwrite_images:
                        v_write_skip += 1
                    else:
                        with open(dest_img, 'wb') as f:
                            f.write(img_bytes)
                        v_write_ok += 1
                else:
                    v_copy_missing += 1
            except Exception as e:
                v_copy_error += 1
                print(f"[ERROR] 测试写入/复制失败 -> {dest_img}: {e}")
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
        print("[COPY] 训练图片复制/写入统计:")
        print(f"  成功复制: {t_copy_ok}")
        print(f"  成功写入: {t_write_ok}")
        print(f"  已存在跳过(复制): {t_copy_skip}")
        print(f"  已存在跳过(写入): {t_write_skip}")
        print(f"  源缺失: {t_copy_missing}")
        print(f"  失败: {t_copy_error}")
        print(f"训练目标目录: {dest_train_images_dir}")
        print("[COPY] 测试图片复制/写入统计:")
        print(f"  成功复制: {v_copy_ok}")
        print(f"  成功写入: {v_write_ok}")
        print(f"  已存在跳过(复制): {v_copy_skip}")
        print(f"  已存在跳过(写入): {v_write_skip}")
        print(f"  源缺失: {v_copy_missing}")
        print(f"  失败: {v_copy_error}")
        print(f"测试目标目录: {dest_test_images_dir}")
    else:
        print("[INFO] 未启用复制；请确保图片分别位于 MMCoIR-train/images/DATIKZ/images 与 MMCoIR-test/images/DATIKZ/images 下，或按 --images-name 自定义")

    # 打印一个示例
    if train_items:
        print("\n=== 训练样本示例（6 键）===")
        print(json.dumps(train_items[0], ensure_ascii=False, indent=2))
    if test_items:
        print("\n=== 测试样本示例（4 键）===")
        print(json.dumps(test_items[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()