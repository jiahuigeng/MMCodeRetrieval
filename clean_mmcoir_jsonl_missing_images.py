#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 MMCoIR JSONL：当记录中的图片路径不存在时，删除对应行。

用法示例：
  python clean_mmcoir_jsonl_missing_images.py --dataset-dir MMCoIR-train/Web2Code --in-place
  python clean_mmcoir_jsonl_missing_images.py --dataset-dir MMCoIR-test/Web2Code --jsonl test.jsonl

说明：
- 自动检测常见图片键：train 使用 "qry_image_path"，test 使用 "qry_img_path"；也可通过 --image-key 指定。
- 路径解析：JSONL 中通常是相对路径如 "Web2Code/images/<id>.png"；脚本会将其视为相对于基目录（如 MMCoIR-train 或 MMCoIR-test）。
- 默认输出到同目录下的 "<原文件名>.cleaned.jsonl"；可使用 --in-place 覆写原文件（会生成 .bak 备份，除非禁用）。
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple


def detect_image_key(jsonl_path: Path, fallback: str = "qry_img_path") -> str:
    candidates = ("qry_img_path", "qry_image_path", "image_path", "img_path")
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for _ in range(1000):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for k in candidates:
                    v = obj.get(k)
                    if isinstance(v, str):
                        return k
    except Exception:
        pass
    return fallback


def resolve_mmcoir_image(base_root: Path, dataset_dir: Path, path_str: str) -> Optional[Path]:
    p = Path(path_str)
    # 绝对路径
    if p.is_absolute():
        return p if p.exists() else None
    # 常规：相对到基目录（MMCoIR-train/MMCoIR-test）
    candidate1 = base_root / p
    if candidate1.exists():
        return candidate1
    # 兼容：相对到数据集子目录（MMCoIR-*/Web2Code）
    candidate2 = dataset_dir / p
    if candidate2.exists():
        return candidate2
    # 仅使用文件名拼接 images 目录（兜底）
    candidate3 = dataset_dir / "images" / p.name
    if candidate3.exists():
        return candidate3
    return None


def clean_jsonl(dataset_dir: Path, jsonl_name: Optional[str], image_key: Optional[str], in_place: bool, backup: bool, dry_run: bool, verbose: bool, list_missing: bool = False, max_list: Optional[int] = None) -> Tuple[int, int, int, Path, Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir 不存在: {dataset_dir}")

    # 选择 JSONL 文件
    if jsonl_name:
        jsonl_path = dataset_dir / jsonl_name
    else:
        train_jsonl = dataset_dir / "train.jsonl"
        test_jsonl = dataset_dir / "test.jsonl"
        jsonl_path = train_jsonl if train_jsonl.exists() else test_jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")

    base_root = dataset_dir.parent  # MMCoIR-train / MMCoIR-test

    # 自动检测图片键
    if not image_key:
        image_key = detect_image_key(jsonl_path)
    print(f"[KEY] 使用图片键: {image_key}")

    # 目标输出路径
    if in_place:
        out_path = jsonl_path.with_suffix(".tmp.jsonl")
        backup_path = jsonl_path.with_suffix(".bak")
    else:
        out_path = jsonl_path.with_name(jsonl_path.stem + ".cleaned.jsonl")
        backup_path = jsonl_path.with_suffix(".bak")

    total = 0
    kept = 0
    removed = 0
    line_no = 0
    listed = 0

    # 清理流程
    fin = jsonl_path.open("r", encoding="utf-8")
    fout = None if dry_run else out_path.open("w", encoding="utf-8")
    for line in fin:
            s = line.strip()
            if not s:
                continue
            total += 1
            line_no += 1
            try:
                obj = json.loads(s)
            except Exception:
                # 非法 JSON 行，保留原样
                if not dry_run:
                    fout.write(line)
                kept += 1
                continue
            img_rel = obj.get(image_key)
            img_ok = False
            if isinstance(img_rel, str) and img_rel.strip():
                resolved = resolve_mmcoir_image(base_root, dataset_dir, img_rel)
                img_ok = resolved is not None
            if img_ok:
                if not dry_run:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
                if verbose and kept % 2000 == 0:
                    print(f"[keep] {kept}/{total}")
            else:
                removed += 1
                if list_missing and (max_list is None or listed < max_list):
                    sid = (
                        obj.get("id")
                        or obj.get("question_id")
                        or obj.get("sid")
                        or (Path(str(img_rel)).stem if isinstance(img_rel, str) else None)
                    )
                    print(f"[missing] line={line_no}, id={sid}, image={img_rel}")
                    listed += 1
                if verbose and removed % 2000 == 0:
                    print(f"[drop] {removed}/{total} -> {img_rel}")

    fin.close()

    # 应用 in-place 变更（dry-run 不做任何文件写入）
    if not dry_run:
        if in_place:
            if backup:
                try:
                    # 备份原文件
                    import shutil
                    shutil.copy2(jsonl_path, backup_path)
                    print(f"[BACKUP] {backup_path}")
                except Exception:
                    print("[WARN] 备份失败，继续覆盖原文件")
            try:
                out_path.replace(jsonl_path)
            except Exception as e:
                raise RuntimeError(f"覆盖原文件失败: {e}")
            final_out = jsonl_path
        else:
            final_out = out_path
    else:
        final_out = jsonl_path

    print(f"[SUMMARY] total={total}, kept={kept}, removed={removed}")
    print(f"[OUTPUT] {final_out}")
    return total, kept, removed, jsonl_path, final_out


def main():
    parser = argparse.ArgumentParser(description="清理 MMCoIR JSONL 中缺失图片路径的行")
    parser.add_argument("--dataset-dir", type=str, required=True, help="数据集子目录，如 MMCoIR-train/Web2Code 或 MMCoIR-test/Web2Code")
    parser.add_argument("--jsonl", type=str, default=None, help="目标 JSONL 文件名，默认优先使用 train.jsonl，否则 test.jsonl")
    parser.add_argument("--image-key", type=str, default=None, help="图片路径键名，默认自动检测（常见：qry_image_path / qry_img_path）")
    parser.add_argument("--in-place", action="store_true", help="原地覆盖 JSONL（默认写入 .cleaned.jsonl）")
    parser.add_argument("--no-backup", action="store_true", help="与 --in-place 一起使用时不生成 .bak 备份")
    parser.add_argument("--dry-run", action="store_true", help="仅输出统计，不写入结果文件")
    parser.add_argument("--list-missing", action="store_true", help="打印缺失图片对应的行号与样本ID")
    parser.add_argument("--max-list", type=int, default=None, help="限制缺失项打印数量，默认不限制")
    parser.add_argument("--verbose", action="store_true", help="输出详细保留/删除进度")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    jsonl_name = args.jsonl
    image_key = args.image_key

    try:
        clean_jsonl(
            dataset_dir,
            jsonl_name,
            image_key,
            in_place=args.in_place,
            backup=(not args.no_backup),
            dry_run=args.dry_run,
            verbose=args.verbose,
            list_missing=args.list_missing,
            max_list=args.max_list,
        )
    except Exception as e:
        print(f"[ERROR] 清理失败: {e}")


if __name__ == "__main__":
    main()