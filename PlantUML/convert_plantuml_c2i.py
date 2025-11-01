#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PlantUML Activity (Act) 与 Sequence (Seq) 两个数据集转换为 code-to-image (c2i) 检索格式。

功能要点：
- 支持 Act / Seq 两个数据集，分别处理其 train / test 子集。
- 同名 .png 与 .txt 文件视为一一配对；从 .txt 读取代码作为查询文本。
- 复制所有目标图片到统一目录：
  - 训练集：MMCoIR-train/images/PlantUML_Act/images、MMCoIR-train/images/PlantUML_Seq/images
  - 测试集：MMCoIR-test/images/PlantUML_Act/images、MMCoIR-test/images/PlantUML_Seq/images
- 若图片位于 Train 下的 "1" 或 "2" 子文件夹，复制时将目标文件名改为 "1_原名.png" 或 "2_原名.png"；其他情况保持原始文件名。
- 生成的 JSONL 字段与 format_checker.py 要求一致：
  - 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
  - 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}
- 统一在 JSONL 中使用正斜杠路径（Linux/Ubuntu 友好）。

示例用法：
  python PlantUML/convert_plantuml_c2i.py --limit-train 50 --limit-test 50 --overwrite-images

可选参数：
- --limit-train N         限制每数据集的训练样本数量（默认无限制）
- --limit-test N          限制每数据集的测试样本数量（默认无限制）
- --subset {act,seq,both} 选择处理数据集（默认 both）
- --overwrite-images      允许覆盖已存在的已复制图片
- --quiet                 降低日志输出
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# 目录常量
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = REPO_ROOT / "datasets" / "PlantUML"
TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
TEST_ROOT = REPO_ROOT / "MMCoIR-test"


@dataclass
class PairItem:
    img_path: Path       # 原始 PNG 路径
    txt_path: Path       # 原始 TXT 路径
    train_subdir: Optional[str]  # 若位于 Train 下的子文件夹名（如 "1" 或 "2"），否则 None


def _is_png(p: Path) -> bool:
    return p.suffix.lower() == ".png"


def _match_txt(png_path: Path) -> Optional[Path]:
    """在同目录下查找与 png 同名的 .txt 文件。"""
    candidate = png_path.with_suffix(".txt")
    return candidate if candidate.exists() else None


def collect_pairs_act(split: str) -> List[PairItem]:
    base = DATASETS_ROOT / "Extra_Large_English_Act_Data_Total" / split.capitalize()
    items: List[PairItem] = []

    if not base.exists():
        return items

    if split.lower() == "train":
        # 训练集可能存在 1/ 与 2/ 子文件夹；也容忍其他层级，统一递归扫描
        for png in base.rglob("*.png"):
            txt = _match_txt(png)
            if not txt:
                continue
            # train_subdir 仅记录第一层子目录名是 "1" 或 "2"
            # 假设结构: Train/<subdir>/file.png
            train_subdir = None
            try:
                rel = png.relative_to(base)
                parts = rel.parts
                if len(parts) >= 2:
                    first = parts[0]
                    if first in {"1", "2"}:
                        train_subdir = first
            except Exception:
                pass
            items.append(PairItem(img_path=png, txt_path=txt, train_subdir=train_subdir))
    else:
        # 测试集通常直接平铺在 Test/ 下
        for png in base.glob("*.png"):
            txt = _match_txt(png)
            if not txt:
                continue
            items.append(PairItem(img_path=png, txt_path=txt, train_subdir=None))

    return items


def collect_pairs_seq(split: str) -> List[PairItem]:
    base = DATASETS_ROOT / "Extra_Large_English_Seq_Data_Total" / split.capitalize()
    items: List[PairItem] = []

    if not base.exists():
        return items

    if split.lower() == "train":
        for png in base.rglob("*.png"):
            txt = _match_txt(png)
            if not txt:
                continue
            train_subdir = None
            try:
                rel = png.relative_to(base)
                parts = rel.parts
                if len(parts) >= 2:
                    first = parts[0]
                    if first in {"1", "2"}:
                        train_subdir = first
            except Exception:
                pass
            items.append(PairItem(img_path=png, txt_path=txt, train_subdir=train_subdir))
    else:
        for png in base.glob("*.png"):
            txt = _match_txt(png)
            if not txt:
                continue
            items.append(PairItem(img_path=png, txt_path=txt, train_subdir=None))

    return items


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def make_dest_name(src_png: Path, train_subdir: Optional[str]) -> str:
    base_name = src_png.name  # 保留原始文件名（含扩展名）
    if train_subdir in {"1", "2"}:
        return f"{train_subdir}_{base_name}"
    return base_name


def copy_image(src: Path, dst_dir: Path, dest_name: str, overwrite: bool) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / dest_name
    if dst.exists() and not overwrite:
        return dst
    shutil.copy2(src, dst)
    return dst


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_posix_rel(*parts: str) -> str:
    return str(Path(*parts).as_posix())


def build_train_rows(items: List[PairItem], out_images_dir_rel: str, prompt: str) -> List[Dict]:
    rows: List[Dict] = []
    for it in items:
        dest_name = make_dest_name(it.img_path, it.train_subdir)
        rel_img = to_posix_rel("images", out_images_dir_rel, "images", dest_name)
        try:
            code = it.txt_path.read_text(encoding="utf-8")
        except Exception:
            code = ""
        rows.append({
            "qry": f"{prompt}\n{code}",
            "qry_image_path": "",
            "pos_text": "<|image_1|>",
            "pos_image_path": rel_img,
            "neg_text": "",
            "neg_image_path": "",
        })
    return rows


def build_test_rows(items: List[PairItem], out_images_dir_rel: str, prompt: str) -> List[Dict]:
    rows: List[Dict] = []
    for it in items:
        dest_name = make_dest_name(it.img_path, it.train_subdir)
        rel_img = to_posix_rel("images", out_images_dir_rel, "images", dest_name)
        try:
            code = it.txt_path.read_text(encoding="utf-8")
        except Exception:
            code = ""
        rows.append({
            "qry_text": f"{prompt}\n{code}",
            "qry_img_path": "",
            "tgt_text": ["<|image_1|>"],
            "tgt_img_path": [rel_img],
        })
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert PlantUML Act/Seq to c2i train/test JSONL and copy images")
    parser.add_argument("--subset", choices=["act", "seq", "both"], default="both", help="选择处理的数据集")
    parser.add_argument("--limit-train", type=int, default=100000, help="限制每数据集训练样本数量(默认100000，0表示全部)")
    parser.add_argument("--limit-test", type=int, default=2000, help="限制每数据集测试样本数量(默认2000，0表示全部)")
    parser.add_argument("--overwrite-images", action="store_true", help="覆盖已存在的已复制图片")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    args = parser.parse_args()

    subsets = []
    if args.subset in ("act", "both"):
        subsets.append("act")
    if args.subset in ("seq", "both"):
        subsets.append("seq")

    total_train = 0
    total_test = 0

    for subset in subsets:
        is_act = subset == "act"
        ds_label = "PlantUML_Act" if is_act else "PlantUML_Seq"
        out_dir_train = TRAIN_ROOT / f"{ds_label}_c2i"
        out_dir_test = TEST_ROOT / f"{ds_label}_c2i"
        out_images_train = TRAIN_ROOT / "images" / ds_label / "images"
        out_images_test = TEST_ROOT / "images" / ds_label / "images"

        prompt = (
            "Please convert this code to PlantUML activity diagram image."
            if is_act
            else "Please convert this code to PlantUML sequence diagram image."
        )

        # 收集与限制数量
        train_items = collect_pairs_act("train") if is_act else collect_pairs_seq("train")
        test_items = collect_pairs_act("test") if is_act else collect_pairs_seq("test")

        if args.limit_train and len(train_items) > args.limit_train:
            train_items = train_items[:args.limit_train]
        if args.limit_test and len(test_items) > args.limit_test:
            test_items = test_items[:args.limit_test]

        # 复制图片（训练集）
        for it in train_items:
            dest_name = make_dest_name(it.img_path, it.train_subdir)
            copy_image(it.img_path, out_images_train, dest_name, overwrite=args.overwrite_images)

        # 复制图片（测试集）
        for it in test_items:
            dest_name = make_dest_name(it.img_path, it.train_subdir)
            copy_image(it.img_path, out_images_test, dest_name, overwrite=args.overwrite_images)

        # 构建 JSONL 行（训练）
        train_rows = build_train_rows(
            train_items,
            out_images_dir_rel=ds_label,
            prompt=prompt,
        )

        # 构建 JSONL 行（测试）
        test_rows = build_test_rows(
            test_items,
            out_images_dir_rel=ds_label,
            prompt=prompt,
        )

        # 写入 JSONL
        write_jsonl(out_dir_train / "train.jsonl", train_rows)
        write_jsonl(out_dir_test / "test.jsonl", test_rows)

        total_train += len(train_rows)
        total_test += len(test_rows)

        if not args.quiet:
            print(f"[OK] {ds_label}: train {len(train_rows)} rows, test {len(test_rows)} rows")
            print(f"      images(train): {out_images_train}")
            print(f"      images(test):  {out_images_test}")

    if not args.quiet:
        print(f"Done. Total rows -> train: {total_train}, test: {total_test}")


if __name__ == "__main__":
    main()