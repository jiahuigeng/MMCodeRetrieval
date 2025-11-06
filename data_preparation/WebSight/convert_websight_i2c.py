#!/usr/bin/env python3
"""
将 HuggingFaceM4/WebSight（经 prepare_websight_subset.py 采样后保存到 datasets/WebSight/train|test）
转换为图片到代码（i2c）检索格式的 JSONL，并复制图片到统一目录。

输入（参考 prepare_websight_subset.py 输出）：
- 本地 Arrow 数据集：datasets/WebSight/train 与 datasets/WebSight/test
- 典型列："image"（HF Image 特征）、"raw_code"（字符串）。
  若无 raw_code，回退尝试 "raw"、"code"、"html"、"svg"、"text" 等列。
  若代码列是路径（如指向 .raw 文件），会尝试读取文件内容。

输出：
- 复制图片至：
  - 训练：MMCoIR-train/images/WebSight/images/<id>.png
  - 测试：MMCoIR-test/images/WebSight/images/<id>.png
- JSONL：
  - 训练：MMCoIR-train/WebSight_i2c/train.jsonl
  - 测试：MMCoIR-test/WebSight_i2c/test.jsonl

JSONL 字段（与 format_checker.py 一致）：
- 训练：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
- 测试：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}

设计约定：
- qry/qry_text 必须以 "<|image_1|>" 开头；后接简短指令（中文默认："请将该图片转换为代码。"）。
- 图片路径在 JSONL 中使用 POSIX 风格相对路径："WebSight/images/<id>.png"。
- 代码为正样本文本；目标图片为空占位（训练为空字符串，测试为 [""]）。

用法示例：
  python WebSight/convert_websight_i2c.py --limit-train 1000 --limit-test 200 \
    --overwrite-images --quiet

"""

import argparse
import json
import os
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_from_disk
from PIL import Image


IMG_TOKEN = "<|image_1|>"
DEFAULT_PROMPT = "Please convert this image to code."

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "datasets" / "WebSight"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DEFAULT_OUT_NAME = "WebSight_i2c"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _posix_rel(dataset_name: str, sample_id: str) -> str:
    # 统一以 images/ 开头，便于 format_checker 以默认 img_root 拼接
    return f"images/{dataset_name}/images/{sample_id}.png"


def _extract_image_ref(img_entry: Any) -> Tuple[str, Any]:
    """返回 (kind, value)，kind in {"path", "pil", "bytes", "unknown"}。
    兼容 HF Image 特征的常见结构。
    """
    if isinstance(img_entry, dict):
        b = img_entry.get("bytes")
        p = img_entry.get("path")
        if b is not None:
            return "bytes", b
        if isinstance(p, str) and p:
            return "path", p
        return "unknown", img_entry
    if hasattr(img_entry, "convert"):
        return "pil", img_entry
    if isinstance(img_entry, (bytes, bytearray)):
        return "bytes", img_entry
    if isinstance(img_entry, str):
        # 一些数据可能直接是路径字符串
        return "path", img_entry
    return "unknown", img_entry


def _save_image(img_kind: str, img_value: Any, dst_png: Path, overwrite: bool = False) -> bool:
    try:
        if dst_png.exists() and not overwrite:
            return True
        _ensure_dir(dst_png.parent)
        if img_kind == "path":
            src = Path(str(img_value))
            if src.exists():
                if src.suffix.lower() == ".png":
                    shutil.copyfile(src, dst_png)
                else:
                    im = Image.open(src)
                    im.save(dst_png, format="PNG")
                return True
            return False
        elif img_kind == "pil":
            im: Image.Image = img_value
            im.save(dst_png, format="PNG")
            return True
        elif img_kind == "bytes":
            im = Image.open(BytesIO(img_value))
            im.save(dst_png, format="PNG")
            return True
        else:
            return False
    except Exception:
        return False


def _read_text_file(path: Path) -> Optional[str]:
    try:
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return None
    return None


def _extract_code(sample: Dict[str, Any], input_dir: Path, code_field: str) -> Optional[str]:
    """从样本中提取代码文本。优先使用 code_field；若缺失，回退常见候选列。
    当值为路径（含 .raw/.html 等）时尝试读取文件内容。
    """
    candidates = [code_field, "raw", "raw_html", "code", "html", "svg", "text"]
    for key in candidates:
        if key not in sample:
            continue
        v = sample.get(key)
        if isinstance(v, str):
            # 路径形式：相对或绝对
            suffix = Path(v).suffix.lower()
            if suffix in {".raw", ".html", ".htm", ".json", ".txt"}:
                p = Path(v)
                if not p.is_absolute():
                    # 相对路径：基于 datasets/WebSight 下的 train/test 子目录进行猜测拼接
                    # 注：HF Image/文本常为绝对缓存路径，若为相对则回到 input_dir 逐级尝试
                    for sub in ("train", "test", ""):
                        pp = (input_dir / sub / v).resolve()
                        txt = _read_text_file(pp)
                        if txt is not None:
                            return txt
                # 绝对路径或已经存在的相对路径
                txt = _read_text_file(p)
                if txt is not None:
                    return txt
            else:
                # 普通字符串即为代码
                return v
        elif isinstance(v, dict):
            # 常见结构：{"path": "/.../file.raw", "bytes": b"..."}
            p = v.get("path")
            if isinstance(p, str) and p:
                txt = _read_text_file(Path(p))
                if txt is not None:
                    return txt
            b = v.get("bytes")
            if isinstance(b, (bytes, bytearray)):
                try:
                    return b.decode("utf-8", errors="ignore")
                except Exception:
                    pass
    return None


def _normalize_qry(prompt: str) -> str:
    # 强制图像 token 在最前面
    base = prompt or DEFAULT_PROMPT
    return f"{IMG_TOKEN}\n{base}"


def build_train_item(sample_id: str, img_rel: str, code_text: str, prompt: str) -> Dict[str, Any]:
    return {
        "qry": _normalize_qry(prompt),
        "qry_image_path": img_rel,
        "pos_text": str(code_text),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def build_test_item(sample_id: str, img_rel: str, code_text: str, prompt: str) -> Dict[str, Any]:
    return {
        "qry_text": _normalize_qry(prompt),
        "qry_img_path": img_rel,
        "tgt_text": [str(code_text)],
        "tgt_img_path": [""]
    }


def process_split(ds, images_dir: Path, dataset_name: str, split_tag: str, limit: Optional[int], input_dir: Path, code_field: str, prompt: str, overwrite: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """遍历一个 HF Dataset split，复制图片并构建训练/测试条目。
    返回 (train_rows, test_rows)。若 split_tag=="train" 仅填充 train_rows；若=="test" 仅填充 test_rows。
    """
    rows_train: List[Dict[str, Any]] = []
    rows_test: List[Dict[str, Any]] = []
    total = len(ds)
    n = min(limit or total, total)
    # 为了稳定命名，使用顺序索引（不打乱）。
    for i in range(n):
        ex = ds[i]
        # 提取代码
        code = _extract_code(ex, input_dir, code_field)
        if not isinstance(code, str) or not code.strip():
            # 跳过缺失代码的样本
            continue
        # 提取图片
        img_entry = ex.get("image") or ex.get("img") or ex.get("screenshot") or ex.get("rendered_image")
        kind, val = _extract_image_ref(img_entry)
        if kind == "unknown":
            # 无法识别图片结构则跳过
            continue
        sample_id = f"websight_{split_tag}_{i:06d}"
        dst_png = images_dir / f"{sample_id}.png"
        ok = _save_image(kind, val, dst_png, overwrite=overwrite)
        if not ok:
            continue
        rel_img = _posix_rel(dataset_name, sample_id)
        if split_tag == "train":
            rows_train.append(build_train_item(sample_id, rel_img, code, prompt))
        else:
            rows_test.append(build_test_item(sample_id, rel_img, code, prompt))
    return rows_train, rows_test


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert WebSight (saved Arrow) to i2c JSONL and copy images.")
    p.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help="datasets/WebSight 根目录（包含 train|test 子目录）")
    p.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="训练输出根目录（MMCoIR-train）")
    p.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="测试输出根目录（MMCoIR-test）")
    p.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME, help="数据集输出子目录名（默认 WebSight_i2c）")
    p.add_argument("--limit-train", type=int, default=1000, help="训练子集最多样本数（默认 1000）")
    p.add_argument("--limit-test", type=int, default=200, help="测试子集最多样本数（默认 200）")
    p.add_argument("--seed", type=int, default=42, help="预留种子参数（当前使用顺序采样，不随机）")
    p.add_argument("--code-field", type=str, default="raw_code", help="代码列名（默认 raw_code，回退尝试 raw/code/html/svg/text）")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="查询指令文本（将置于 <|image_1|> 之后）")
    p.add_argument("--overwrite-images", action="store_true", help="若目标图片已存在则覆盖")
    p.add_argument("--quiet", action="store_true", help="静默模式，减少控制台输出")
    args = p.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    out_name = args.out_name
    dataset_name = "WebSight"

    if not args.quiet:
        print(f"Input dir: {input_dir} | exists={input_dir.exists()}")
        print(f"Train root: {train_root}")
        print(f"Test  root: {test_root}")
        print(f"Out name:   {out_name}")
        print(f"Limits:     train={args.limit_train}, test={args.limit_test}")
        print(f"Code field: {args.code_field}")
        print(f"Prompt:     {args.prompt}")

    # 加载 Arrow 数据集
    train_ds = None
    test_ds = None
    try:
        train_ds = load_from_disk(str(input_dir / "train"))
    except Exception as e:
        print(f"[WARN] 加载 train 失败: {e}")
    try:
        test_ds = load_from_disk(str(input_dir / "test"))
    except Exception as e:
        print(f"[WARN] 加载 test 失败: {e}")

    # 目标图片目录：固定使用顶层 MMCoIR-train/test 下的 images 路径
    train_images_dir = DEFAULT_TRAIN_ROOT / "images" / dataset_name / "images"
    test_images_dir = DEFAULT_TEST_ROOT / "images" / dataset_name / "images"
    _ensure_dir(train_images_dir)
    _ensure_dir(test_images_dir)

    # 构建 JSONL 行
    rows_train: List[Dict[str, Any]] = []
    rows_test: List[Dict[str, Any]] = []
    if train_ds is not None:
        tr_rows, _ = process_split(
            train_ds,
            train_images_dir,
            dataset_name,
            split_tag="train",
            limit=args.limit_train,
            input_dir=input_dir,
            code_field=args.code_field,
            prompt=args.prompt,
            overwrite=args.overwrite_images,
        )
        rows_train = tr_rows
    if test_ds is not None:
        _, te_rows = process_split(
            test_ds,
            test_images_dir,
            dataset_name,
            split_tag="test",
            limit=args.limit_test,
            input_dir=input_dir,
            code_field=args.code_field,
            prompt=args.prompt,
            overwrite=args.overwrite_images,
        )
        rows_test = te_rows

    # 写入 JSONL
    train_jsonl = train_root / out_name / "train.jsonl"
    test_jsonl = test_root / out_name / "test.jsonl"
    if rows_train:
        write_jsonl(train_jsonl, rows_train)
        if not args.quiet:
            print(f"[SAVE] Train JSONL -> {train_jsonl} | rows={len(rows_train)}")
    else:
        if not args.quiet:
            print("[SAVE] Train JSONL skipped (no rows)")
    if rows_test:
        write_jsonl(test_jsonl, rows_test)
        if not args.quiet:
            print(f"[SAVE] Test JSONL  -> {test_jsonl} | rows={len(rows_test)}")
    else:
        if not args.quiet:
            print("[SAVE] Test JSONL skipped (no rows)")

    if not args.quiet:
        print("[DONE] WebSight i2c conversion completed.")


if __name__ == "__main__":
    main()