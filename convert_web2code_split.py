#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考 convert_omnisvg_split.py，编写 Web2Code 转换脚本：

- 明确数据源：训练从 Web2Code.json，测试从 Web2Code_eval.jsonl（均位于 datasets/Web2Code）
- 训练集：提取包含 image 与 conversations 的样本，conversations[human]->conversations[gpt/assistant] 形成 (q, a)
- 测试集：支持多种格式回退解析（无 conversations 时，尝试 question/answer、instruction/output、messages 等）
- 采样 train=100000、test=2000（不足则取可用最大值）
- q 与图像 token 组合形成检索查询文本：训练用 "qry"，测试用 "qry_text"
- 复制图片到 MMCoIR-train/Web2Code/images 与 MMCoIR-test/Web2Code/images，统一相对路径为 Web2Code/images/<id>.png
- 输出 JSONL：
  * 训练集：{"qry", "qry_image_path", "pos_text", "pos_image_path", "neg_text", "neg_image_path"}
  * 测试集：{"qry_text", "qry_img_path", "tgt_text", "tgt_img_path"}（tgt_* 为列表）

注意：
- 若 human 文本中包含 "<image>"，将其替换为项目统一的 "<|image_1|>"；若未包含，则自动在前缀添加 "<|image_1|>\n"。
- 图片路径解析支持多种目录回退：<input-dir>/<image>、<input-dir>/Web2Code_image/<image> 等。
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple, Set

REPO_ROOT = Path(__file__).parent
DEFAULT_INPUT_DIR = REPO_ROOT / "datasets" / "Web2Code"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "MMCoIR-train"
DEFAULT_TEST_ROOT = REPO_ROOT / "MMCoIR-test"
DATASET_NAME = "Web2Code"
IMG_DST_SUBDIR = "images"
DEFAULT_TRAIN_FILE = "Web2Code.json"
DEFAULT_TEST_FILE = "Web2Code_eval.jsonl"


def normalize_image_token(q_text: str) -> str:
    """将数据集中出现的 "<image>" 替换为项目统一 token "<|image_1|>"；
    若未包含任何图像 token，则在前面加上 "<|image_1|>\n"。
    """
    q_text = q_text or ""
    if "<|image_1|>" in q_text:
        return q_text
    if "<image>" in q_text:
        return q_text.replace("<image>", "<|image_1|>")
    return "<|image_1|>\n" + q_text


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    def _iter() -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield obj
    return _iter()


def read_json(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 一些数据仓库会用 {"data": [...]} 包裹
            for key in ("data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        else:
            return []
    except Exception:
        return []


def resolve_image_path(root_dir: Path, img_rel: str) -> Optional[Path]:
    """在多种可能的目录下解析图片相对路径。若找不到，返回 None。"""
    candidates = [
        root_dir / img_rel,
        root_dir / "Web2Code_image" / img_rel,
        root_dir / "images" / img_rel,
        root_dir / "Web2Code_eval_image" / img_rel,
        root_dir / "Web2Code_images" / img_rel,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def ensure_dirs(train_root: Path, test_root: Path, dataset: str) -> Tuple[Path, Path]:
    train_images = train_root / dataset / IMG_DST_SUBDIR
    test_images = test_root / dataset / IMG_DST_SUBDIR
    train_images.mkdir(parents=True, exist_ok=True)
    test_images.mkdir(parents=True, exist_ok=True)
    return train_images, test_images


def copy_image(src: Path, dst: Path) -> bool:
    try:
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 使用 shutil.copy2 保留基本元数据
        import shutil
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def to_train_item(sample_id: str, q_text: str, a_text: str) -> Dict[str, Any]:
    rel_img = f"{DATASET_NAME}/{IMG_DST_SUBDIR}/{sample_id}.png"
    return {
        "qry": normalize_image_token(q_text),
        "qry_image_path": rel_img,
        "pos_text": str(a_text),
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def to_test_item(sample_id: str, q_text: str, a_text: str) -> Dict[str, Any]:
    rel_img = f"{DATASET_NAME}/{IMG_DST_SUBDIR}/{sample_id}.png"
    return {
        "qry_text": normalize_image_token(q_text),
        "qry_img_path": rel_img,
        "tgt_text": [str(a_text)],
        "tgt_img_path": [""]
    }


def extract_test_qa(obj: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """支持多种测试数据结构的 (q, a) 提取：
    1) conversations: 与训练相同的结构
    2) question/answer
    3) instruction/output
    4) messages: list[ {role: user|assistant, content/text/value: str} ]
    5) prompt/completion
    返回 (q, a) 或 None。
    """
    # 1) conversations
    conv = obj.get("conversations")
    qa = None
    if isinstance(conv, list) and len(conv) >= 2:
        qa = extract_qa(conv)
        if qa:
            return qa
    # 2) question/answer
    q = obj.get("question")
    a = obj.get("answer")
    if isinstance(q, str) and isinstance(a, str):
        return q, a
    # 3) instruction/output
    q = obj.get("instruction")
    a = obj.get("output")
    if isinstance(q, str) and isinstance(a, str):
        return q, a
    # 4) messages
    msgs = obj.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        q1 = None
        a1 = None
        for m in msgs:
            role = (m or {}).get("role") or (m or {}).get("from")
            val = (m or {}).get("content") or (m or {}).get("text") or (m or {}).get("value")
            if role in ("user", "human") and isinstance(val, str) and q1 is None:
                q1 = val
            if role in ("assistant", "gpt") and isinstance(val, str) and a1 is None:
                a1 = val
        if isinstance(q1, str) and isinstance(a1, str):
            return q1, a1
    # 5) prompt/completion
    q = obj.get("prompt")
    a = obj.get("completion")
    if isinstance(q, str) and isinstance(a, str):
        return q, a
    return None


def extract_qa(conv: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """从 conversations 列表中提取 (q, a)。优先选择前两个消息：human -> gpt/assistant。"""
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    q = None
    a = None
    # 选择第一个 human
    for msg in conv:
        role = (msg or {}).get("from") or (msg or {}).get("role")
        val = (msg or {}).get("value") or (msg or {}).get("text")
        if role in ("human", "user") and isinstance(val, str):
            q = val
            break
    # 在找到 human 之后，找第一条 gpt/assistant
    found_q = q is not None
    if found_q:
        for msg in conv:
            role = (msg or {}).get("from") or (msg or {}).get("role")
            val = (msg or {}).get("value") or (msg or {}).get("text")
            if role in ("gpt", "assistant") and isinstance(val, str):
                a = val
                break
    if q and a:
        return q, a
    return None


def process_web2code(
    input_dir: Path,
    train_file: Path,
    test_file: Path,
    train_root: Path,
    test_root: Path,
    train_count: int,
    test_count: int,
    seed: int,
    scan_limit: Optional[int] = None,
    verbose_scan: bool = False,
) -> None:
    print(f"Input dir: {input_dir} {'(exists)' if input_dir.exists() else '(missing)'}")
    print(f"Train root: {train_root}")
    print(f"Test root:  {test_root}")

    if not input_dir.exists():
        print("[ERROR] 输入目录不存在，请先运行 download_web2code.py 下载本地快照。")
        return
    # 训练源：仅读取 train_file（默认 Web2Code.json）
    print(f"[SRC] Train file: {train_file} {'(exists)' if train_file.exists() else '(missing)'}")
    train_iter: Iterable[Dict[str, Any]] = []
    if train_file.suffix.lower() == ".jsonl":
        train_iter = read_jsonl(train_file)
    else:
        train_iter = read_json(train_file)
    usable_train: List[Tuple[str, str, str, Path]] = []  # (id, q, a, img_src)
    miss_train_img = 0
    miss_train_qa = 0
    for obj in train_iter:
        img_rel = obj.get("image")
        conv = obj.get("conversations")
        qa = extract_qa(conv)
        if qa is None:
            miss_train_qa += 1
            continue
        q, a = qa
        img_src = resolve_image_path(input_dir, str(img_rel)) if isinstance(img_rel, str) else None
        if img_src is None:
            miss_train_img += 1
            continue
        rec_id = obj.get("id") or Path(str(img_rel)).stem
        usable_train.append((str(rec_id), q, a, img_src))
    print(f"[INFO] 训练可用记录: {len(usable_train)} (缺失QA: {miss_train_qa}, 缺失图片: {miss_train_img})")
    if not usable_train:
        print("[WARN] 训练集没有可用样本；已退出。")
        return

    # 测试源：仅读取 test_file（默认 Web2Code_eval.jsonl）
    print(f"[SRC] Test  file: {test_file} {'(exists)' if test_file.exists() else '(missing)'}")
    test_iter: Iterable[Dict[str, Any]] = []
    if test_file.suffix.lower() == ".jsonl":
        test_iter = read_jsonl(test_file)
    else:
        test_iter = read_json(test_file)
    usable_test: List[Tuple[str, str, str, Path]] = []
    miss_test_img = 0
    miss_test_qa = 0
    for obj in test_iter:
        qa = extract_test_qa(obj)
        if qa is None:
            miss_test_qa += 1
            continue
        q, a = qa
        img_rel = obj.get("image") or obj.get("image_path") or obj.get("img")
        img_src = resolve_image_path(input_dir, str(img_rel)) if isinstance(img_rel, str) else None
        if img_src is None:
            miss_test_img += 1
            continue
        rec_id = obj.get("id") or Path(str(img_rel)).stem
        usable_test.append((str(rec_id), q, a, img_src))
    print(f"[INFO] 测试可用记录: {len(usable_test)} (缺失QA: {miss_test_qa}, 缺失图片: {miss_test_img})")
    if not usable_test:
        print("[WARN] 测试集没有可用样本；已退出。")
        return

    # 采样（确定性）
    rnd = random.Random(seed)
    rnd.shuffle(usable_train)
    rnd.shuffle(usable_test)
    train_n = min(train_count, len(usable_train))
    test_n = min(test_count, len(usable_test))
    train_sel = usable_train[:train_n]
    test_sel = usable_test[:test_n]
    print(f"[PLAN] Train: {len(train_sel)} | Test: {len(test_sel)} (requested: {train_count}/{test_count})")

    # 目标路径
    out_train_json = train_root / DATASET_NAME / "train.jsonl"
    out_test_json = test_root / DATASET_NAME / "test.jsonl"
    out_train_json.parent.mkdir(parents=True, exist_ok=True)
    out_test_json.parent.mkdir(parents=True, exist_ok=True)

    # 复制图片
    train_images_dir, test_images_dir = ensure_dirs(train_root, test_root, DATASET_NAME)
    print(f"[COPY] Train images -> {train_images_dir}")
    miss_train = 0
    for i, (sid, _q, _a, img_src) in enumerate(train_sel, 1):
        dst = train_images_dir / f"{sid}.png"
        if not copy_image(img_src, dst):
            miss_train += 1
        if i % 5000 == 0:
            print(f"  [train] {i}/{len(train_sel)}")
    if miss_train:
        print(f"[WARN] 训练集图片缺失或复制失败: {miss_train}/{len(train_sel)}")

    print(f"[COPY] Test images  -> {test_images_dir}")
    miss_test = 0
    for i, (sid, _q, _a, img_src) in enumerate(test_sel, 1):
        dst = test_images_dir / f"{sid}.png"
        if not copy_image(img_src, dst):
            miss_test += 1
        if i % 2000 == 0:
            print(f"  [test] {i}/{len(test_sel)}")
    if miss_test:
        print(f"[WARN] 测试集图片缺失或复制失败: {miss_test}/{len(test_sel)}")

    # 构建并保存 JSONL
    print(f"[SAVE] Train JSONL -> {out_train_json}")
    with out_train_json.open("w", encoding="utf-8") as f:
        for sid, q, a, _img in train_sel:
            item = to_train_item(sid, q, a)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SAVE] Test JSONL  -> {out_test_json}")
    with out_test_json.open("w", encoding="utf-8") as f:
        for sid, q, a, _img in test_sel:
            item = to_test_item(sid, q, a)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] Web2Code split completed.")
    print(f"Train JSONL: {out_train_json}")
    print(f"Test JSONL:  {out_test_json}")
    print(f"Train images: {train_images_dir}")
    print(f"Test images:  {test_images_dir}")


def main():
    parser = argparse.ArgumentParser(description="将 MBZUAI/Web2Code 转为检索训练/测试 JSONL 并复制图片")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Web2Code 本地数据快照目录（通过 download_web2code.py 下载）")
    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE, help="训练源文件（默认 Web2Code.json）")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE, help="测试源文件（默认 Web2Code_eval.jsonl）")
    parser.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT), help="输出训练根目录（默认 MMCoIR-train）")
    parser.add_argument("--test-root", type=str, default=str(DEFAULT_TEST_ROOT), help="输出测试根目录（默认 MMCoIR-test）")
    parser.add_argument("--train-count", type=int, default=100_000, help="训练采样数（默认 100000）")
    parser.add_argument("--test-count", type=int, default=2_000, help="测试采样数（默认 2000）")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子（默认 42）")
    parser.add_argument("--scan-limit", type=int, default=None, help="最多扫描的原始样本数（默认全部）")
    parser.add_argument("--verbose-scan", action="store_true", help="扫描时打印文件路径")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    train_file = Path(args.train_file if args.train_file else DEFAULT_TRAIN_FILE)
    test_file = Path(args.test_file if args.test_file else DEFAULT_TEST_FILE)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    process_web2code(
        input_dir=input_dir,
        train_file=train_file,
        test_file=test_file,
        train_root=train_root,
        test_root=test_root,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
        scan_limit=args.scan_limit,
        verbose_scan=args.verbose_scan,
    )


if __name__ == "__main__":
    main()