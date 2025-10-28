#!/usr/bin/env python3
"""
任务说明（请勿擅自更改）
- 参考 convert_chart2code.py，将 datasets/chart2code_160k/json 下的所有 .json 合并并去重。
- 随机抽取 100000 个样本作为训练集，2000 个样本作为测试集（默认，可通过参数调整）。
- 输出：
  - 训练集 JSONL 到 `MMCoIR-train/Chart2Code/train.jsonl`
  - 测试集 JSONL 到 `MMCoIR-test/Chart2Code/test.jsonl`
- 复制图片：
  - 从 `datasets/chart2code_160k/images/` 复制训练集的图片到 `MMCoIR-train/Chart2Code/images/`
  - 从 `datasets/chart2code_160k/images/` 复制测试集的图片到 `MMCoIR-test/Chart2Code/images/`
- JSONL 中的图片路径字段从 `Chart2Code` 开始（例如：`Chart2Code/images/<filename>`）。
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path


def load_json(path: str):
    print(f"加载: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(items, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"保存 {len(items)} 样本 -> {path}")
    with open(path, 'w', encoding='utf-8') as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')


def extract_image_filename(sample) -> str:
    img = sample.get('image', '')
    if img.startswith('images/'):
        return img.split('/', 1)[1]
    return os.path.basename(img)


def convert_train(sample):
    conversations = sample['conversations']
    human = conversations[0]['value']
    gpt = conversations[1]['value']
    query_text = human.replace('<image>', '').strip()
    query = f"<|image_1|>\n{query_text}"
    fname = extract_image_filename(sample)
    return {
        "qry": query,
        "qry_image_path": f"Chart2Code/images/{fname}",
        "pos_text": gpt,
        "pos_image_path": "",
        "neg_text": "",
        "neg_image_path": "",
    }


def convert_test(sample):
    conversations = sample['conversations']
    human = conversations[0]['value']
    gpt = conversations[1]['value']
    query_text = human.replace('<image>', '').strip()
    query = f"<|image_1|>\n{query_text}"
    fname = extract_image_filename(sample)
    return {
        "qry_text": query,
        "qry_img_path": f"Chart2Code/images/{fname}",
        "tgt_text": [gpt],
        "tgt_img_path": [""],
    }


def copy_images(filenames, src_dir: str, dst_dir: str, tag: str):
    os.makedirs(dst_dir, exist_ok=True)
    total = len(filenames)
    missing = 0
    for i, fn in enumerate(filenames, 1):
        src = Path(src_dir) / fn
        dst = Path(dst_dir) / fn
        if not src.exists():
            missing += 1
            if missing <= 5:
                print(f"[警告] 源图片不存在: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            if i <= 5:
                print(f"[跳过] 拷贝失败: {src} -> {dst}: {e}")
        if i % 5000 == 0:
            print(f"[{tag}] 图片拷贝进度: {i}/{total}")
    if missing:
        print(f"[{tag}] 缺失图片数量: {missing}/{total}")


def main():
    parser = argparse.ArgumentParser(description="随机抽样拆分 Chart2Code 并复制图片")
    parser.add_argument("--input-dir", default=os.path.join("datasets", "chart2code_160k", "json"), help="源 JSON 目录")
    parser.add_argument("--src-images-dir", default=os.path.join("datasets", "chart2code_160k", "images"), help="源图片目录")
    parser.add_argument("--train-root", default=os.path.join("MMCoIR-train", "Chart2Code"), help="训练输出根目录")
    parser.add_argument("--test-root", default=os.path.join("MMCoIR-test", "Chart2Code"), help="测试输出根目录")
    parser.add_argument("--train-count", type=int, default=100_000, help="训练集样本数")
    parser.add_argument("--test-count", type=int, default=2_000, help="测试集样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 收集源 JSON 文件
    files = [os.path.join(args.input_dir, fn) for fn in os.listdir(args.input_dir) if fn.lower().endswith('.json')]
    if not files:
        raise SystemExit(f"未找到 JSON 文件于: {args.input_dir}")
    print(f"将合并 {len(files)} 个文件: {', '.join([os.path.basename(p) for p in files])}")

    # 合并与去重（按id）
    combined = []
    for fp in files:
        combined.extend(load_json(fp))
    print(f"合并后样本总数: {len(combined)}")

    unique = {}
    for s in combined:
        sid = s.get('id')
        if sid is None:
            # 兜底：使用图片文件名做去重键
            sid = f"{extract_image_filename(s)}"
        if sid not in unique:
            unique[sid] = s
    all_samples = list(unique.values())
    total = len(all_samples)
    print(f"去重后样本总数: {total}")

    # 随机抽样（不重叠）
    random.seed(args.seed)
    test_n = min(args.test_count, total)
    remaining_n = max(0, total - test_n)
    train_n = min(args.train_count, remaining_n)

    indices = list(range(total))
    test_idx = set(random.sample(indices, test_n))
    remain_idx = [i for i in indices if i not in test_idx]
    train_idx = set(random.sample(remain_idx, train_n))

    test_samples = [all_samples[i] for i in test_idx]
    train_samples = [all_samples[i] for i in train_idx]

    print(f"最终采样：train={len(train_samples)}，test={len(test_samples)}（总={total}）")

    # 转换格式
    train_conv = [convert_train(s) for s in train_samples]
    test_conv = [convert_test(s) for s in test_samples]

    # 输出路径
    train_json_path = os.path.join(args.train_root, 'train.jsonl')
    test_json_path = os.path.join(args.test_root, 'test.jsonl')
    train_images_dir = os.path.join(args.train_root, 'images')
    test_images_dir = os.path.join(args.test_root, 'images')

    # 保存 JSONL
    save_jsonl(train_conv, train_json_path)
    save_jsonl(test_conv, test_json_path)

    # 拷贝图片
    train_fns = [extract_image_filename(s) for s in train_samples]
    test_fns = [extract_image_filename(s) for s in test_samples]

    copy_images(train_fns, args.src_images_dir, train_images_dir, tag='train')
    copy_images(test_fns, args.src_images_dir, test_images_dir, tag='test')

    # 完成信息
    print("\n完成！")
    print(f"训练 JSONL: {train_json_path}")
    print(f"测试 JSONL:  {test_json_path}")
    print(f"训练图片目录: {train_images_dir}")
    print(f"测试图片目录:  {test_images_dir}")

    # 样例输出
    if train_conv:
        print("\n=== 训练样本示例 ===")
        print(json.dumps(train_conv[0], ensure_ascii=False, indent=2))
    if test_conv:
        print("\n=== 测试样本示例 ===")
        print(json.dumps(test_conv[0], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()