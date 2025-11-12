#!/usr/bin/env python3
import os
import sys
import json
from typing import List, Tuple


def find_score_files(base_dir: str) -> List[str]:
    score_files = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith("_score.json"):
                score_files.append(os.path.join(root, f))
    return score_files


def extract_model_and_dataset(path: str, base_dir: str) -> Tuple[str, str]:
    # model name: immediate subdirectory under base_dir
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    model = parts[0] if len(parts) > 1 else "unknown_model"
    filename = os.path.basename(path)
    # dataset: filename without trailing "_score.json"
    if filename.endswith("_score.json"):
        dataset = filename[: -len("_score.json")]
    else:
        dataset = os.path.splitext(filename)[0]
    return model, dataset


def read_metrics(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lin = data.get("ndcg_linear@10")
        exp = data.get("ndcg_exponential@10")
        return lin, exp
    except Exception as e:
        return None, None


def main():
    # Base dir default: exps/mmcoir_legacy_test under repo root
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.join("exps", "mmcoir_legacy_test")

    if not os.path.isdir(base_dir):
        print(f"找不到目录: {base_dir}")
        sys.exit(1)

    files = find_score_files(base_dir)
    if not files:
        print(f"没有找到 *_score.json 文件于 {base_dir}")
        sys.exit(0)

    # Collect and sort by dataset name (alphabetical)
    rows = []
    for p in files:
        model, dataset = extract_model_and_dataset(p, base_dir)
        lin, exp = read_metrics(p)
        rows.append((model, dataset, lin, exp))

    rows.sort(key=lambda x: x[1].lower())

    # Print header and sorted rows
    print("model\tdataset\tndcg_linear@10\tndcg_exponential@10")
    for model, dataset, lin, exp in rows:
        lin_str = "" if lin is None else str(lin)
        exp_str = "" if exp is None else str(exp)
        print(f"{model}\t{dataset}\t{lin_str}\t{exp_str}")


if __name__ == "__main__":
    main()