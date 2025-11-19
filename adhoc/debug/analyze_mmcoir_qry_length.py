import os
import json
import argparse
import statistics
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def find_subsets(base_dir: str) -> List[Tuple[str, str]]:
    """
    Walk base_dir and return list of (subset_name, test_jsonl_path) for directories
    that contain a test.jsonl file.
    """
    results = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    for entry in os.listdir(base_dir):
        subset_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(subset_dir):
            continue
        test_path = os.path.join(subset_dir, "test.jsonl")
        if os.path.isfile(test_path):
            results.append((entry, test_path))
    return sorted(results, key=lambda x: x[0])


def load_qry_lengths(test_jsonl_path: str) -> List[int]:
    """
    Read a JSONL file and return a list of character lengths for the 'qry_text' field.
    If 'qry_text' is missing or not a string, it will be coerced to str().
    """
    lengths = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed line
                continue
            text = obj.get("qry_text", "")
            # Coerce anything to string to be robust
            try:
                s = str(text)
            except Exception:
                s = ""
            lengths.append(len(s))
    return lengths


def describe(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "p95": 0,
        }
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    p95_idx = max(0, int(0.95 * (n - 1)))
    return {
        "count": n,
        "mean": float(statistics.mean(lengths_sorted)),
        "median": float(statistics.median(lengths_sorted)),
        "min": float(lengths_sorted[0]),
        "max": float(lengths_sorted[-1]),
        "p95": float(lengths_sorted[p95_idx]),
    }


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(summary: Dict[str, Dict[str, float]], output_csv: str):
    ensure_dir(os.path.dirname(output_csv))
    headers = ["subset", "count", "mean", "median", "min", "max", "p95"]
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for subset, stats in summary.items():
            row = [
                subset,
                str(int(stats["count"])),
                f"{stats['mean']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['min']:.0f}",
                f"{stats['max']:.0f}",
                f"{stats['p95']:.0f}",
            ]
            f.write(",".join(row) + "\n")


def plot_histograms(per_subset_lengths: Dict[str, List[int]], output_dir: str, bins: int = 50):
    ensure_dir(output_dir)
    # Per-subset histograms
    for subset, lengths in per_subset_lengths.items():
        if not lengths:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=bins, color="#4e79a7", alpha=0.85, edgecolor="white")
        plt.title(f"qry_text length distribution | {subset}")
        plt.xlabel("Length (characters)")
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"mmcoir_qry_len_hist_{subset}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    # Aggregated histogram across all subsets
    all_lengths = []
    for lengths in per_subset_lengths.values():
        all_lengths.extend(lengths)
    if all_lengths:
        plt.figure(figsize=(8, 5))
        plt.hist(all_lengths, bins=bins, color="#59a14f", alpha=0.85, edgecolor="white")
        plt.title("qry_text length distribution | ALL subsets")
        plt.xlabel("Length (characters)")
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "mmcoir_qry_len_hist_ALL.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze qry_text length distribution in MMCoIR_test subsets")
    parser.add_argument("--base_dir", type=str, default=os.path.join("data", "MMCoIR_test"),
                        help="Base directory containing subset folders, each with test.jsonl")
    parser.add_argument("--output_dir", type=str, default=os.path.join("adhoc", "debug", "mmcoir_qry_len_report"),
                        help="Output directory for CSV and plots")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins")
    args = parser.parse_args()

    subsets = find_subsets(args.base_dir)
    if not subsets:
        raise RuntimeError(f"No subsets with test.jsonl found under {args.base_dir}")

    per_subset_lengths: Dict[str, List[int]] = {}
    summary: Dict[str, Dict[str, float]] = {}
    for subset_name, test_path in subsets:
        lengths = load_qry_lengths(test_path)
        per_subset_lengths[subset_name] = lengths
        summary[subset_name] = describe(lengths)

    # Save CSV summary
    csv_path = os.path.join(args.output_dir, "mmcoir_qry_length_stats.csv")
    save_csv(summary, csv_path)

    # Plot histograms
    plot_histograms(per_subset_lengths, args.output_dir, bins=args.bins)

    # Simple console report
    print(f"Processed {len(subsets)} subsets under {args.base_dir}")
    print(f"CSV summary saved to: {csv_path}")
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()