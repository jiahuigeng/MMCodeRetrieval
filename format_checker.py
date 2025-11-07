import argparse
import json
import os
from typing import Dict, List, Tuple


REQUIRED_KEYS_DEFAULT = [
    "qry",
    "qry_image_path",
    "pos_text",
    "pos_image_path",
    "neg_text",
    "neg_image_path",
]


def find_train_jsonl_files(root: str, split: str = "train") -> List[str]:
    """
    Recursively find all `<split>.jsonl` files under the given root directory.

    Example: root/data/MMCoIR_train/<subset>/<split>.jsonl
    """
    targets: List[str] = []
    split_filename = f"{split}.jsonl"
    for dirpath, dirnames, filenames in os.walk(root):
        if split_filename in filenames:
            targets.append(os.path.join(dirpath, split_filename))
    return sorted(targets)


def check_jsonl_file(file_path: str, required_keys: List[str]) -> Dict:
    """
    Check a JSONL file to ensure each record contains all required keys.

    Returns a dict with:
      - file_path
      - total_lines
      - valid_lines
      - invalid_lines
      - examples_missing (list of tuples: (line_no, missing_keys)) [up to 10]
      - decode_errors (list of line numbers that failed to parse)
    """
    total = 0
    valid = 0
    invalid = 0
    decode_errors: List[int] = []
    examples_missing: List[Tuple[int, List[str]]] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    # Treat empty lines as decode errors
                    decode_errors.append(idx)
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    decode_errors.append(idx)
                    continue

                missing = [k for k in required_keys if k not in obj]
                if missing:
                    invalid += 1
                    if len(examples_missing) < 10:
                        examples_missing.append((idx, missing))
                else:
                    valid += 1
    except FileNotFoundError:
        return {
            "file_path": file_path,
            "total_lines": 0,
            "valid_lines": 0,
            "invalid_lines": 0,
            "examples_missing": [],
            "decode_errors": [],
            "error": "file_not_found",
        }
    except Exception as e:
        return {
            "file_path": file_path,
            "total_lines": 0,
            "valid_lines": 0,
            "invalid_lines": 0,
            "examples_missing": [],
            "decode_errors": [],
            "error": f"unexpected_error: {e}",
        }

    return {
        "file_path": file_path,
        "total_lines": total,
        "valid_lines": valid,
        "invalid_lines": invalid,
        "examples_missing": examples_missing,
        "decode_errors": decode_errors,
    }


def print_report(results: List[Dict], required_keys: List[str]) -> None:
    """Print a human-readable summary report for all files checked."""
    print("== Format Checker Report ==")
    print(f"Required keys: {', '.join(required_keys)}")
    print("")

    total_files = len(results)
    ok_files = 0
    bad_files = 0
    total_lines = 0
    total_invalid = 0
    total_decode_errors = 0

    for r in results:
        file_path = r.get("file_path")
        error = r.get("error")
        if error:
            print(f"[ERROR] {file_path}: {error}")
            bad_files += 1
            continue

        total_lines += r["total_lines"]
        total_invalid += r["invalid_lines"]
        total_decode_errors += len(r["decode_errors"])

        if r["invalid_lines"] == 0 and len(r["decode_errors"]) == 0:
            print(f"[OK] {file_path}: {r['total_lines']} lines, all records valid")
            ok_files += 1
        else:
            print(
                f"[FAIL] {file_path}: total={r['total_lines']}, valid={r['valid_lines']}, "
                f"invalid={r['invalid_lines']}, decode_errors={len(r['decode_errors'])}"
            )
            bad_files += 1
            if r["examples_missing"]:
                print("  Examples of missing keys (line_no: missing_keys):")
                for line_no, missing in r["examples_missing"]:
                    print(f"    - {line_no}: {', '.join(missing)}")
            if r["decode_errors"]:
                preview = ", ".join(map(str, r["decode_errors"][:10]))
                more = "" if len(r["decode_errors"]) <= 10 else " ..."
                print(f"  JSON decode error lines: {preview}{more}")

    print("")
    print("== Summary ==")
    print(f"Files scanned: {total_files}")
    print(f"Files OK: {ok_files}")
    print(f"Files with issues: {bad_files}")
    print(f"Total lines: {total_lines}")
    print(f"Invalid records: {total_invalid}")
    print(f"Decode errors: {total_decode_errors}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check that each subset's JSONL contains the required keys. "
            "Recursively searches for <split>.jsonl under the given root."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing subsets, e.g., data/MMCoIR_train",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="JSONL split filename prefix (default: train → train.jsonl)",
    )
    parser.add_argument(
        "--required-keys",
        type=str,
        nargs="*",
        default=REQUIRED_KEYS_DEFAULT,
        help=(
            "Keys required in each JSONL record. Default: "
            + ", ".join(REQUIRED_KEYS_DEFAULT)
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    files = find_train_jsonl_files(root, split=args.split)
    if not files:
        print(
            f"No '{args.split}.jsonl' files found under root: {root}. "
            "Check the path and split name."
        )
        return

    results = [check_jsonl_file(fp, args.required_keys) for fp in files]
    print_report(results, args.required_keys)


if __name__ == "__main__":
    main()