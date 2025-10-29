#!/usr/bin/env python3
"""
Download OmniSVG MMSVG datasets (Illustration and Icon) from Hugging Face.
- Saves each dataset to a dedicated folder with JSONL and optional SVG files.
- Supports limiting number of samples for quick validation.
- Supports Hugging Face token via --hf_token or env var HUGGINGFACE_HUB_TOKEN.

Example:
  python download_omnisvg.py --output_root datasets --limit 100 \
    --datasets OmniSVG/MMSVG-Illustration OmniSVG/MMSVG-Icon

If the datasets are gated, you must accept terms on Hugging Face and provide a token.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from typing import Optional, Iterable


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", name).strip("._") or "item"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_jsonl(records: Iterable[dict], jsonl_path: Path):
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_svg(svg_code: str, svg_path: Path, resume: bool = True):
    if resume and svg_path.exists():
        return
    svg_path.write_text(svg_code, encoding="utf-8")

def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def count_svg_files(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    try:
        return sum(1 for p in dir_path.glob("*.svg") if p.is_file())
    except Exception:
        return 0

def dataset_ready(
    output_root: Path,
    ds_slug: str,
    split: Optional[str],
    require_jsonl: bool,
    require_svg: bool,
    min_count: Optional[int] = None,
) -> bool:
    """Return True if local artifacts indicate dataset is already downloaded.

    Priority: JSONL presence is considered sufficient when requested.
    If JSONL is not requested, fall back to checking SVG files.
    """
    out_dir = output_root / ds_slug
    jsonl_ok = True
    svg_ok = True

    if require_jsonl:
        split_name = (split or "all").replace("/", "_")
        jsonl_path = out_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            jsonl_ok = False
        else:
            cnt = count_lines(jsonl_path)
            jsonl_ok = (cnt >= (min_count or 1))
        # JSONL presence is sufficient to skip
        if jsonl_ok:
            return True

    if require_svg:
        svgs_dir = out_dir / "svgs"
        svg_cnt = count_svg_files(svgs_dir)
        svg_ok = (svg_cnt >= (min_count or 1))
        if svg_ok:
            return True

    return False


def load_hf_dataset(dataset_id: str, split: Optional[str] = None):
    from datasets import load_dataset
    if split:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id)


def try_login(hf_token: Optional[str]):
    if hf_token:
        try:
            from huggingface_hub import login
            login(hf_token, add_to_git_credential=False)
            print("Logged in to Hugging Face Hub with provided token.")
        except Exception as e:
            eprint(f"Warning: failed to login with token: {e}")
    else:
        token_env = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if token_env:
            try:
                from huggingface_hub import login
                login(token_env, add_to_git_credential=False)
                print("Logged in to Hugging Face Hub with token from env.")
            except Exception as e:
                eprint(f"Warning: failed to login with env token: {e}")


def download_dataset(
    dataset_id: str,
    output_root: Path,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    save_svg: bool = True,
    save_jsonl_flag: bool = True,
    resume: bool = True,
):
    ds_slug = sanitize_filename(dataset_id.split("/")[-1])
    out_dir = output_root / ds_slug
    svgs_dir = out_dir / "svgs"
    # Skip entirely if artifacts already exist
    min_needed = limit if (limit is not None) else None
    if dataset_ready(output_root, ds_slug, split, save_jsonl_flag, save_svg, min_needed):
        print(f"[SKIP] {dataset_id} already present under {out_dir} (split={split or 'all'}).")
        return True
    ensure_dir(out_dir)
    if save_svg:
        ensure_dir(svgs_dir)

    print(f"Loading dataset: {dataset_id} split={split or 'all'}")
    try:
        dataset = load_hf_dataset(dataset_id, split)
    except Exception as e:
        eprint("\nERROR: Failed to load dataset. If it is gated, please:")
        eprint("  1) Accept the dataset terms on Hugging Face")
        eprint("  2) Provide a token via --hf_token or env HUGGINGFACE_HUB_TOKEN")
        eprint(f"Details: {e}\n")
        return False

    # Determine fields dynamically
    fields = None
    try:
        if hasattr(dataset, "features"):
            fields = list(dataset.features.keys())
    except Exception:
        pass

    print(f"Detected fields: {fields}")

    records = []
    total = len(dataset) if hasattr(dataset, "__len__") else None
    n = 0
    for i, ex in enumerate(dataset):
        rec = {"dataset": dataset_id}
        if isinstance(ex, dict):
            # Prefer canonical keys, but fall back to full example
            if "id" in ex:
                rec["id"] = str(ex["id"])
            else:
                rec["id"] = str(i)
            if "svg" in ex:
                rec["svg"] = ex["svg"]
            if "description" in ex:
                rec["description"] = ex["description"]
            # Keep any extra fields for completeness
            for k, v in ex.items():
                if k not in rec:
                    rec[k] = v
        else:
            rec["id"] = str(i)
            rec["data"] = ex

        if save_svg and ("svg" in rec) and isinstance(rec["svg"], str):
            svg_id = sanitize_filename(rec.get("id", str(i)))
            svg_path = svgs_dir / f"{svg_id}.svg"
            try:
                write_svg(rec["svg"], svg_path, resume=resume)
            except Exception as e:
                eprint(f"Warning: failed to write SVG {svg_path}: {e}")

        records.append(rec)
        n += 1
        if limit and n >= limit:
            break

    if save_jsonl_flag:
        jsonl_name = f"{(split or 'all').replace('/', '_')}.jsonl"
        jsonl_path = out_dir / jsonl_name
        try:
            save_jsonl(records, jsonl_path)
            print(f"Saved JSONL: {jsonl_path} ({len(records)} records)")
        except Exception as e:
            eprint(f"Warning: failed to save JSONL {jsonl_path}: {e}")

    if total is not None:
        print(f"Processed {n}/{total} examples from {dataset_id}.")
    else:
        print(f"Processed {n} examples from {dataset_id}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download OmniSVG MMSVG datasets to local JSONL/SVGS.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "OmniSVG/MMSVG-Illustration",
            "OmniSVG/MMSVG-Icon",
        ],
        help="Hugging Face dataset IDs to download.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="datasets",
        help="Output root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (e.g., train). Use empty to load all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per dataset for quick validation.",
    )
    parser.add_argument(
        "--no_save_svg",
        action="store_true",
        help="If set, do not write individual .svg files.",
    )
    parser.add_argument(
        "--no_save_jsonl",
        action="store_true",
        help="If set, do not write JSONL files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip writing existing SVG files (resume mode).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for gated datasets.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate setup and show intended actions without downloading.",
    )

    args = parser.parse_args()

    try_login(args.hf_token)

    output_root = Path(args.output_root)
    ensure_dir(output_root)

    if args.dry_run:
        print("Dry run mode: no data will be downloaded.")
        print(f"Datasets: {args.datasets}")
        print(f"Output root: {output_root}")
        print(f"Split: {args.split}")
        print(f"Limit: {args.limit}")
        print(f"Save SVG: {not args.no_save_svg}")
        print(f"Save JSONL: {not args.no_save_jsonl}")
        print("If datasets are gated, ensure you have accepted terms and provided a token via --hf_token or env HUGGINGFACE_HUB_TOKEN.")
        print("Done.")
        return

    ok_all = True
    for ds in args.datasets:
        ok = download_dataset(
            dataset_id=ds,
            output_root=output_root,
            split=(args.split or None),
            limit=args.limit,
            save_svg=(not args.no_save_svg),
            save_jsonl_flag=(not args.no_save_jsonl),
            resume=args.resume,
        )
        ok_all = ok_all and bool(ok)

    if not ok_all:
        eprint("One or more datasets failed. See errors above.")
        sys.exit(2)

    print("Done.")


if __name__ == "__main__":
    main()