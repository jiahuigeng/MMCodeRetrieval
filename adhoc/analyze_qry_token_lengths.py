import os
import json
import argparse
from typing import List, Tuple, Optional, Union

try:
    from transformers import AutoProcessor, AutoTokenizer
except Exception:
    AutoProcessor = None
    AutoTokenizer = None


def parse_model_spec(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse a model spec string like:
    "repo_or_path;backbone;optional_output_dir"
    Returns (repo_or_path, backbone, output_dir)
    """
    parts = [p.strip() for p in spec.split(';') if p.strip()]
    repo = parts[0] if len(parts) > 0 else ''
    backbone = parts[1] if len(parts) > 1 else None
    outdir = parts[2] if len(parts) > 2 else None
    return repo, backbone, outdir


def load_tokenizer(model_spec: str):
    repo, backbone, _ = parse_model_spec(model_spec)

    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
                return processor.tokenizer
        except Exception:
            pass

    if AutoTokenizer is not None:
        try:
            return AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from '{repo}': {e}")

    raise RuntimeError("transformers not available to load tokenizer.")


def iter_jsonl_files(root_dir: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.jsonl'):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _ensure_list_text(value: Union[str, List[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def compute_avg_token_length(jsonl_path: str, tokenizer, mode_hint: Optional[str] = None) -> Tuple[int, float, str]:
    """
    Compute average token length for a JSONL file.
    - If filename suggests 'test', use 'qry_text'.
    - If filename suggests 'train', use 'qry'.
    - Otherwise: prefer 'qry_text' if present, else 'qry'.

    Returns: (count, average_tokens, used_field)
    """
    fname = os.path.basename(jsonl_path).lower()
    if mode_hint is None:
        if 'test' in fname:
            mode_hint = 'test'
        elif 'train' in fname:
            mode_hint = 'train'

    used_field = 'qry_text' if mode_hint == 'test' else ('qry' if mode_hint == 'train' else None)

    total_tokens = 0
    count = 0
    used_field_final = None

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Auto-select field if not predetermined
            field = used_field
            if field is None:
                field = 'qry_text' if 'qry_text' in obj else ('qry' if 'qry' in obj else None)
            if field is None:
                continue

            texts = _ensure_list_text(obj.get(field))
            if not texts:
                continue
            used_field_final = field

            for text in texts:
                try:
                    ids = tokenizer.encode(text, add_special_tokens=True)
                    total_tokens += len(ids)
                    count += 1
                except Exception:
                    # Fallback: approximate by whitespace-split if tokenizer errors
                    total_tokens += len(text.split())
                    count += 1

    avg = (total_tokens / count) if count > 0 else 0.0
    return count, avg, (used_field_final or (used_field or 'unknown'))


def main():
    parser = argparse.ArgumentParser(description="Analyze average token counts for queries in JSONL datasets.")
    parser.add_argument('--model_spec', type=str, required=True,
                        help="Model spec like 'TIGER-Lab/VLM2Vec-Qwen2VL-7B;qwen2_vl;'. Only first segment is required.")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory containing JSONL files across subfolders.")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_spec)
    jsonl_files = iter_jsonl_files(args.data_dir)
    if not jsonl_files:
        print(f"No JSONL files found under: {args.data_dir}")
        return

    agg = {
        'train': {'total_tokens': 0, 'count': 0},
        'test': {'total_tokens': 0, 'count': 0},
        'other': {'total_tokens': 0, 'count': 0},
    }

    print(f"Found {len(jsonl_files)} JSONL files. Processing...\n")
    for fp in jsonl_files:
        count, avg, field = compute_avg_token_length(fp, tokenizer)
        base = os.path.relpath(fp, args.data_dir)
        tag = 'test' if 'test' in os.path.basename(fp).lower() else ('train' if 'train' in os.path.basename(fp).lower() else 'other')
        # For aggregation, recompute totals using avg*count
        agg[tag]['total_tokens'] += avg * count
        agg[tag]['count'] += count
        print(f"[File] {base} | type={tag} | field={field} | n={count} | avg_tokens={avg:.2f}")

    print("\n=== Aggregated ===")
    for tag in ['train', 'test', 'other']:
        total = agg[tag]['total_tokens']
        cnt = agg[tag]['count']
        avg = (total / cnt) if cnt > 0 else 0.0
        print(f"{tag}: n={cnt} | avg_tokens={avg:.2f}")


if __name__ == '__main__':
    main()