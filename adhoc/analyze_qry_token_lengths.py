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


def _token_len(tokenizer, text: str) -> int:
    try:
        ids = tokenizer.encode(text, add_special_tokens=True)
        return len(ids)
    except Exception:
        return len(text.split())


def compute_file_stats(jsonl_path: str, tokenizer, mode_hint: Optional[str] = None):
    """
    Compute token statistics for a JSONL file.
    - For files inferred as 'test': main field 'qry_text'; extra field 'tgt_text_first' (first element of list).
    - For files inferred as 'train': main field 'qry'; extra field 'pos_text' (all elements if list).
    - Otherwise: main field auto-select 'qry_text' if present else 'qry'. No extras.

    Returns a dict:
    {
      'tag': 'train'|'test'|'other',
      'main': {'field': str, 'count': int, 'total': int, 'avg': float},
      'extras': {
         'tgt_text_first': {'count': int, 'total': int, 'avg': float},
         'pos_text': {'count': int, 'total': int, 'avg': float}
      }
    }
    """
    fname = os.path.basename(jsonl_path).lower()
    if mode_hint is None:
        if 'test' in fname:
            mode_hint = 'test'
        elif 'train' in fname:
            mode_hint = 'train'
        else:
            mode_hint = 'other'

    used_field = 'qry_text' if mode_hint == 'test' else ('qry' if mode_hint == 'train' else None)

    main_total = 0
    main_count = 0
    used_field_final = None

    extra_tgt_total = 0
    extra_tgt_count = 0

    extra_pos_total = 0
    extra_pos_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Main field selection
            field = used_field
            if field is None:
                field = 'qry_text' if 'qry_text' in obj else ('qry' if 'qry' in obj else None)
            if field is not None:
                texts = _ensure_list_text(obj.get(field))
                if texts:
                    used_field_final = field
                    for text in texts:
                        main_total += _token_len(tokenizer, text)
                        main_count += 1

            # Extras
            if mode_hint == 'test':
                # tgt_text_first: use first element only if list, else use value
                tgt_val = obj.get('tgt_text')
                if isinstance(tgt_val, list):
                    if len(tgt_val) > 0 and tgt_val[0] is not None:
                        extra_tgt_total += _token_len(tokenizer, str(tgt_val[0]))
                        extra_tgt_count += 1
                elif tgt_val is not None:
                    extra_tgt_total += _token_len(tokenizer, str(tgt_val))
                    extra_tgt_count += 1

            if mode_hint == 'train':
                # pos_text: count all elements if list, else single value
                pos_texts = _ensure_list_text(obj.get('pos_text'))
                for pt in pos_texts:
                    extra_pos_total += _token_len(tokenizer, pt)
                    extra_pos_count += 1

    main_avg = (main_total / main_count) if main_count > 0 else 0.0
    stats = {
        'tag': mode_hint,
        'main': {
            'field': (used_field_final or (used_field or 'unknown')),
            'count': main_count,
            'total': main_total,
            'avg': main_avg,
        },
        'extras': {}
    }

    if mode_hint == 'test':
        stats['extras']['tgt_text_first'] = {
            'count': extra_tgt_count,
            'total': extra_tgt_total,
            'avg': (extra_tgt_total / extra_tgt_count) if extra_tgt_count > 0 else 0.0,
        }
    if mode_hint == 'train':
        stats['extras']['pos_text'] = {
            'count': extra_pos_count,
            'total': extra_pos_total,
            'avg': (extra_pos_total / extra_pos_count) if extra_pos_count > 0 else 0.0,
        }

    return stats


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

    agg_main = {
        'train': {'total_tokens': 0, 'count': 0},
        'test': {'total_tokens': 0, 'count': 0},
        'other': {'total_tokens': 0, 'count': 0},
    }
    agg_pos_train = {'total_tokens': 0, 'count': 0}
    agg_tgt_test = {'total_tokens': 0, 'count': 0}

    print(f"Found {len(jsonl_files)} JSONL files. Processing...\n")
    for fp in jsonl_files:
        stats = compute_file_stats(fp, tokenizer)
        base = os.path.relpath(fp, args.data_dir)
        tag = stats['tag']
        main = stats['main']
        # Aggregate main field
        agg_main[tag]['total_tokens'] += main['total']
        agg_main[tag]['count'] += main['count']
        print(f"[File] {base} | type={tag} | field={main['field']} | n={main['count']} | avg_tokens={main['avg']:.2f}")

        # Extras printing + aggregation
        if tag == 'test' and 'tgt_text_first' in stats['extras']:
            e = stats['extras']['tgt_text_first']
            agg_tgt_test['total_tokens'] += e['total']
            agg_tgt_test['count'] += e['count']
            print(f"  [Extra] tgt_text_first | n={e['count']} | avg_tokens={e['avg']:.2f}")
        if tag == 'train' and 'pos_text' in stats['extras']:
            e = stats['extras']['pos_text']
            agg_pos_train['total_tokens'] += e['total']
            agg_pos_train['count'] += e['count']
            print(f"  [Extra] pos_text | n={e['count']} | avg_tokens={e['avg']:.2f}")

    print("\n=== Aggregated (main) ===")
    for tag in ['train', 'test', 'other']:
        total = agg_main[tag]['total_tokens']
        cnt = agg_main[tag]['count']
        avg = (total / cnt) if cnt > 0 else 0.0
        print(f"{tag}: n={cnt} | avg_tokens={avg:.2f}")

    print("\n=== Aggregated (extras) ===")
    # Train pos_text
    t_total = agg_pos_train['total_tokens']
    t_cnt = agg_pos_train['count']
    t_avg = (t_total / t_cnt) if t_cnt > 0 else 0.0
    print(f"train.pos_text: n={t_cnt} | avg_tokens={t_avg:.2f}")
    # Test tgt_text_first
    s_total = agg_tgt_test['total_tokens']
    s_cnt = agg_tgt_test['count']
    s_avg = (s_total / s_cnt) if s_cnt > 0 else 0.0
    print(f"test.tgt_text_first: n={s_cnt} | avg_tokens={s_avg:.2f}")


if __name__ == '__main__':
    main()