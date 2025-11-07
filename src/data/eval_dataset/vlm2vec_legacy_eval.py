import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS
from src.model.processor import process_input_text
from typing import List


def _path_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _strict_validate_map(batch_dict, *, data_type: str, image_root: str):
    """
    Perform lightweight per-example validation based on data_type.
    - i2t: require qry_img_path exists; require tgt_text list non-empty
    - t2i: require tgt_img_path exists; qry_text present
    - t2t: require qry_text and tgt_text
    - ti2i: require qry_img_path and tgt_img_path exist
    - ti2ti: require qry_img_path and tgt_img_path exist
    This function returns the batch unchanged; raises ValueError on violations.
    """
    n = len(batch_dict.get('qry_text', []))
    # Helper to join image_root safely
    def join_root(p):
        return os.path.join(image_root, p) if image_root else p

    # Iterate per-row to build precise error messages
    for idx in range(n):
        # Common checks
        if 'qry_text' not in batch_dict:
            raise ValueError("Missing field 'qry_text' in dataset example.")

        if data_type in ['i2t', 'ti2t', 'ti2i', 'ti2ti']:
            if 'qry_img_path' not in batch_dict:
                raise ValueError("Missing field 'qry_img_path' for data_type requiring image on query side.")
            qpath = batch_dict['qry_img_path'][idx]
            qfull = join_root(qpath)
            if not isinstance(qpath, str) or not qpath.strip():
                raise ValueError(f"Invalid 'qry_img_path' at row {idx}: {qpath}")
            if not _path_exists(qfull):
                raise ValueError(f"Image not found for 'qry_img_path' at row {idx}: {qfull}")

        if data_type in ['i2t', 't2t', 'ti2t']:
            if 'tgt_text' not in batch_dict:
                raise ValueError("Missing field 'tgt_text' for text candidates.")
            tgt_texts: List[str] = batch_dict['tgt_text'][idx]
            if not isinstance(tgt_texts, list) or len(tgt_texts) == 0:
                raise ValueError(f"Invalid 'tgt_text' at row {idx}: expect non-empty list of strings.")

        if data_type in ['t2i', 'ti2i', 'ti2ti']:
            if 'tgt_img_path' not in batch_dict:
                raise ValueError("Missing field 'tgt_img_path' for image candidates.")
            tgt_paths: List[str] = batch_dict['tgt_img_path'][idx]
            if not isinstance(tgt_paths, list) or len(tgt_paths) == 0:
                raise ValueError(f"Invalid 'tgt_img_path' at row {idx}: expect non-empty list of image paths.")
            for p in tgt_paths:
                full = join_root(p)
                if not isinstance(p, str) or not p.strip():
                    raise ValueError(f"Invalid candidate image path at row {idx}: {p}")
                if not _path_exists(full):
                    raise ValueError(f"Candidate image not found at row {idx}: {full}")

    # Return batch unchanged; validation-only
    return batch_dict


@add_metainfo_hook
def _prepare_t2i(batch_dict, *args, **kwargs):
    """
    Legacy VLM2Vec text-to-image retrieval format:
    expects fields: `qry_inst`, `qry_text`, `tgt_inst`, `tgt_text`, `tgt_img_path`.
    - Query: text only
    - Candidates: images with optional captions
    """
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs.get('image_root', '')

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, tgt_inst, tgt_captions, tgt_img_paths in (
            zip(batch_dict.get('qry_inst', [''] * len(batch_dict['qry_text'])),
                batch_dict['qry_text'],
                batch_dict.get('tgt_inst', [''] * len(batch_dict['tgt_text'])),
                batch_dict['tgt_text'],
                batch_dict['tgt_img_path'])):
        # Build query text with image token (text-only query for t2i)
        qry_inst = qry_inst.replace("<|image_1|>", VLM_IMAGE_TOKENS[model_backbone])
        query_text = qry_inst + ' ' + qry_text + '\n'
        query_texts.append([query_text])
        query_images.append([None])

        # Candidate side: image list with optional caption processing
        if tgt_captions and tgt_captions[0].strip():
            # If captions exist, prepend instruction with image token to each
            tgt_inst = tgt_inst.replace("<|image_1|>", "")
            tgt_inst_captions = []
            for tgt_cap in tgt_captions:
                tgt_inst_caption = process_input_text(tgt_inst + ' ' + tgt_cap, model_backbone, text='', add_image_token=True)
                tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
                tgt_inst_captions.append(tgt_inst_caption)
            cand_texts.append(tgt_inst_captions)
        else:
            # No caption case: use pure instruction with image token
            tgt_inst = tgt_inst.replace("<|image_1|>", "")
            tgt_inst_caption = process_input_text(tgt_inst, model_backbone, text='', add_image_token=True)
            tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
            cand_texts.append([tgt_inst_caption] * len(tgt_img_paths))

        cand_img_paths = [os.path.join(image_root, p) if image_root else p for p in tgt_img_paths]
        img_list = [{"bytes": [None], "paths": [path],
                     "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for path in cand_img_paths]
        cand_images.append(img_list)
        dataset_infos.append({
            "cand_names": tgt_img_paths,
            "label_name": tgt_img_paths[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


@add_metainfo_hook
def _prepare_i2t(batch_dict, *args, **kwargs):
    """
    Legacy VLM2Vec image-to-text retrieval format:
    expects fields: `qry_inst`, `qry_text`, `qry_img_path`, `tgt_text`.
    - Query: image + text
    - Candidates: text list
    """
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs.get('image_root', '')

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, qry_img_path, tgt_texts in (
            zip(batch_dict.get('qry_inst', [''] * len(batch_dict['qry_text'])),
                batch_dict['qry_text'],
                batch_dict['qry_img_path'],
                batch_dict['tgt_text'])):
        # Strip legacy image token from instruction; re-inject proper token via processor
        qry_inst = qry_inst.replace("<|image_1|>", "")
        qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        query_texts.append([qry_text])
        qry_img_path = os.path.join(image_root, qry_img_path) if image_root else qry_img_path
        query_images.append([{ "bytes": [None], "paths": [qry_img_path],
                               "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] }])

        cand_texts.append(tgt_texts)
        cand_images.append([None] * len(tgt_texts))
        dataset_infos.append({
            "cand_names": tgt_texts,
            "label_name": tgt_texts[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


@add_metainfo_hook
def _prepare_t2t(batch_dict, *args, **kwargs):
    """
    Text-to-Text retrieval (both sides are text-only).
    expects fields: `qry_inst`, `qry_text`, `tgt_text`.
    - Query: text only (no image)
    - Candidates: text list (no image)
    """
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, tgt_texts in (
            zip(batch_dict.get('qry_inst', [''] * len(batch_dict['qry_text'])),
                batch_dict['qry_text'],
                batch_dict['tgt_text'])):

        qry_inst = qry_inst.replace("<|image_1|>", "").strip()
        # Do not add image tokens for pure text
        qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=False)
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        query_texts.append([qry_text])
        query_images.append([None])

        cand_texts.append(tgt_texts)
        cand_images.append([None] * len(tgt_texts))
        dataset_infos.append({
            "cand_names": tgt_texts,
            "label_name": tgt_texts[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


@add_metainfo_hook
def _prepare_ti2i(batch_dict, *args, **kwargs):
    """
    Text+Image to Image retrieval.
    expects fields: `qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`.
    - Query: image + text
    - Candidates: images with optional captions
    """
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs.get('image_root', '')

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, qry_img_path, tgt_inst, tgt_captions, tgt_img_paths in (
            zip(batch_dict.get('qry_inst', [''] * len(batch_dict['qry_text'])),
                batch_dict['qry_text'],
                batch_dict['qry_img_path'],
                batch_dict.get('tgt_inst', [''] * len(batch_dict['tgt_text'])),
                batch_dict['tgt_text'],
                batch_dict['tgt_img_path'])):

        # Query: add proper image token and include image
        qry_inst = qry_inst.replace("<|image_1|>", "").strip()
        qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        query_texts.append([qry_text])
        qry_img_path = os.path.join(image_root, qry_img_path) if image_root else qry_img_path
        query_images.append([{ "bytes": [None], "paths": [qry_img_path],
                               "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] }])

        # Candidates: images with optional captions
        if tgt_captions and tgt_captions[0].strip():
            tgt_inst = tgt_inst.replace("<|image_1|>", "")
            tgt_inst_captions = []
            for tgt_cap in tgt_captions:
                tgt_inst_caption = process_input_text(tgt_inst + ' ' + tgt_cap, model_backbone, text='', add_image_token=True)
                tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
                tgt_inst_captions.append(tgt_inst_caption)
            cand_texts.append(tgt_inst_captions)
        else:
            tgt_inst = tgt_inst.replace("<|image_1|>", "")
            tgt_inst_caption = process_input_text(tgt_inst, model_backbone, text='', add_image_token=True)
            tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
            cand_texts.append([tgt_inst_caption] * len(tgt_img_paths))

        cand_img_paths = [os.path.join(image_root, p) if image_root else p for p in tgt_img_paths]
        img_list = [{"bytes": [None], "paths": [path],
                     "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for path in cand_img_paths]
        cand_images.append(img_list)
        dataset_infos.append({
            "cand_names": tgt_img_paths,
            "label_name": tgt_img_paths[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


@add_metainfo_hook
def _prepare_ti2ti(batch_dict, *args, **kwargs):
    """
    Text+Image to Text+Image retrieval (both sides multimodal).
    expects fields: `qry_inst`, `qry_text`, `qry_img_path`, `tgt_inst`, `tgt_text`, `tgt_img_path`.
    - Query: image + text
    - Candidates: image + text list
    """
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs.get('image_root', '')

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, qry_img_path, tgt_inst, tgt_captions, tgt_img_paths in (
            zip(batch_dict.get('qry_inst', [''] * len(batch_dict['qry_text'])),
                batch_dict['qry_text'],
                batch_dict['qry_img_path'],
                batch_dict.get('tgt_inst', [''] * len(batch_dict['tgt_text'])),
                batch_dict['tgt_text'],
                batch_dict['tgt_img_path'])):

        # Query side
        qry_inst = qry_inst.replace("<|image_1|>", "").strip()
        qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        query_texts.append([qry_text])
        qry_img_path = os.path.join(image_root, qry_img_path) if image_root else qry_img_path
        query_images.append([{ "bytes": [None], "paths": [qry_img_path],
                               "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] }])

        # Candidate side: image + (optional) text
        tgt_inst = tgt_inst.replace("<|image_1|>", "")
        if tgt_captions and tgt_captions[0].strip():
            tgt_inst_captions = []
            for tgt_cap in tgt_captions:
                tgt_inst_caption = process_input_text(tgt_inst + ' ' + tgt_cap, model_backbone, text='', add_image_token=True)
                tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
                tgt_inst_captions.append(tgt_inst_caption)
            cand_texts.append(tgt_inst_captions)
        else:
            tgt_inst_caption = process_input_text(tgt_inst, model_backbone, text='', add_image_token=True)
            tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
            cand_texts.append([tgt_inst_caption] * len(tgt_img_paths))

        cand_img_paths = [os.path.join(image_root, p) if image_root else p for p in tgt_img_paths]
        img_list = [{"bytes": [None], "paths": [path],
                     "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for path in cand_img_paths]
        cand_images.append(img_list)
        dataset_infos.append({
            "cand_names": tgt_img_paths,
            "label_name": tgt_img_paths[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "vlm2vec_legacy"
DEFAULT_HF_PATH = "ziyjiang/MMEB_Test_Instruct"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_vlm2vec_legacy_dataset(model_args, data_args, *args, **kwargs):
    """
    A thin adapter to make current eval.py compatible with legacy VLM2Vec-style HF datasets.
    Required kwargs:
    - dataset_name: subset name (e.g., MSCOCO_t2i, VisDial, MSCOCO_i2t)
    Optional kwargs:
    - hf_path: HF dataset path (default: ziyjiang/MMEB_Test_Instruct)
    - data_type: one of {'t2i', 'i2t'}; if not given, inferred by suffix in dataset_name
    - image_root: base dir for images (joined to relative paths)
    - num_sample_per_subset: subsample count for quick runs
    """
    subset_name = kwargs["dataset_name"]
    hf_path = kwargs.get("hf_path", DEFAULT_HF_PATH)
    eval_type = kwargs.get("eval_type", None)
    strict_validate = kwargs.get("strict_validate", False)
    # infer data_type if not explicitly provided
    data_type = kwargs.get("data_type", None)
    if data_type is None:
        # Suffix-based inference first
        if subset_name.endswith("_t2i"):
            data_type = "t2i"
        elif subset_name.endswith("_i2t") or subset_name.endswith("_ti2t"):
            # treat i2t and ti2t the same (image+text -> text)
            data_type = "i2t"
        elif subset_name.endswith("_t2t"):
            data_type = "t2t"
        elif subset_name.endswith("_ti2i"):
            data_type = "ti2i"
        elif subset_name.endswith("_ti2ti"):
            data_type = "ti2ti"
        # Known names fallback for legacy sets
        elif subset_name in ["VisDial", "WebQA", "EDIS", "Wiki-SS-NQ", "VisualNews_t2i", "MSCOCO_t2i"]:
            data_type = "t2i"
        else:
            data_type = "i2t"

    # Prefer local JSONL loading:
    # 1) explicitly requested via eval_type: local
    # 2) OR auto-fallback if --data_basedir is set and local JSONL exists
    split = kwargs.get("dataset_split", "test")
    base_dir = data_args.data_basedir or ""
    jsonl_path = os.path.join(base_dir, subset_name, f"{split}.jsonl")
    use_local = (eval_type == "local") or (base_dir and os.path.exists(jsonl_path))
    if use_local:
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Local JSONL not found: {jsonl_path}. Please ensure --data_basedir points to MMCoIR-{split} root and dataset_name='{subset_name}' exists."
            )
        # Load local JSONL as HF dataset
        dataset = load_dataset("json", data_files={split: jsonl_path}, split=split)
    else:
        dataset = load_dataset(hf_path, subset_name, split=kwargs.get("dataset_split", "test"))
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", sys.maxsize))
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    prepare_map = {
        't2i': _prepare_t2i,
        'i2t': _prepare_i2t,
        't2t': _prepare_t2t,
        'ti2t': _prepare_i2t,   # alias to i2t
        'ti2i': _prepare_ti2i,
        'ti2ti': _prepare_ti2ti,
    }
    prepare_fn = prepare_map.get(data_type, _prepare_i2t)
    # Optional strict validation before preparation
    if strict_validate:
        dataset = dataset.map(lambda x: _strict_validate_map(x, data_type=data_type, image_root=kwargs.get('image_root', '')),
                              batched=True, batch_size=1024, num_proc=1,
                              drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.map(lambda x: prepare_fn(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    corpus = None  # No additional corpus
    return dataset, corpus