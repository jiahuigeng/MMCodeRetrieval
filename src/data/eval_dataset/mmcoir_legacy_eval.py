import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.model.processor import process_input_text


@add_metainfo_hook
def i2t_data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_text, qry_img_path, tgt_text in (
            zip(batch_dict['qry_text'], batch_dict['qry_img_path'], batch_dict['tgt_text'])):
        # Query side: use concatenated text and image
        # Ensure image token added for image-conditioned query
        q_text = process_input_text("", model_backbone, text=str(qry_text), add_image_token=True)
        # keep formatting consistent
        q_text = q_text.replace(" \n", "\n") + "\n"
        query_texts.append([q_text])
        qry_img_path = os.path.join(image_root, str(qry_img_path))
        query_images.append([{
            "bytes": [None],
            "paths": [qry_img_path],
            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)],
        }])

        # Candidate side: single correct text only (support list[str] -> take first)
        tgt_text_val = tgt_text[0] if isinstance(tgt_text, (list, tuple)) and len(tgt_text) > 0 else tgt_text
        cand_texts.append([str(tgt_text_val)])
        cand_images.append([None])
        # Use the text itself as candidate key; if duplicates exist, dedup happens in candidate generation
        dataset_infos.append({
            "cand_names": [str(tgt_text_val)],
            "label_name": str(tgt_text_val),
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


@add_metainfo_hook
def t2i_data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_text, tgt_img_path in (
            zip(batch_dict['qry_text'], batch_dict['tgt_img_path'])):
        # Query side: pure text query (already concatenated)
        q_text = process_input_text("", model_backbone, text=str(qry_text), add_image_token=False)
        q_text = q_text.replace(" \n", "\n") + "\n"
        query_texts.append([q_text])
        query_images.append([None])

        # Candidate side: image only; add image token on target side for VLMs
        cand_texts.append([process_input_text("", model_backbone, add_image_token=True)])
        tgt_img_val = tgt_img_path[0] if isinstance(tgt_img_path, (list, tuple)) and len(tgt_img_path) > 0 else tgt_img_path
        tgt_path_full = os.path.join(image_root, str(tgt_img_val))
        cand_images.append([ImageVideoInstance(bytes=[None], paths=[tgt_path_full],
                                               resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)]).to_dict()])
        dataset_infos.append({
            "cand_names": [str(tgt_img_val)],
            "label_name": str(tgt_img_val),
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


def _load_local_json_dataset(data_path, split):
    # Support file path or directory containing <split>.jsonl
    if os.path.isdir(data_path):
        data_file = os.path.join(data_path, f"{split}.jsonl")
    else:
        data_file = data_path
    dataset = load_dataset("json", data_files={split: data_file}, split=split)
    return dataset


DATASET_PARSER_NAME_I2T = "mmcoir_legacy_i2t"
DATASET_PARSER_NAME_T2I = "mmcoir_legacy_t2i"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME_I2T)
def load_mmcoir_legacy_i2t(model_args, data_args, *args, **kwargs):
    # Expected kwargs: data_path, dataset_split, image_root, optional num_sample_per_subset
    data_path = kwargs.get("data_path")
    dataset_split = kwargs.get("dataset_split", "test")
    if not data_path:
        raise ValueError("mmcoir_legacy_i2t requires 'data_path' in task_config (local JSONL path or directory)")

    dataset = _load_local_json_dataset(data_path, dataset_split)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and isinstance(num_sample_per_subset, str) and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if isinstance(num_sample_per_subset, int) and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: i2t_data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None


@AutoEvalPairDataset.register(DATASET_PARSER_NAME_T2I)
def load_mmcoir_legacy_t2i(model_args, data_args, *args, **kwargs):
    # Expected kwargs: data_path, dataset_split, image_root, optional num_sample_per_subset
    data_path = kwargs.get("data_path")
    dataset_split = kwargs.get("dataset_split", "test")
    if not data_path:
        raise ValueError("mmcoir_legacy_t2i requires 'data_path' in task_config (local JSONL path or directory)")

    dataset = _load_local_json_dataset(data_path, dataset_split)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and isinstance(num_sample_per_subset, str) and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if isinstance(num_sample_per_subset, int) and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: t2i_data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None