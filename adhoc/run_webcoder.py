#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


def load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到图片：{p.resolve()}")
    img = Image.open(p).convert("RGB")
    return img


def main():
    parser = argparse.ArgumentParser(
        description="使用 xcodemind/webcoder 从图片与文字提示生成结果（通常为HTML/CSS）。"
    )
    parser.add_argument(
        "--image", "-i", required=True, help="输入图片路径，例如 ./web2code2m.png"
    )
    parser.add_argument(
        "--text",
        "-t",
        default="Generate HTML and CSS for this design.",
        help="可选的文字提示，将作为条件生成的 header 文本渲染到图片上方。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="生成的最大新token数量，HTML/CSS较长时可以适当增大（注意显存）。",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="运行设备，默认自动选择GPU可用则用cuda，否则cpu。",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型精度：auto/float16/bfloat16/float32（GPU建议float16或bfloat16）。",
    )
    args = parser.parse_args()

    # dtype 解析
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(">> 加载处理器与模型（第一次会从 Hugging Face 下载权重）...")
    processor = AutoProcessor.from_pretrained("xcodemind/webcoder")
    # 使用低精度可显著节省显存（需GPU支持）
    model = Pix2StructForConditionalGeneration.from_pretrained(
        "xcodemind/webcoder",
        torch_dtype=torch_dtype if torch_dtype is not None else None,
        device_map="auto" if args.device == "cuda" else None,
    )

    # 移动到设备
    if args.device == "cuda" and not hasattr(model, "hf_device_map"):
        model = model.to("cuda")

    # 读取图片
    image = load_image(args.image)

    # 条件生成：把 text 作为 header 文本与图像一起编码
    # 参考 transformers 文档对 Pix2Struct 的 conditional generation 用法
    # （processor(text=..., images=..., return_tensors="pt")）
    inputs = processor(
        text=args.text,
        images=image,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # 将张量放到模型同设备
    device_for_inputs = model.device if hasattr(model, "device") else (
        torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    )
    inputs = {k: v.to(device_for_inputs) for k, v in inputs.items()}

    print(">> 开始生成（可能需要一些时间，取决于显卡/CPU性能与max_new_tokens）...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,      # 需要更具确定性的结果时可保持False；想多样化可改True并设定温度
            num_beams=1,          # 也可以调大做束搜索，但会更慢更占显存
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    print("\n========== 生成结果（BEGIN） ==========")
    print(output_text)
    print("=========== 生成结果（END） ===========\n")


if __name__ == "__main__":
    main()
