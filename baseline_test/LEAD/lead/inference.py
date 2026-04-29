"""
推理流程：输入构建、单样本推理。

此文件从 run_qwen25vl_example.py 提取而来，
核心推理逻辑（LEAD/CoT 调用、输入处理、输出解码）与原始实现完全一致。
"""

import argparse
from typing import Dict, Any

import torch
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from .generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_cot,
    generate_lead,
)


def prepare_inputs(
    processor: AutoProcessor,
    messages,
    device: torch.device,
) -> Dict[str, Any]:
    """
    将对话消息编码为模型可接受的输入张量。

    Args:
        processor: Qwen2.5-VL 处理器。
        messages: 对话消息列表（含 image/text 内容）。
        device: 目标设备。

    Returns:
        dict: 包含 input_ids、attention_mask、pixel_values 等张量。
    """
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    encoded = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    encoded = encoded.to(device)
    return dict(encoded)


def run_single_inference(
    model,
    processor,
    tokenizer,
    args: argparse.Namespace,
    verbose: bool = True,
) -> str:
    """
    对单个样本执行推理，返回模型生成文本。

    核心逻辑与原始 run_qwen25vl_example.py 中的 main() 完全一致，
    仅将 model/processor/tokenizer 从全局变量改为显式参数传入。

    Args:
        model: 已加载的 Qwen2.5-VL 模型。
        processor: 对应的处理器。
        tokenizer: 对应的分词器。
        args: 命令行参数（包含 image, prompt, method, alpha 等）。

    Returns:
        str: 模型生成的文本（已去除首尾空白）。
    """
    set_seed(args.seed)
    compute_device = args.device
    if compute_device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(compute_device)

    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    model_inputs = prepare_inputs(processor, messages, device)

    if verbose:
        print("input_ids len:", model_inputs["input_ids"].shape[1])
        if "image_grid_thw" in model_inputs:
            thw = model_inputs["image_grid_thw"]
            print("image_grid_thw:", thw.tolist())

    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)

    math_ids_set = get_math_symbols_ids(tokenizer)
    math_ids_tensor = (
        torch.tensor(list(math_ids_set), device=device) if math_ids_set else None
    )

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "verbose": verbose,
    }

    if args.method == "cot_greedy":
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        if args.method == "lead":
            if math_ids_tensor is not None:
                model_inputs["math_ids_tensor"] = math_ids_tensor
            model_inputs["alpha_0"] = args.alpha
            model_inputs["max_switch_count"] = args.max_switch_count
            model_inputs["convergence_words"] = "</think>"
            outputs = generate_lead(
                model,
                tokenizer,
                **model_inputs,
                **gen_kwargs,
            )
        else:
            outputs = generate_cot(
                model,
                tokenizer,
                **model_inputs,
                **gen_kwargs,
            )

    prompt_len = model_inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if verbose:
        print("=== Prompt ===")
        print(args.prompt)
        print("=== Model Output ===")
        print(generated_text.strip())

    return generated_text.strip()
