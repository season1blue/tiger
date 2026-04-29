import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, set_seed


def normalize_yes_no(text: str) -> str:
    lowered = text.strip().lower()
    match = re.search(r"\b(yes|no)\b", lowered)
    if match is not None:
        return match.group(1)
    if re.search(r"\b(no|not|n't|none|never)\b", lowered):
        return "no"
    return "yes"


def build_inputs(processor, model, image_path: str, question: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def main(args):
    model_path = os.path.expanduser(args.model_path)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    ).eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            question = line["text"]
            prompt = f"{question} Please answer yes or no only."

            image_path = os.path.join(args.image_folder, image_file)
            inputs = build_inputs(processor, model, image_path, prompt)

            if args.use_visaug:
                # Placeholder to keep ClearSight CLI compatibility for Qwen2.5-VL.
                # The original AttnAdapter is tied to LLaVA LlamaAttention modules.
                _ = (args.enh_para, args.sup_para)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            input_token_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, input_token_len:]
            raw_text = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            text = normalize_yes_no(raw_text)
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": prompt,
                        "text": text,
                        "raw_text": raw_text,
                        "model_id": "qwen2_5_vl",
                        "image": image_file,
                        "metadata": {
                            "use_visaug": bool(args.use_visaug),
                            "enh_para": args.enh_para,
                            "sup_para": args.sup_para,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--use-visaug", action="store_true", default=False)
    parser.add_argument("--enh-para", type=float, default=1.15)
    parser.add_argument("--sup-para", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    args = parser.parse_args()

    if args.torch_dtype == "auto":
        args.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    set_seed(args.seed)
    main(args)
