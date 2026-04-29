import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling


def _patch_qwen25_prepare_inputs_for_generation_cd():
    """Inject VCD helper hook expected by vcd_sample.py for Qwen2.5-VL."""
    if hasattr(Qwen2_5_VLForConditionalGeneration, "prepare_inputs_for_generation_cd"):
        return

    original_prepare = Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation

    def prepare_inputs_for_generation_cd(self, input_ids, **kwargs):
        kwargs = dict(kwargs)
        images_cd = kwargs.pop("images_cd", None)
        if images_cd is not None:
            kwargs["pixel_values"] = images_cd
        return original_prepare(self, input_ids, **kwargs)

    Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation_cd = prepare_inputs_for_generation_cd


_patch_qwen25_prepare_inputs_for_generation_cd()

# cd_comment: Step 1 in README. Replace HF sampling with VCD-aware sampling.
evolve_vcd_sampling()


def normalize_yes_no(text: str) -> str:
    cleaned = text.strip()
    low = cleaned.lower()

    match = re.search(r"\b(yes|no)\b", low)
    if match is not None:
        return match.group(1)

    if re.search(r"\b(no|not|n't|none|never)\b", low):
        return "no"
    return "yes"


def eval_model(args):
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
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        prompt = f"{question} Answer with Yes or No only."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
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
        inputs = {
            key: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        images_cd = None
        if args.use_cd:
            # cd_comment: Step 2 in README. Build distorted visual input v'.
            images_cd = add_diffusion_noise(inputs["pixel_values"], args.noise_step)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                # cd_comment: Step 3 in README. Pass contrastive branch and VCD hyperparameters.
                # vcd_utils/vcd_sample.py combines clean/distorted logits during decoding.
                images_cd=images_cd,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
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
                    "metadata": {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--use_cd", action="store_true", default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    args = parser.parse_args()
    if args.torch_dtype == "auto":
        args.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    set_seed(args.seed)
    eval_model(args)