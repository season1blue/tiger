import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]

for _p in (str(EXPERIMENTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, set_seed
from transformers.generation.utils import GenerationMixin

from icd_utils.icd_sample import evolve_icd_sampling


def _patch_qwen25_generation_hooks():
    if not hasattr(Qwen2_5_VLForConditionalGeneration, "prepare_inputs_for_generation_cd"):
        original_prepare = Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation

        def prepare_inputs_for_generation_cd(self, input_ids, **kwargs):
            kwargs = dict(kwargs)
            input_ids_cd = kwargs.pop("input_ids_cd", None)
            if input_ids_cd is not None:
                input_ids = input_ids_cd
            return original_prepare(self, input_ids, **kwargs)

        Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation_cd = prepare_inputs_for_generation_cd

    if getattr(GenerationMixin._validate_model_kwargs, "_icd_qwen25_patched", False):
        return

    original_validate = GenerationMixin._validate_model_kwargs

    def _validate_model_kwargs_with_icd(self, model_kwargs):
        model_kwargs = dict(model_kwargs)
        for key in ("input_ids_cd", "attention_mask_cd", "preprompt_cd", "cd_alpha", "cd_beta", "use_cd"):
            model_kwargs.pop(key, None)
        return original_validate(self, model_kwargs)

    _validate_model_kwargs_with_icd._icd_qwen25_patched = True
    GenerationMixin._validate_model_kwargs = _validate_model_kwargs_with_icd


_patch_qwen25_generation_hooks()


def normalize_yes_no(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()

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


def infer_once(model, processor, image_path, question, cd_question, args):
    prompt = f"{question}{args.prompt_suffix}"
    clean_inputs = build_inputs(processor, model, image_path, prompt)
    used_prompt = prompt

    if cd_question is not None:
        cd_prompt = f"{cd_question} {question}{args.prompt_suffix}".strip()
        cd_inputs = build_inputs(processor, model, image_path, cd_prompt)
        clean_inputs["input_ids_cd"] = cd_inputs["input_ids"]
        if "attention_mask" in cd_inputs:
            clean_inputs["attention_mask_cd"] = cd_inputs["attention_mask"]
        used_prompt = cd_prompt

    with torch.inference_mode():
        output_ids = model.generate(
            **clean_inputs,
            use_cd=cd_question is not None and args.use_cd,
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    input_token_len = clean_inputs["input_ids"].shape[1]
    generated = output_ids[:, input_token_len:]
    raw_text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    answer = normalize_yes_no(raw_text)
    return used_prompt, raw_text, answer


def run_pope(args):
    model_path = os.path.expanduser(args.model_path)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    ).eval()

    if args.format == "yn_format":
        args.prompt_suffix = " Answer with yes or no only."
    elif args.format == "ow_format":
        args.prompt_suffix = " Answer with one word only."
    else:
        args.prompt_suffix = ""

    preprompts = [
        "You are an object detector to recognize every different objects.",
        "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
        "I want you avoid any specific identification or categorization of the objects depicted.",
        "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
        "You are an object detector to provide a general overview or impression of the image.",
    ]

    data_files = {
        "coco_adversarial": f"{args.question_folder}/coco_pope_adversarial.json",
        "coco_popular": f"{args.question_folder}/coco_pope_popular.json",
        "coco_random": f"{args.question_folder}/coco_pope_random.json",
        "aokvqa_adversarial": f"{args.question_folder}/aokvqa_pope_seem_adversarial.json",
        "aokvqa_popular": f"{args.question_folder}/aokvqa_pope_seem_popular.json",
        "aokvqa_random": f"{args.question_folder}/aokvqa_pope_seem_random.json",
        "gqa_random": f"{args.question_folder}/gqa_pope_seem_random.json",
        "gqa_popular": f"{args.question_folder}/gqa_pope_seem_popular.json",
        "gqa_adversarial": f"{args.question_folder}/gqa_pope_seem_adversarial.json",
    }

    sub_folder = "icd" if args.use_cd else "baseline"
    save_folder = {
        f"{args.save_folder}/{sub_folder}/gqa_random/{args.format}": data_files["gqa_random"],
        f"{args.save_folder}/{sub_folder}/gqa_adversarial/{args.format}": data_files["gqa_adversarial"],
        f"{args.save_folder}/{sub_folder}/gqa_popular/{args.format}": data_files["gqa_popular"],
        f"{args.save_folder}/{sub_folder}/coco_adversarial/{args.format}": data_files["coco_adversarial"],
        f"{args.save_folder}/{sub_folder}/coco_popular/{args.format}": data_files["coco_popular"],
        f"{args.save_folder}/{sub_folder}/coco_random/{args.format}": data_files["coco_random"],
        f"{args.save_folder}/{sub_folder}/aokvqa_adversarial/{args.format}": data_files["aokvqa_adversarial"],
        f"{args.save_folder}/{sub_folder}/aokvqa_popular/{args.format}": data_files["aokvqa_popular"],
        f"{args.save_folder}/{sub_folder}/aokvqa_random/{args.format}": data_files["aokvqa_random"],
    }

    for save_path, value in save_folder.items():
        data = json.load(open(value, "r"))
        if "gqa" in save_path:
            image_root = args.gvqa_image_root
        else:
            image_root = args.coco_image_root

        print(save_path, image_root)
        for preprompt in preprompts:
            prompt_folder = os.path.join(save_path, str(preprompt))
            Path(prompt_folder).mkdir(parents=True, exist_ok=True)
            print("processing preprompt:", preprompt)

            result_normal = []
            result_question = []
            for item in tqdm(data):
                image_file = item["image"]
                image_path = os.path.join(image_root, image_file)
                question = item["text"]

                normal_prompt, raw_text, answer = infer_once(
                    model,
                    processor,
                    image_path,
                    question,
                    None,
                    args,
                )
                result_normal.append(
                    {
                        "question_id": item["question_id"],
                        "image": image_file,
                        "question": question,
                        "prompt": normal_prompt,
                        "answer": answer,
                        "raw_text": raw_text,
                    }
                )

                if args.use_cd:
                    question_prompt, raw_text, answer = infer_once(
                        model,
                        processor,
                        image_path,
                        question,
                        preprompt,
                        args,
                    )
                    result_question.append(
                        {
                            "question_id": item["question_id"],
                            "image": image_file,
                            "question": question,
                            "prompt": question_prompt,
                            "answer": answer,
                            "raw_text": raw_text,
                        }
                    )

            json.dump(result_normal, open(os.path.join(prompt_folder, "normal.json"), "w"))
            if len(result_question) > 0:
                json.dump(result_question, open(os.path.join(prompt_folder, "question.json"), "w"))


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--use_cd", action="store_true", default=False, help="use contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gvqa_image_root", type=str, default=None)
    parser.add_argument("--coco_image_root", type=str, default=None)
    parser.add_argument("--question_folder", type=str, default="./experiments/data/pope")
    parser.add_argument("--save_folder", type=str, default="./pope_results/qwen2_5_vl")
    parser.add_argument("--format", type=str, default="yn_format", choices=["no_format", "ow_format", "yn_format"])
    return parser


def main():
    args = get_args_parser().parse_args()
    if args.torch_dtype == "auto":
        args.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None

    set_seed(args.seed)
    evolve_icd_sampling()
    run_pope(args)


if __name__ == "__main__":
    main()