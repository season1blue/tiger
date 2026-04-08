import argparse
import json
import math
import os
import warnings
import sys

WORKSPACE_ROOT = "/data/ssz/memvr_simple"
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import shortuuid
import torch
from tqdm import tqdm
from transformers550 import AutoProcessor
from transformers550.generation import GenerationConfig
from transformers550.utils import logging as hf_logging
from transformers550.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from memvr import apply_memvr_qwen25

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def get_model_device(model):
    return next(model.parameters()).device


def load_qwen_model(args):
    if args.vision_retracing not in {"default", "append", "add", "adapt"}:
        raise ValueError("Invalid vision retracing mode. Choose from default, append, add, adapt.")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    checkpoint = model_path if os.path.exists(model_path) else args.model_path

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map=args.cuda_device,
        trust_remote_code=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
    model.generation_config.top_p = 0.01

    if args.apply_memvr == "memvr":
        apply_memvr_qwen25(
            self=model,
            starting_layer=args.starting_layer,
            ending_layer=args.ending_layer,
            entropy_threshold=args.entropy_threshold,
            retracing_ratio=args.retracing_ratio,
        )
    else:
        model.model.language_model.layers[0].mlp.apply_memvr = False

    return model_name, processor, model


def run_qa_eval(args, model_name, processor, model):
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    device = get_model_device(model)
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions, total=len(questions), desc="Qwen eval", ncols=100):
            qid = line["question_id"]
            image_name = line["image"]
            prompt = line["text"]
            image_path = os.path.join(args.image_folder, image_name)

            with torch.inference_mode():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            ans_file.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "prompt": prompt,
                        "text": output,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )


def run_chair_eval(args, processor, model):
    with open(args.chair_image_list, "r") as f:
        img_files = f.read().splitlines()

    with open(args.question_file, "r") as f:
        coco_anns = json.load(f)

    category_dict = {int(c["id"]): c["name"] for c in coco_anns["categories"]}
    img_dict = {img["id"]: {"name": img["file_name"], "anns": []} for img in coco_anns["images"]}
    for ann in coco_anns["annotations"]:
        img_dict[ann["image_id"]]["anns"].append(category_dict[ann["category_id"]])

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    max_samples = min(len(img_files), args.chair_max_samples)

    device = get_model_device(model)
    with open(args.answers_file, "w") as out_f:
        for img_idx in tqdm(range(max_samples), total=max_samples, desc="CHAIR eval", ncols=100):
            img_file = img_files[img_idx]
            img_id = int(img_file.split(".jpg")[0][-6:])
            img_info = img_dict[img_id]
            assert img_info["name"] == img_file

            image_path = os.path.join(args.image_folder, img_file)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": image_path},
                        {"type": "text", "text": "Please describe this image in detail."},
                    ],
                }
            ]

            with torch.inference_mode():
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            out_f.write(json.dumps({"image_id": img_id, "caption": output}) + "\n")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", type=str, default="qa", choices=["qa", "chair"])

    parser.add_argument("--model-path", type=str, default="/data/ssz/llms/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--vision-retracing", type=str, default="default")
    parser.add_argument("--retracing-ratio", type=float, default=0.0)
    parser.add_argument("--entropy-threshold", type=float, default=0.75)
    parser.add_argument("--starting-layer", type=int, default=5)
    parser.add_argument("--ending-layer", type=int, default=16)
    parser.add_argument("--apply-memvr", type=str, default="default")

    parser.add_argument("--chair-image-list", type=str, default="/data/ssz/Datasets/chair/shuffled_img_files.txt")
    parser.add_argument("--chair-max-samples", type=int, default=500)
    return parser


def main():
    args = build_parser().parse_args()
    model_name, processor, model = load_qwen_model(args)

    if args.task_type == "chair":
        run_chair_eval(args, processor, model)
    else:
        run_qa_eval(args, model_name, processor, model)


if __name__ == "__main__":
    main()
