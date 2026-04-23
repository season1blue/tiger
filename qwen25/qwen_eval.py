import argparse
import json
import math
import os
from pathlib import Path
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
from qwen25.analysis_logger import normalize_binary_answer, trace_qwen25_sample
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


def resolve_method(args):
    return args.method or "base"


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

    method = resolve_method(args)
    # Keep a single source of truth for runtime mode: base / memvr / evo.
    for decoder_layer in model.model.language_model.layers:
        decoder_layer.mlp.memvr_method = method

    if method in {"memvr", "evo"}:
        apply_memvr_qwen25(
            self=model,
            starting_layer=args.starting_layer,
            ending_layer=args.ending_layer,
            entropy_threshold=args.entropy_threshold,
            retracing_ratio=args.retracing_ratio,
            retrace_delay_layers=args.retrace_delay_layers,
            retrace_target_layers=args.retrace_target_layers,
            method=method,
            state_drift_threshold=args.state_drift_threshold,
            state_drift_pooling=args.state_drift_pooling,
        )
    else:
        for decoder_layer in model.model.language_model.layers:
            decoder_layer.mlp.apply_memvr = False

    return model_name, processor, model


def run_qa_eval(args, model_name, processor, model):
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    device = get_model_device(model)
    total_samples = len(questions)
    triggered_samples = 0
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions, total=len(questions), desc="Qwen eval", ncols=100):
            qid = line["question_id"]
            image_name = line["image"]
            prompt = line["text"]
            ground_truth_answer = line.get("label", line.get("answer", line.get("gt_answer", None)))
            image_path = os.path.join(args.image_folder, image_name)

            with torch.inference_mode():
                before_trigger_total = getattr(model.model.language_model, "_memvr_trigger_total", 0)
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

                if args.analysis_log_dir:
                    generated_tensor = generated_ids_trimmed[0].unsqueeze(0)
                    generated_answer_norm = normalize_binary_answer(output)
                    ground_truth_norm = normalize_binary_answer(ground_truth_answer)
                    final_prediction_correct = (
                        generated_answer_norm is not None
                        and ground_truth_norm is not None
                        and generated_answer_norm == ground_truth_norm
                    )
                    is_hallucinated = None
                    if generated_answer_norm in {"yes", "no"} and ground_truth_norm in {"yes", "no"}:
                        # Treat both error types as hallucination for binary POPE-style QA:
                        # 1) say "yes" when GT is "no" (false positive)
                        # 2) say "no" when GT is "yes" (false negative)
                        is_hallucinated = int(generated_answer_norm != ground_truth_norm)

                    analysis_metadata = {
                        "sample_id": str(qid),
                        "question_id": qid,
                        "dataset_name": args.dataset_name or args.task_type,
                        "model_name": model_name,
                        "prompt": prompt,
                        "generated_answer": output,
                        "ground_truth_answer": ground_truth_answer,
                        "final_prediction_correct": final_prediction_correct,
                        "is_hallucinated": is_hallucinated,
                    }
                    trace_qwen25_sample(
                        model=model,
                        processor=processor,
                        inputs=inputs,
                        generated_ids_trimmed=generated_tensor,
                        metadata=analysis_metadata,
                        analysis_log_dir=args.analysis_log_dir,
                    )
                after_trigger_total = getattr(model.model.language_model, "_memvr_trigger_total", 0)
                if after_trigger_total > before_trigger_total:
                    triggered_samples += 1

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

    if total_samples > 0:
        triggered_ratio = triggered_samples / total_samples
    else:
        triggered_ratio = 0.0
    stats = {
        "total_samples": total_samples,
        "triggered_samples": triggered_samples,
        "triggered_ratio": triggered_ratio,
    }
    stats_file = f"{answers_file}.{resolve_method(args)}_stats.json"
    try:
        stats_dir = os.path.dirname(stats_file)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[MemVR Stats] warning: failed to write stats_file={stats_file}: {exc}")
        stats_file = "<write_failed>"

    print(
        f"[MemVR Stats] total_samples={total_samples} "
        f"triggered_samples={triggered_samples} "
        f"triggered_ratio={triggered_ratio:.4f} "
        f"stats_file={stats_file}"
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

            if args.chair_print_output and args.chair_debug_every > 0 and ((img_idx + 1) % args.chair_debug_every == 0):
                preview = output.replace("\n", " ").strip()
                if len(preview) > args.chair_output_max_chars:
                    preview = preview[: args.chair_output_max_chars] + "..."
                print(f"[CHAIR OUTPUT] image={img_file} caption={preview}")

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

    parser.add_argument("--method", type=str, default=None, choices=["base", "memvr", "evo"])

    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--vision-retracing", type=str, default="default")
    parser.add_argument("--retracing-ratio", type=float, default=0.0)
    parser.add_argument("--retrace-delay-layers", type=int, default=1)
    parser.add_argument("--retrace-target-layers", type=str, default="")
    parser.add_argument("--state-drift-threshold", type=float, default=0.5)
    parser.add_argument("--state-drift-pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--entropy-threshold", type=float, default=0.75)
    parser.add_argument("--starting-layer", type=int, default=5)
    parser.add_argument("--ending-layer", type=int, default=16)
    parser.add_argument("--apply-memvr", type=str, default="default")

    parser.add_argument("--chair-image-list", type=str, default="/data/ssz/Datasets/chair/shuffled_img_files.txt")
    parser.add_argument("--chair-max-samples", type=int, default=500)
    parser.add_argument("--chair-print-output", action="store_true")
    parser.add_argument("--chair-debug-every", type=int, default=1)
    parser.add_argument("--chair-output-max-chars", type=int, default=240)
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--analysis-log-dir", type=str, default="")
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
