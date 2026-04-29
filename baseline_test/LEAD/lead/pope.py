"""
POPE inference entrypoint for LEAD.

This module runs LEAD or CoT decoding on a prepared POPE question JSONL
file and writes answers in the format expected by the POPE evaluator.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from .data import load_jsonl
from .inference import run_single_inference
from .utils import save_jsonl


DEFAULT_MODEL_PATH = "/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEAD POPE inference")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="lead",
        choices=["lead", "cot", "cot_greedy"],
    )
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--max-switch-count", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--do-sample",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save intermediate answers every N samples (0 to disable).",
    )
    return parser.parse_args()


def resolve_image_path(image_value: str, image_folder: str) -> str:
    if not image_value:
        return image_value
    if os.path.isabs(image_value):
        return image_value
    return os.path.join(image_folder, image_value)


def build_prompt(question: str) -> str:
    return f"{question}\nAnswer with Yes or No only."


def main() -> None:
    args = parse_args()

    question_path = Path(args.question_file)
    if not question_path.is_file():
        raise FileNotFoundError(f"question file not found: {question_path}")

    image_folder = Path(args.image_folder)
    if not image_folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    questions: List[Dict] = load_jsonl(str(question_path))
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    compute_device = args.device
    if compute_device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = torch.device(compute_device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    answers: List[Dict] = []
    total = len(questions)
    print(f"[POPE] loaded {total} samples", flush=True)
    start_time = time.time()

    for idx, sample in enumerate(questions, start=1):
        image_value = sample.get("image", "")
        image_path = resolve_image_path(image_value, str(image_folder))
        question_text = sample.get("question") or sample.get("text") or ""

        sample_args = argparse.Namespace(**vars(args))
        sample_args.image = image_path
        sample_args.prompt = build_prompt(question_text)

        answer_text = run_single_inference(
            model,
            processor,
            tokenizer,
            sample_args,
            verbose=False,
        )

        answers.append(
            {
                "question_id": sample.get("question_id", sample.get("id")),
                "text": answer_text,
            }
        )

        if args.save_every > 0 and idx % args.save_every == 0:
            save_jsonl(answers, args.answers_file)
            elapsed = time.time() - start_time
            avg = elapsed / idx
            eta = avg * (total - idx)
            print(
                f"[POPE] progress {idx}/{total} | avg={avg:.2f}s/sample | eta={eta/60:.1f}m",
                flush=True,
            )

    save_jsonl(answers, args.answers_file)
    elapsed = time.time() - start_time
    avg = elapsed / max(total, 1)
    print(
        f"[POPE] finished {total}/{total} | total={elapsed/60:.1f}m | avg={avg:.2f}s/sample",
        flush=True,
    )


if __name__ == "__main__":
    main()