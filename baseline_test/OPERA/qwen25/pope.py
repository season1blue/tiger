from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL_PATH = "/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL POPE inference with OPERA-compatible options")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=50)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--do-sample", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--use-opera", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--strict-opera", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--scale-factor", type=float, default=50.0)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num-attn-candidates", type=int, default=5)
    parser.add_argument("--penalty-weights", type=float, default=1.0)

    return parser.parse_args()


def resolve_image_path(image_value: str, image_folder: str) -> str:
    if not image_value:
        return image_value
    if os.path.isabs(image_value):
        return image_value
    return os.path.join(image_folder, image_value)


def build_prompt(question: str) -> str:
    return f"{question}\nAnswer with Yes or No only."


def normalize_yes_no(text: str) -> str:
    cleaned = text.strip()
    low = cleaned.lower()

    # Direct, high-confidence English matches.
    match = re.search(r"\b(yes|no)\b", low)
    if match is not None:
        return match.group(1)

    # Common Chinese affirm/deny signals.
    if any(tok in cleaned for tok in ("不是", "没有", "并非", "否", "不")):
        return "no"
    if any(tok in cleaned for tok in ("是", "有", "对", "正确")):
        return "yes"

    # Conservative fallback aligned with common POPE normalization: explicit negation -> no, else yes.
    if re.search(r"\b(no|not|n't|none|never)\b", low):
        return "no"
    return "yes"


def get_image_token_id(model, tokenizer) -> int:
    for token_text in ("<|image_pad|>", "<|vision_pad|>"):
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            return token_id
    cfg_token_id = getattr(model.config, "image_token_id", None)
    if isinstance(cfg_token_id, int):
        return cfg_token_id
    return -1


def build_key_position(input_ids: torch.Tensor, image_token_id: int) -> Dict[str, int] | None:
    if image_token_id < 0:
        return None
    row = input_ids[0]
    pos = (row == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if pos.numel() == 0:
        return None
    image_start = int(pos[0].item())
    image_end = int(pos[-1].item())
    response_start = int(row.shape[0])
    return {
        "image_start": image_start,
        "image_end": image_end,
        "response_start": response_start,
    }


def prepare_inputs(processor, image_path: str, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    encoded = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    question_path = Path(args.question_file)
    if not question_path.is_file():
        raise FileNotFoundError(f"question file not found: {question_path}")

    image_folder = Path(args.image_folder)
    if not image_folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    questions = load_jsonl(str(question_path))
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    compute_device = args.device
    if compute_device == "auto":
        if torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if "," in visible and visible.strip():
                # With model sharding, keep model-output gathering away from cuda:0.
                num_visible = len([x for x in visible.split(",") if x.strip()])
                compute_device = f"cuda:{max(0, num_visible - 1)}"
            else:
                compute_device = "cuda"
        else:
            compute_device = "cpu"
    device = torch.device(compute_device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    has_opera_args = "opera_decoding" in inspect.signature(model.generate).parameters
    if args.use_opera and not has_opera_args:
        raise RuntimeError(
            "[POPE] current transformers does not expose opera_decoding. "
            "Please load local transformers550 with OPERA patch via PYTHONPATH."
        )

    image_token_id = get_image_token_id(model, tokenizer)

    answers: List[Dict[str, Any]] = []
    total = len(questions)
    print(f"[POPE] loaded {total} samples", flush=True)
    start_time = time.time()

    for idx, sample in enumerate(questions, start=1):
        image_value = sample.get("image", "")
        image_path = resolve_image_path(image_value, str(image_folder))
        prompt = build_prompt(sample.get("question") or sample.get("text") or "")

        model_inputs = prepare_inputs(processor, image_path, prompt, device)
        prompt_len = model_inputs["input_ids"].shape[1]
        key_position = build_key_position(model_inputs["input_ids"], image_token_id)

        with torch.no_grad():
            if args.use_opera:
                if key_position is None:
                    msg = "[POPE] OPERA requires valid key_position (image token span)."
                    if args.strict_opera:
                        raise RuntimeError(msg)
                    print(f"{msg} fallback to normal decode.", flush=True)
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample,
                        num_beams=args.beam,
                    )
                elif has_opera_args:
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample,
                        num_beams=args.beam,
                        opera_decoding=True,
                        key_position=key_position,
                        scale_factor=args.scale_factor,
                        threshold=args.threshold,
                        num_attn_candidates=args.num_attn_candidates,
                        penalty_weights=args.penalty_weights,
                    )
            else:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    num_beams=args.beam,
                )

        raw_text = tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        text = normalize_yes_no(raw_text)

        answers.append(
            {
                "question_id": sample.get("question_id", sample.get("id", idx - 1)),
                "text": text,
                "raw_text": raw_text,
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
    print(f"[POPE] finished {total}/{total} | total={elapsed/60:.1f}m | avg={avg:.2f}s/sample", flush=True)


if __name__ == "__main__":
    main()
