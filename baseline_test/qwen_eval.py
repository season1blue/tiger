import argparse
import json
import math
import os
import uuid
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        raise ValueError(f"chunk-idx out of range: {k}, total chunks: {len(chunks)}")
    return chunks[k]


def load_questions(path: str):
    p = Path(path)
    text = p.read_text().strip()
    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_model_and_processor(args):
    checkpoint = os.path.expanduser(args.model_path)
    if not os.path.exists(checkpoint):
        checkpoint = args.model_path

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    if AutoModelForImageTextToText is not None:
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif AutoModelForVision2Seq is not None:
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        raise RuntimeError("No suitable model class found in installed transformers version.")

    model = model.to(args.cuda_device).eval()
    return processor, model


def build_inputs(processor, image_path: str, prompt: str, device: str):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def normalize_pred(text: str) -> int:
    t = text.split(".")[0].replace(",", " ")
    words = t.split()
    if "No" in words or "no" in words or "not" in words:
        return 0
    return 1


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate_pope(question_file: str, answers_file: str):
    questions = load_questions(question_file)
    answers = [json.loads(x) for x in Path(answers_file).read_text().splitlines() if x.strip()]
    answers_by_qid = {str(a["question_id"]): a.get("text", "") for a in answers}

    labels = []
    preds = []
    missing = 0

    for q in questions:
        if "label" not in q:
            continue

        qid = str(q["question_id"])
        gt = str(q["label"]).strip().lower()
        labels.append(1 if gt == "yes" else 0)

        if qid not in answers_by_qid:
            missing += 1
            preds.append(0)
            continue

        preds.append(normalize_pred(answers_by_qid[qid]))

    tp = tn = fp = fn = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    yes_ratio = safe_div(sum(preds), len(preds)) if preds else 0.0

    return {
        "samples": len(labels),
        "missing_answers": missing,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Yes ratio": round(yes_ratio, 4),
    }


def run_qa_eval(args, processor, model):
    questions = load_questions(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions, total=len(questions), desc="Baseline Qwen eval", ncols=100):
            qid = line["question_id"]
            image_name = line["image"]
            prompt = line["text"]
            image_path = os.path.join(args.image_folder, image_name)

            with torch.inference_mode():
                inputs = build_inputs(processor, image_path, prompt, args.cuda_device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=args.temperature,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
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
                        "answer_id": str(uuid.uuid4()),
                        "model_id": args.model_path,
                        "metadata": {"baseline": True},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")

    parser.add_argument("--compute-pope-metrics", action="store_true", default=False)
    parser.add_argument("--metrics-file", type=str, default="")
    return parser


def main():
    args = build_parser().parse_args()
    processor, model = load_model_and_processor(args)
    run_qa_eval(args, processor, model)

    if args.compute_pope_metrics:
        metrics = evaluate_pope(args.question_file, args.answers_file)
        metrics_file = args.metrics_file or f"{args.answers_file}.pope_metrics.json"
        Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_file).write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
        print(f"[POPE Metrics] {json.dumps(metrics, ensure_ascii=False)}")
        print(f"[POPE Metrics] saved to: {metrics_file}")


if __name__ == "__main__":
    main()
