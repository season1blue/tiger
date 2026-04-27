#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_PROMPTS: Dict[str, str] = {
	"conv": (
		"You are an impartial evaluator for visual question answering. Compare two answers to the same question "
		"using the provided context. Score each answer from 1 to 10 by correctness, relevance, and helpfulness. "
		"First line MUST be exactly: '<score1> <score2>' (two numbers). Then give a short explanation."
	),
	"detail": (
		"You are an impartial evaluator for image understanding. Compare two answers using the provided context. "
		"Reward factual detail, coverage, and faithfulness. Score each answer from 1 to 10. "
		"First line MUST be exactly: '<score1> <score2>' (two numbers). Then give a short explanation."
	),
	"complex": (
		"You are an impartial evaluator for complex visual reasoning. Compare two answers using the provided context. "
		"Reward reasoning quality, correctness, and grounding. Score each answer from 1 to 10. "
		"First line MUST be exactly: '<score1> <score2>' (two numbers). Then give a short explanation."
	),
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate LLaVA-Bench answers with GPT review and summarize scores.")
	parser.add_argument("--question", required=True, help="Path to questions.jsonl")
	parser.add_argument("--context", required=True, help="Path to context.jsonl")
	parser.add_argument("--reference-answer", required=True, help="Path to reference answers (e.g. answers_gpt4.jsonl)")
	parser.add_argument("--model-answer", required=True, help="Path to model answers jsonl")
	parser.add_argument("--review-output", required=True, help="Output path for review jsonl")
	parser.add_argument("--summary-output", default=None, help="Optional output path for summary json")
	parser.add_argument("--reviewer-model", default="gpt-4o-mini", help="Reviewer model name")
	parser.add_argument("--api-key", default=None, help="OpenAI API key (default: OPENAI_API_KEY env)")
	parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL")
	parser.add_argument(
		"--api-mode",
		default="chat",
		choices=["chat", "responses"],
		help="API style: chat=completions endpoint, responses=responses endpoint",
	)
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--max-tokens", type=int, default=512)
	parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Retry sleep when request fails")
	parser.add_argument("--max-retries", type=int, default=8)
	parser.add_argument("--only-summary", action="store_true", help="Skip GPT review and summarize existing review file")
	parser.add_argument(
		"--overwrite-review",
		action="store_true",
		help="Overwrite review output and rerun GPT review from the first sample",
	)
	return parser.parse_args()


def load_jsonl(path: str) -> List[dict]:
	with open(path, "r", encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def parse_score_pair(review_text: str) -> Tuple[float, float]:
	first_line = review_text.strip().splitlines()[0] if review_text.strip() else ""
	nums = re.findall(r"[-+]?\d*\.?\d+", first_line)
	if len(nums) >= 2:
		return float(nums[0]), float(nums[1])
	return -1.0, -1.0


def build_content(question: dict, context_row: dict, ref_ans: dict, model_ans: dict) -> str:
	caption = context_row.get("caption", "")
	if isinstance(caption, list):
		caption = "\n".join(caption)

	category = question.get("category", "conv")
	prompt = DEFAULT_PROMPTS.get(category, DEFAULT_PROMPTS["conv"])

	return (
		f"[Context]\n{caption}\n\n"
		f"[Question]\n{question.get('text', '')}\n\n"
		f"[Assistant 1]\n{ref_ans.get('text', '')}\n\n[End of Assistant 1]\n\n"
		f"[Assistant 2]\n{model_ans.get('text', '')}\n\n[End of Assistant 2]\n\n"
		f"[System]\n{prompt}\n"
	)


def call_reviewer(client, model: str, content: str, max_tokens: int, temperature: float, max_retries: int, sleep_seconds: float) -> str:
	last_error = None
	for _ in range(max_retries):
		try:
			resp = client.chat.completions.create(
				model=model,
				messages=[
					{
						"role": "system",
						"content": "You are a helpful and precise assistant for checking answer quality.",
					},
					{"role": "user", "content": content},
				],
				temperature=temperature,
				max_tokens=max_tokens,
			)
			return resp.choices[0].message.content or ""
		except Exception as e:
			last_error = e
			time.sleep(sleep_seconds)
	raise RuntimeError(f"Reviewer request failed after retries: {last_error}")


def call_reviewer_responses(client, model: str, content: str, max_tokens: int, temperature: float, max_retries: int, sleep_seconds: float) -> str:
	last_error = None
	for _ in range(max_retries):
		try:
			resp = client.responses.create(
				model=model,
				input=[
					{
						"role": "system",
						"content": [{"type": "input_text", "text": "You are a helpful and precise assistant for checking answer quality."}],
					},
					{"role": "user", "content": [{"type": "input_text", "text": content}]},
				],
				temperature=temperature,
				max_output_tokens=max_tokens,
			)
			text = getattr(resp, "output_text", None)
			if text:
				return text
			# Fallback for gateways that return content blocks without output_text.
			out = []
			for item in getattr(resp, "output", []) or []:
				for c in getattr(item, "content", []) or []:
					if getattr(c, "type", "") in {"output_text", "text"} and getattr(c, "text", None):
						out.append(c.text)
			return "\n".join(out)
		except Exception as e:
			last_error = e
			time.sleep(sleep_seconds)
	raise RuntimeError(f"Reviewer request failed after retries: {last_error}")


def normalize_base_url(base_url: str) -> str:
	url = base_url.rstrip("/")
	if url.endswith("/v1") or "/v1/" in url:
		return url
	return f"{url}/v1"


def run_review(args: argparse.Namespace) -> None:
	from openai import OpenAI

	api_key = args.api_key or os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

	client_kwargs = {"api_key": api_key}
	base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
	if base_url:
		client_kwargs["base_url"] = normalize_base_url(base_url)
	client = OpenAI(**client_kwargs)

	questions = load_jsonl(args.question)
	references = load_jsonl(args.reference_answer)
	predictions = load_jsonl(args.model_answer)
	contexts = load_jsonl(args.context)

	if not (len(questions) == len(references) == len(predictions)):
		raise ValueError(
			f"Length mismatch: questions={len(questions)}, reference={len(references)}, model={len(predictions)}"
		)

	image_to_context = {row.get("image"): row for row in contexts}

	review_path = Path(args.review_output)
	review_path.parent.mkdir(parents=True, exist_ok=True)

	existing = []
	if args.overwrite_review and review_path.exists():
		print(f"[review] overwrite enabled, remove existing file: {review_path}")
		review_path.unlink()

	if review_path.exists():
		existing = load_jsonl(str(review_path))
	start_idx = len(existing)
	total = len(questions)

	if start_idx > 0:
		remaining = max(total - start_idx, 0)
		print(f"[review] resume from {start_idx + 1}/{total} (existing={start_idx}, remaining={remaining})")
	if start_idx >= total:
		print("[review] no pending samples. Use --overwrite-review to rerun all samples.")
		return

	mode = "a" if review_path.exists() else "w"
	with open(review_path, mode, encoding="utf-8") as wf:
		for idx, (q, a_ref, a_model) in enumerate(zip(questions, references, predictions)):
			if idx < start_idx:
				continue

			image = q.get("image")
			if image not in image_to_context:
				raise KeyError(f"Image '{image}' from question not found in context file")

			content = build_content(q, image_to_context[image], a_ref, a_model)
			if args.api_mode == "responses":
				review_text = call_reviewer_responses(
					client=client,
					model=args.reviewer_model,
					content=content,
					max_tokens=args.max_tokens,
					temperature=args.temperature,
					max_retries=args.max_retries,
					sleep_seconds=args.sleep_seconds,
				)
			else:
				review_text = call_reviewer(
					client=client,
					model=args.reviewer_model,
					content=content,
					max_tokens=args.max_tokens,
					temperature=args.temperature,
					max_retries=args.max_retries,
					sleep_seconds=args.sleep_seconds,
				)
			s1, s2 = parse_score_pair(review_text)

			row = {
				"id": idx + 1,
				"question_id": q.get("question_id", idx),
				"image": image,
				"category": f"llava_bench_{q.get('category', 'conv')}",
				"answer1_id": a_ref.get("answer_id", a_ref.get("question_id", idx)),
				"answer2_id": a_model.get("answer_id", a_model.get("question_id", idx)),
				"tuple": [s1, s2],
				"content": review_text,
			}
			wf.write(json.dumps(row, ensure_ascii=False) + "\n")
			wf.flush()
			print(f"[review] {idx + 1}/{total} -> {s1:.2f}, {s2:.2f}")


def summarize_review(review_rows: List[dict]) -> Dict[str, Dict[str, float]]:
	grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

	for r in review_rows:
		cat = r.get("category", "llava_bench_conv")
		pair = r.get("tuple", [-1, -1])
		if not isinstance(pair, list) or len(pair) < 2:
			continue
		s1, s2 = float(pair[0]), float(pair[1])
		if s1 <= 0 or s2 < 0:
			continue
		grouped[cat].append((s1, s2))
		grouped["llava_bench_all"].append((s1, s2))

	def aggregate(values: List[Tuple[float, float]]) -> Dict[str, float]:
		if not values:
			return {"ratio": -1.0, "ref_score_x10": -1.0, "model_score_x10": -1.0, "count": 0}
		ref_avg = sum(v[0] for v in values) / len(values)
		model_avg = sum(v[1] for v in values) / len(values)
		ratio = (model_avg / ref_avg * 100.0) if ref_avg > 0 else -1.0
		return {
			"ratio": round(ratio, 1),
			"ref_score_x10": round(ref_avg * 10.0, 1),
			"model_score_x10": round(model_avg * 10.0, 1),
			"count": len(values),
		}

	summary: Dict[str, Dict[str, float]] = {}
	for cat in ["llava_bench_conv", "llava_bench_detail", "llava_bench_complex", "llava_bench_all"]:
		summary[cat] = aggregate(grouped.get(cat, []))

	valid_ratios = [
		summary[k]["ratio"]
		for k in ["llava_bench_conv", "llava_bench_detail", "llava_bench_complex"]
		if summary[k]["ratio"] >= 0
	]
	summary["average_ratio"] = round(sum(valid_ratios) / len(valid_ratios), 1) if valid_ratios else -1.0
	return summary


def print_table(summary: Dict[str, Dict[str, float]]) -> None:
	print("\n===== LLaVA-Bench Summary =====")
	print("Category        Ratio(%)   Ref(x10)  Model(x10)  Count")
	name_map = {
		"llava_bench_conv": "Convs",
		"llava_bench_detail": "Detail",
		"llava_bench_complex": "Complex",
		"llava_bench_all": "All",
	}
	for key in ["llava_bench_conv", "llava_bench_detail", "llava_bench_complex", "llava_bench_all"]:
		row = summary[key]
		print(
			f"{name_map[key]:<13} {row['ratio']:>8} {row['ref_score_x10']:>10} {row['model_score_x10']:>11} {row['count']:>6}"
		)
	print(f"Average Ratio: {summary['average_ratio']}")
	print("===============================\n")


def run_summary(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
	rows = load_jsonl(args.review_output)
	summary = summarize_review(rows)
	print_table(summary)
	if args.summary_output:
		out = Path(args.summary_output)
		out.parent.mkdir(parents=True, exist_ok=True)
		with open(out, "w", encoding="utf-8") as f:
			json.dump(summary, f, ensure_ascii=False, indent=2)
		print(f"[saved] summary -> {out}")
	return summary


def main() -> None:
	args = parse_args()
	if not args.only_summary:
		run_review(args)
	run_summary(args)


if __name__ == "__main__":
	main()
