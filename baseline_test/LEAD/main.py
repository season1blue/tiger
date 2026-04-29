"""
PhysUniBench 评测主入口。

使用 LEAD 或 CoT 方法，
在 PhysUniBench 基准上对视觉语言模型进行推理评测。

用法示例：
    python main.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --method lead
    python main.py --limit 10 --max_new_tokens 512   # 调试模式
    python main.py --eval_only --results output/results.jsonl  # 仅评估
"""

import argparse
import os

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)

from lead import (
    run_single_inference,
    load_dataset,
    get_dataset_statistics,
    evaluate_dataset,
    print_evaluation_report,
    save_evaluation_report,
    format_prompt_from_sample,
    setup_logger,
    save_jsonl,
    save_json,
    Timer,
)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="PhysUniBench Evaluation with LEAD / CoT",
    )

    # ---- 模型与数据 ----
    parser.add_argument(
        "--model_name",
        type=str,
        default="/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace 模型名或本地权重路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集 JSONL 文件路径（默认 data/physunibench.jsonl）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录（默认 output/）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅运行前 N 条样本（用于调试），不指定则运行全部",
    )

    # ---- 数据过滤 ----
    parser.add_argument(
        "--subtopics",
        type=str,
        nargs="*",
        default=None,
        help="按子领域过滤，如 --subtopics Mechanics Optics",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["english", "chinese"],
        help="按语言过滤",
    )
    parser.add_argument(
        "--min_difficulty",
        type=int,
        default=1,
        help="最低难度（默认 1）",
    )
    parser.add_argument(
        "--max_difficulty",
        type=int,
        default=5,
        help="最高难度（默认 5）",
    )

    # ---- 推理方法 ----
    parser.add_argument(
        "--method",
        type=str,
        default="lead",
        choices=["lead", "cot", "cot_greedy"],
        help="推理方法：lead / cot / cot_greedy",
    )
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="LEAD alpha_0 参数")
    parser.add_argument("--max_switch_count", type=int, default=5,
                        help="LEAD 最大切换次数")

    # ---- 采样参数 ----
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=25600)
    parser.add_argument(
        "--do_sample",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="输入张量所使用的设备，默认自动选择 CUDA 或 CPU",
    )

    # ---- 评估模式 ----
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅评估已有结果，不运行推理",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="已有结果文件路径（配合 --eval_only 使用）",
    )

    # ---- 兼容 inference.py 中使用的字段（由循环动态赋值） ----
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")

    return parser.parse_args()


def run_eval_only(args, logger):
    """
    仅评估模式：加载已有结果并输出评估报告。

    Args:
        args: 命令行参数。
        logger: logger 实例。
    """
    from lead import load_jsonl

    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(project_root, "output")
    results_path = args.results or os.path.join(output_dir, "results.jsonl")

    logger.info(f"Loading results from: {results_path}")
    dataset = load_jsonl(results_path)
    logger.info(f"Loaded {len(dataset)} samples")

    eval_results = evaluate_dataset(dataset)
    print_evaluation_report(eval_results)

    report_path = os.path.join(output_dir, "eval_report.json")
    save_evaluation_report(eval_results, report_path)
    logger.info(f"Evaluation report saved to: {report_path}")


def main():
    """主流程：加载模型 → 遍历数据集 → 逐样本推理 → 评估 → 保存结果。"""
    args = parse_args()

    # ---- 路径解析 ----
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    output_dir = args.output_dir or os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")

    logger = setup_logger("lead", log_dir=log_dir)

    # ---- 仅评估模式 ----
    if args.eval_only:
        run_eval_only(args, logger)
        return

    dataset_path = args.dataset or os.path.join(data_dir, "physunibench.jsonl")
    output_path = os.path.join(output_dir, "results.jsonl")
    report_path = os.path.join(output_dir, "eval_report.json")

    # ---- 加载模型（仅加载一次） ----
    logger.info(f"Loading model: {args.model_name}")
    with Timer("model_loading") as t:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Model loaded in {t}")

    # ---- 加载数据集 ----
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(
        dataset_path,
        data_dir,
        subtopics=args.subtopics,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        language=args.language,
    )
    if args.limit is not None:
        dataset = dataset[: args.limit]
        logger.info(f"Limited to first {len(dataset)} samples (--limit={args.limit})")

    stats = get_dataset_statistics(dataset)
    logger.info(f"Dataset: {stats['total']} samples, "
                f"subtopics={list(stats['subtopics'].keys())}, "
                f"languages={list(stats['languages'].keys())}")

    # ---- 保存运行配置 ----
    config = {
        "model_name": args.model_name,
        "method": args.method,
        "alpha": args.alpha,
        "max_switch_count": args.max_switch_count,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dataset": dataset_path,
        "num_samples": len(dataset),
    }
    save_json(config, os.path.join(output_dir, "config.json"))

    # ---- 逐样本推理 ----
    logger.info(f"Starting inference with method={args.method}")
    total_timer = Timer("total_inference").start()

    for idx, sample in enumerate(dataset):
        prompt = format_prompt_from_sample(sample)
        image_url = sample.get("image", "")

        logger.info(f"[{idx + 1}/{len(dataset)}] id={sample.get('id', '?')} "
                     f"image={os.path.basename(image_url)}")

        args.image = image_url
        args.prompt = prompt

        with Timer("sample") as sample_timer:
            try:
                sample["model_answer"] = run_single_inference(
                    model, processor, tokenizer, args,
                )
            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")
                sample["model_answer"] = None
                torch.cuda.empty_cache()
                continue

        logger.info(f"  Completed in {sample_timer}")

    total_timer.stop()
    logger.info(f"Inference completed in {total_timer}")

    # ---- 保存结果 ----
    save_jsonl(dataset, output_path)
    logger.info(f"Results saved to: {output_path}")

    # ---- 评估 ----
    logger.info("Running evaluation...")
    eval_results = evaluate_dataset(dataset)
    eval_results["config"] = config
    print_evaluation_report(eval_results)
    save_evaluation_report(eval_results, report_path)
    logger.info(f"Evaluation report saved to: {report_path}")

    logger.info(f"Done. Total samples: {len(dataset)}, "
                f"Accuracy: {eval_results['accuracy']:.2%}")


if __name__ == "__main__":
    main()
