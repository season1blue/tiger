"""
答案评估与准确率统计。

支持选择题（MCQ）精确匹配和基于规则的答案提取。
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def extract_mcq_answer(text: str) -> Optional[str]:
    """
    从模型输出文本中提取选择题答案字母。

    按优先级尝试多种模式：
    1. "The answer is (X)" 类明确声明
    2. "\\boxed{X}" LaTeX 格式
    3. 文本末尾的独立大写字母

    Args:
        text: 模型生成的文本。

    Returns:
        str or None: 提取到的答案字母（A/B/C/D），提取失败返回 None。
    """
    if not text:
        return None

    patterns = [
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"\\boxed\{([A-Da-d])\}",
        r"\*\*([A-Da-d])\*\*",
        r"(?:^|\n)\s*([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    last_letters = re.findall(r"\b([A-D])\b", text[-200:])
    if last_letters:
        return last_letters[-1].upper()

    return None


def evaluate_single(
    prediction: str,
    ground_truth: str,
) -> Tuple[bool, Optional[str]]:
    """
    评估单个样本的预测结果。

    Args:
        prediction: 模型输出文本。
        ground_truth: 标准答案字母。

    Returns:
        tuple: (is_correct, extracted_answer)
    """
    extracted = extract_mcq_answer(prediction)
    is_correct = (
        extracted is not None
        and extracted.upper() == ground_truth.strip().upper()
    )
    return is_correct, extracted


def evaluate_dataset(
    dataset: List[Dict],
    prediction_key: str = "model_answer",
    answer_key: str = "answer",
) -> Dict:
    """
    评估整个数据集的预测准确率。

    Args:
        dataset: 包含模型预测的样本列表。
        prediction_key: 预测结果对应的键名。
        answer_key: 标准答案对应的键名。

    Returns:
        dict: 评估结果，包含总准确率、各子领域准确率、各难度准确率等。
    """
    total = 0
    correct = 0
    failed_extraction = 0

    subtopic_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    language_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for sample in dataset:
        prediction = sample.get(prediction_key)
        ground_truth = sample.get(answer_key, "")

        if prediction is None:
            failed_extraction += 1
            total += 1
            continue

        is_correct, extracted = evaluate_single(prediction, ground_truth)
        sample["extracted_answer"] = extracted
        sample["is_correct"] = is_correct

        total += 1
        if is_correct:
            correct += 1

        subtopic = sample.get("subtopic", "unknown")
        subtopic_stats[subtopic]["total"] += 1
        if is_correct:
            subtopic_stats[subtopic]["correct"] += 1

        difficulty = sample.get("difficulty", 0)
        difficulty_stats[difficulty]["total"] += 1
        if is_correct:
            difficulty_stats[difficulty]["correct"] += 1

        language = sample.get("language", "unknown")
        language_stats[language]["total"] += 1
        if is_correct:
            language_stats[language]["correct"] += 1

    def _acc(stats: Dict) -> Dict:
        return {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in sorted(stats.items())
        }

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed_extraction,
        "by_subtopic": _acc(subtopic_stats),
        "by_difficulty": _acc(difficulty_stats),
        "by_language": _acc(language_stats),
    }


def print_evaluation_report(results: Dict) -> None:
    """
    打印格式化的评估报告。

    Args:
        results: evaluate_dataset 返回的结果字典。
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.2%} "
          f"({results['correct']}/{results['total']})")

    if results["failed_extraction"] > 0:
        print(f"Failed Extractions: {results['failed_extraction']}")

    if results.get("by_subtopic"):
        print(f"\n{'--- By Subtopic ---':^60}")
        print(f"{'Subtopic':<40} {'Accuracy':>8} {'Count':>8}")
        print("-" * 60)
        for sub, stats in results["by_subtopic"].items():
            print(f"{sub:<40} {stats['accuracy']:>7.1%} "
                  f"{stats['correct']:>3}/{stats['total']:<4}")

    if results.get("by_difficulty"):
        print(f"\n{'--- By Difficulty ---':^60}")
        print(f"{'Level':<40} {'Accuracy':>8} {'Count':>8}")
        print("-" * 60)
        for diff, stats in results["by_difficulty"].items():
            print(f"Difficulty {diff:<29} {stats['accuracy']:>7.1%} "
                  f"{stats['correct']:>3}/{stats['total']:<4}")

    if results.get("by_language"):
        print(f"\n{'--- By Language ---':^60}")
        print(f"{'Language':<40} {'Accuracy':>8} {'Count':>8}")
        print("-" * 60)
        for lang, stats in results["by_language"].items():
            print(f"{lang:<40} {stats['accuracy']:>7.1%} "
                  f"{stats['correct']:>3}/{stats['total']:<4}")

    print("=" * 60)


def save_evaluation_report(
    results: Dict,
    output_path: str,
) -> None:
    """
    将评估结果保存为 JSON 文件。

    Args:
        results: evaluate_dataset 返回的结果字典。
        output_path: 输出 JSON 文件路径。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
