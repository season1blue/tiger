import argparse
import json
from pathlib import Path


def normalize_pred(text: str) -> int:
    # Follow the POPE heuristic: if explicit negation appears, map to no.
    t = text.split(".")[0].replace(",", " ")
    words = t.split()
    if "No" in words or "no" in words or "not" in words:
        return 0
    return 1


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate(question_file: Path, answers_file: Path) -> dict:
    questions = [json.loads(x) for x in question_file.open("r")]
    answers = [json.loads(x) for x in answers_file.open("r")]

    answers_by_qid = {str(a["question_id"]): a.get("text", "") for a in answers}

    labels = []
    preds = []
    missing = 0

    for q in questions:
        qid = str(q["question_id"])
        gt = q["label"].strip().lower()
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
    yes_ratio = safe_div(sum(preds), len(preds))

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


def print_metrics(metric_key: str, metrics: dict) -> None:
    print(f"\n=== POPE {metric_key} ===")
    print(f"samples: {metrics['samples']}, missing_answers: {metrics['missing_answers']}")
    print(f"TP={metrics['TP']} FP={metrics['FP']} TN={metrics['TN']} FN={metrics['FN']}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1: {metrics['F1']:.4f}")
    print(f"Yes ratio: {metrics['Yes ratio']:.4f}")


def update_result_json(result_json: Path, metric_key: str, metrics: dict) -> None:
    if result_json.exists():
        content = result_json.read_text().strip()
        data = json.loads(content) if content else {}
    else:
        data = {}

    data[metric_key] = metrics
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", required=True, type=str)
    parser.add_argument("--answers-file", required=True, type=str)
    parser.add_argument("--metric-key", required=True, type=str)
    parser.add_argument("--result-json", required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate(Path(args.question_file), Path(args.answers_file))
    print_metrics(args.metric_key, metrics)
    update_result_json(Path(args.result_json), args.metric_key, metrics)
