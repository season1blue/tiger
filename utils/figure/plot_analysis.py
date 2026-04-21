#!/usr/bin/env python3
"""Offline analysis + plotting for MemVR Qwen2.5-VL logs.

Inputs (default):
  results/Qwen2.5-VL/pope/memvr/analysis/**/analysis_index.jsonl

Outputs (default):
  /mnt/data/ssz/mr/utils/figure/output/

Only matplotlib is used for plotting.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_float_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            arr = x.detach().to(torch.float32).cpu().numpy()
        else:
            arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.size == 0:
        return None
    return arr.astype(np.float64)


def _nanmean_safe(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        v = np.nanmean(arr)
    return float(v)


def _nanmax_safe(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        v = np.nanmax(arr)
    return float(v)


def _label_hallucinated(index_rec: dict[str, Any]) -> int | None:
    hall = index_rec.get("is_hallucinated", None)
    if hall is not None:
        try:
            return int(hall)
        except Exception:
            pass

    corr = index_rec.get("final_prediction_correct", None)
    if corr is None:
        return None
    return 0 if bool(corr) else 1


def _label_correct(index_rec: dict[str, Any]) -> int | None:
    corr = index_rec.get("final_prediction_correct", None)
    if corr is not None:
        return int(bool(corr))

    hall = _label_hallucinated(index_rec)
    if hall is None:
        return None
    return 1 - hall


def _simple_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mask = np.isfinite(y_score)
    y_true = y_true[mask].astype(np.int32)
    y_score = y_score[mask].astype(np.float64)

    if y_true.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    change_idx = np.r_[np.where(np.diff(y_score) != 0)[0], y_true.size - 1]
    tpr = tps[change_idx] / pos
    fpr = fps[change_idx] / neg

    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]

    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def _flatten_layer_step_curve(mat: np.ndarray) -> np.ndarray:
    # mat shape expected: [steps, layers]
    if mat.ndim != 2:
        return np.array([])
    return np.nanmean(mat, axis=0)


def _load_one_sample(pt_path: Path) -> dict[str, Any]:
    rec = torch.load(pt_path, map_location="cpu")

    direction_scores = _to_float_array(rec.get("direction_scores"))
    entropy_scores = _to_float_array(rec.get("layer_entropy"))
    pooled_states = _to_float_array(rec.get("pooled_image_states"))

    sample = {
        "direction_scores": direction_scores,
        "layer_entropy": entropy_scores,
        "pooled_image_states": pooled_states,
        "metadata": rec.get("metadata", {}),
        "steps": rec.get("steps", []),
    }
    return sample


def collect_summary(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index_files = sorted(analysis_root.rglob("analysis_index.jsonl"))

    for idx_file in index_files:
        split_name = idx_file.parent.name
        with idx_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_rec = json.loads(line)
                pt_path = Path(idx_rec.get("log_path", ""))
                if not pt_path.exists():
                    continue

                try:
                    sample = _load_one_sample(pt_path)
                except Exception:
                    continue

                direction = sample["direction_scores"]
                entropy = sample["layer_entropy"]

                max_r = _nanmax_safe(direction) if direction is not None else float("nan")
                mean_r = _nanmean_safe(direction) if direction is not None else float("nan")
                max_entropy = _nanmax_safe(entropy) if entropy is not None else float("nan")
                mean_entropy = _nanmean_safe(entropy) if entropy is not None else float("nan")

                trigger_layer = None
                trigger_step = None
                # Optional heuristic: first layer-step where r is maximal.
                if direction is not None and direction.size > 0 and np.isfinite(direction).any():
                    flat_idx = int(np.nanargmax(direction))
                    s_idx, l_idx = np.unravel_index(flat_idx, direction.shape)
                    trigger_step = int(s_idx + 1)
                    trigger_layer = int(l_idx)

                hall = _label_hallucinated(idx_rec)
                corr = _label_correct(idx_rec)

                rows.append(
                    {
                        "sample_id": idx_rec.get("sample_id"),
                        "question_id": idx_rec.get("question_id"),
                        "dataset_name": idx_rec.get("dataset_name"),
                        "split_name": split_name,
                        "correct_or_not": corr,
                        "hallucinated_or_not": hall,
                        "generated_answer": idx_rec.get("generated_answer"),
                        "ground_truth_answer": idx_rec.get("ground_truth_answer"),
                        "max_r": max_r,
                        "mean_r": mean_r,
                        "max_entropy": max_entropy,
                        "mean_entropy": mean_entropy,
                        "trigger_layer": trigger_layer,
                        "trigger_step": trigger_step,
                        "pt_path": str(pt_path),
                    }
                )
    return rows


def save_summary_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    fields = [
        "sample_id",
        "question_id",
        "dataset_name",
        "split_name",
        "correct_or_not",
        "hallucinated_or_not",
        "generated_answer",
        "ground_truth_answer",
        "max_r",
        "mean_r",
        "max_entropy",
        "mean_entropy",
        "trigger_layer",
        "trigger_step",
        "pt_path",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _split_groups(rows: list[dict[str, Any]], value_key: str) -> tuple[list[float], list[float]]:
    hall_vals: list[float] = []
    corr_vals: list[float] = []
    for r in rows:
        v = r.get(value_key)
        if v is None or not np.isfinite(v):
            continue
        hall = r.get("hallucinated_or_not")
        corr = r.get("correct_or_not")
        if hall is not None:
            if int(hall) == 1:
                hall_vals.append(float(v))
            elif int(hall) == 0:
                corr_vals.append(float(v))
        elif corr is not None:
            if int(corr) == 1:
                corr_vals.append(float(v))
            else:
                hall_vals.append(float(v))
    return corr_vals, hall_vals


def plot_box(values_correct: list[float], values_hall: list[float], title: str, ylabel: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    data = [values_correct, values_hall]
    labels = ["Correct", "Hallucinated"]
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc(rows: list[dict[str, Any]], out_png: Path, out_auc_json: Path) -> None:
    y = []
    r_scores = []
    e_scores = []
    for r in rows:
        hall = r.get("hallucinated_or_not")
        if hall is None:
            continue
        y.append(int(hall))
        r_scores.append(float(r.get("max_r", np.nan)))
        e_scores.append(float(r.get("max_entropy", np.nan)))

    y_arr = np.asarray(y, dtype=np.int32)
    r_arr = np.asarray(r_scores, dtype=np.float64)
    e_arr = np.asarray(e_scores, dtype=np.float64)

    fpr_r, tpr_r, auc_r = _simple_roc_auc(y_arr, r_arr)
    fpr_e, tpr_e, auc_e = _simple_roc_auc(y_arr, e_arr)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_r, tpr_r, label=f"max_r (AUC={auc_r:.4f})", linewidth=2)
    if np.isfinite(auc_e):
        plt.plot(fpr_e, tpr_e, label=f"max_entropy (AUC={auc_e:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: max_r vs max_entropy for hallucination")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    out_auc_json.parent.mkdir(parents=True, exist_ok=True)
    out_auc_json.write_text(
        json.dumps(
            {
                "auc_max_r": auc_r,
                "auc_max_entropy": auc_e,
                "num_samples": int(y_arr.size),
                "label_rule": "use hallucinated_or_not if available, else correct_or_not==0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_layerwise_curve(rows: list[dict[str, Any]], out_path: Path) -> None:
    corr_curves = []
    hall_curves = []

    for r in rows:
        pt_path = Path(r["pt_path"])
        try:
            sample = _load_one_sample(pt_path)
        except Exception:
            continue
        direction = sample["direction_scores"]
        if direction is None or direction.ndim != 2:
            continue
        curve = _flatten_layer_step_curve(direction)
        if curve.size == 0:
            continue

        hall = r.get("hallucinated_or_not")
        corr = r.get("correct_or_not")
        if hall is not None:
            if int(hall) == 1:
                hall_curves.append(curve)
            elif int(hall) == 0:
                corr_curves.append(curve)
        elif corr is not None:
            if int(corr) == 1:
                corr_curves.append(curve)
            else:
                hall_curves.append(curve)

    if not corr_curves and not hall_curves:
        return

    max_len = 0
    for c in corr_curves + hall_curves:
        max_len = max(max_len, len(c))

    def _pad(curves):
        arr = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for i, c in enumerate(curves):
            arr[i, : len(c)] = c
        return arr

    corr_arr = _pad(corr_curves) if corr_curves else None
    hall_arr = _pad(hall_curves) if hall_curves else None

    layers = np.arange(max_len)
    plt.figure(figsize=(7, 5))

    if corr_arr is not None:
        m = np.nanmean(corr_arr, axis=0)
        s = np.nanstd(corr_arr, axis=0)
        plt.plot(layers, m, label="Correct", linewidth=2)
        plt.fill_between(layers, m - s, m + s, alpha=0.2)

    if hall_arr is not None:
        m = np.nanmean(hall_arr, axis=0)
        s = np.nanstd(hall_arr, axis=0)
        plt.plot(layers, m, label="Hallucinated", linewidth=2)
        plt.fill_between(layers, m - s, m + s, alpha=0.2)

    plt.xlabel("Layer")
    plt.ylabel("Mean r^(l)")
    plt.title("Layer-wise trajectory instability")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def _pick_case(rows: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    # mode: correct | hallucinated | corrected
    candidates = []
    for r in rows:
        corr = r.get("correct_or_not")
        hall = r.get("hallucinated_or_not")
        if mode == "correct":
            if corr == 1 or hall == 0:
                candidates.append(r)
        elif mode == "hallucinated":
            if hall == 1 or corr == 0:
                candidates.append(r)
        elif mode == "corrected":
            # If no explicit corrected flag exists, choose a correct sample with high max_r as proxy.
            if r.get("corrected_by_ours") in (1, True):
                candidates.append(r)
            elif (corr == 1 or hall == 0) and np.isfinite(float(r.get("max_r", np.nan))):
                candidates.append(r)

    if not candidates:
        return None

    if mode == "hallucinated":
        return sorted(candidates, key=lambda x: float(x.get("max_r", -1e9)), reverse=True)[0]
    if mode == "corrected":
        return sorted(candidates, key=lambda x: float(x.get("max_r", -1e9)), reverse=True)[0]
    return candidates[0]


def plot_pca_case(case_row: dict[str, Any], out_path: Path, title: str) -> None:
    sample = _load_one_sample(Path(case_row["pt_path"]))
    pooled = sample["pooled_image_states"]
    if pooled is None:
        return

    # pooled: [steps, layers, hidden_dim]
    if pooled.ndim != 3:
        return

    # mean over decoding steps -> layer trajectory in hidden space
    layer_vectors = np.nanmean(pooled, axis=0)
    traj_2d = _pca_2d(layer_vectors)

    plt.figure(figsize=(6, 6))
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], marker="o", linewidth=1.8)

    for i in range(len(traj_2d) - 1):
        dx = traj_2d[i + 1, 0] - traj_2d[i, 0]
        dy = traj_2d[i + 1, 1] - traj_2d[i, 1]
        plt.arrow(traj_2d[i, 0], traj_2d[i, 1], dx, dy, head_width=0.02, length_includes_head=True, alpha=0.5)

    trig_layer = case_row.get("trigger_layer")
    if trig_layer is not None and 0 <= int(trig_layer) < len(traj_2d):
        i = int(trig_layer)
        plt.scatter([traj_2d[i, 0]], [traj_2d[i, 1]], c="red", s=70, label="Trigger layer")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(alpha=0.2)
    if trig_layer is not None:
        plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MemVR offline analysis figures")
    parser.add_argument(
        "--analysis-root",
        type=str,
        default="/mnt/data/ssz/mr/results/Qwen2.5-VL/pope/memvr/analysis",
        help="Root folder containing split subfolders with analysis_index.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/ssz/mr/utils/figure/output",
        help="Folder to save summary csv/json and figures",
    )
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_summary(analysis_root)
    if not rows:
        raise RuntimeError(f"No analysis samples found under: {analysis_root}")

    summary_csv = output_dir / "summary.csv"
    save_summary_csv(rows, summary_csv)

    # Box plots for r and entropy distributions.
    corr_max_r, hall_max_r = _split_groups(rows, "max_r")
    corr_mean_r, hall_mean_r = _split_groups(rows, "mean_r")
    plot_box(corr_max_r, hall_max_r, "max_r: Correct vs Hallucinated", "max_r", output_dir / "fig_box_max_r.png")
    plot_box(corr_mean_r, hall_mean_r, "mean_r: Correct vs Hallucinated", "mean_r", output_dir / "fig_box_mean_r.png")

    corr_max_e, hall_max_e = _split_groups(rows, "max_entropy")
    corr_mean_e, hall_mean_e = _split_groups(rows, "mean_entropy")
    if corr_max_e or hall_max_e:
        plot_box(
            corr_max_e,
            hall_max_e,
            "max_entropy: Correct vs Hallucinated",
            "max_entropy",
            output_dir / "fig_box_max_entropy.png",
        )
    if corr_mean_e or hall_mean_e:
        plot_box(
            corr_mean_e,
            hall_mean_e,
            "mean_entropy: Correct vs Hallucinated",
            "mean_entropy",
            output_dir / "fig_box_mean_entropy.png",
        )

    # ROC + AUC summary.
    plot_roc(rows, output_dir / "fig_roc_r_vs_entropy.png", output_dir / "auc_summary.json")

    # Layer-wise mean curve.
    plot_layerwise_curve(rows, output_dir / "fig_layerwise_r_curve.png")

    # PCA case studies.
    c1 = _pick_case(rows, "correct")
    c2 = _pick_case(rows, "hallucinated")
    c3 = _pick_case(rows, "corrected")
    if c1 is not None:
        plot_pca_case(c1, output_dir / "fig_pca_case_correct.png", "PCA Trajectory: Correct Case")
    if c2 is not None:
        plot_pca_case(c2, output_dir / "fig_pca_case_hallucinated.png", "PCA Trajectory: Hallucinated Case")
    if c3 is not None:
        plot_pca_case(c3, output_dir / "fig_pca_case_corrected.png", "PCA Trajectory: Corrected/Proxy Case")

    print(f"[done] summary: {summary_csv}")
    print(f"[done] figures dir: {output_dir}")


if __name__ == "__main__":
    main()
