import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


METHOD_DISPLAY = OrderedDict(
    [
        ("base", "Base"),
        ("memvr", "MemVR"),
        ("evo", "Evo"),
    ]
)

METHOD_COLORS = {
    "base": "#6C7A89",
    "memvr": "#8F63D2",
    "evo": "#1F8EFA",
}


def _nanmean_from_tensor(value):
    if value is None:
        return float("nan")
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() == 0:
        return float("nan")
    finite_mask = torch.isfinite(tensor)
    if not torch.any(finite_mask):
        return float("nan")
    return float(tensor[finite_mask].mean().item())


def _resolve_method_dirs(current_results_dir: Path, run_tag: str):
    current_results_dir = current_results_dir.resolve()
    if current_results_dir.name in METHOD_DISPLAY:
        mme_root = current_results_dir.parent
    elif current_results_dir.parent.name in METHOD_DISPLAY:
        mme_root = current_results_dir.parent.parent
    else:
        raise ValueError(f"Could not resolve MME root from: {current_results_dir}")

    method_dirs = {}
    for method in METHOD_DISPLAY:
        method_dir = mme_root / method
        if run_tag:
            method_dir = method_dir / run_tag
        method_dirs[method] = method_dir
    return mme_root, method_dirs


def _resolve_layer_index(num_layers: int, layer_index: int) -> int:
    resolved = layer_index if layer_index >= 0 else num_layers + layer_index
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(f"Invalid layer index {layer_index} for {num_layers} layers")
    return resolved


def _build_method_features(method_dir: Path, method: str, layer_index: int):
    analysis_dir = method_dir / "analysis"
    output_dir = method_dir / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(path for path in analysis_dir.glob("*.pt") if path.is_file())
    if not sample_files:
        return None

    rows = []
    feature_vectors = []
    resolved_layer_index = None

    for sample_path in sample_files:
        record = torch.load(sample_path, map_location="cpu")
        pooled_states = record.get("pooled_image_states")
        if pooled_states is None or pooled_states.ndim != 3 or pooled_states.shape[0] == 0:
            continue

        if resolved_layer_index is None:
            resolved_layer_index = _resolve_layer_index(pooled_states.shape[1], layer_index)

        metadata = record.get("metadata", {})
        feature_vector = pooled_states[:, resolved_layer_index, :].float().mean(dim=0).numpy()
        feature_vectors.append(feature_vector)

        rows.append(
            {
                "sample_id": str(metadata.get("sample_id", sample_path.stem)),
                "question_id": metadata.get("question_id", ""),
                "method": method,
                "final_prediction_correct": metadata.get("final_prediction_correct"),
                "is_hallucinated": metadata.get("is_hallucinated"),
                "generated_answer": metadata.get("generated_answer", ""),
                "ground_truth_answer": metadata.get("ground_truth_answer", ""),
                "mean_entropy": _nanmean_from_tensor(record.get("layer_entropy")[:, resolved_layer_index]),
                "mean_text_image_similarity": _nanmean_from_tensor(
                    record.get("text_image_similarity")[:, resolved_layer_index]
                ),
                "mean_update_norm": _nanmean_from_tensor(record.get("image_state_update_norms")[:, resolved_layer_index]),
                "mean_direction_shift": _nanmean_from_tensor(record.get("direction_scores")[:, resolved_layer_index]),
            }
        )

    if not feature_vectors:
        return None

    features = np.stack(feature_vectors, axis=0)
    npz_path = output_dir / "pca_features.npz"
    np.savez_compressed(
        npz_path,
        features=features,
        sample_ids=np.array([row["sample_id"] for row in rows], dtype=object),
        question_ids=np.array([row["question_id"] for row in rows], dtype=object),
        final_prediction_correct=np.array([row["final_prediction_correct"] for row in rows], dtype=object),
        is_hallucinated=np.array([row["is_hallucinated"] for row in rows], dtype=object),
        mean_entropy=np.array([row["mean_entropy"] for row in rows], dtype=np.float32),
        mean_text_image_similarity=np.array(
            [row["mean_text_image_similarity"] for row in rows], dtype=np.float32
        ),
        mean_update_norm=np.array([row["mean_update_norm"] for row in rows], dtype=np.float32),
        mean_direction_shift=np.array([row["mean_direction_shift"] for row in rows], dtype=np.float32),
        resolved_layer_index=np.array([resolved_layer_index], dtype=np.int32),
        method=np.array([method], dtype=object),
    )

    jsonl_path = output_dir / "pca_features.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    halluc_flags = np.array(
        [row["is_hallucinated"] for row in rows if row["is_hallucinated"] is not None], dtype=np.float32
    )
    correct_flags = np.array(
        [row["final_prediction_correct"] for row in rows if row["final_prediction_correct"] is not None],
        dtype=np.float32,
    )
    summary = {
        "method": method,
        "sample_count": len(rows),
        "feature_dim": int(features.shape[1]),
        "resolved_layer_index": int(resolved_layer_index),
        "hallucination_rate": float(halluc_flags.mean()) if halluc_flags.size else None,
        "accuracy": float(correct_flags.mean()) if correct_flags.size else None,
        "mean_entropy": float(np.nanmean([row["mean_entropy"] for row in rows])),
        "mean_text_image_similarity": float(np.nanmean([row["mean_text_image_similarity"] for row in rows])),
        "mean_update_norm": float(np.nanmean([row["mean_update_norm"] for row in rows])),
        "mean_direction_shift": float(np.nanmean([row["mean_direction_shift"] for row in rows])),
        "analysis_dir": str(analysis_dir),
        "npz_path": str(npz_path),
        "jsonl_path": str(jsonl_path),
    }

    summary_path = output_dir / "pca_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "method": method,
        "method_dir": method_dir,
        "output_dir": output_dir,
        "summary": summary,
        "summary_path": summary_path,
        "npz_path": npz_path,
        "rows": rows,
        "features": features,
    }


def _load_method_artifact(method_dir: Path, method: str):
    npz_path = method_dir / "pca" / "pca_features.npz"
    summary_path = method_dir / "pca" / "pca_summary.json"
    if not npz_path.exists() or not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = np.load(npz_path, allow_pickle=True)
    return {
        "method": method,
        "method_dir": method_dir,
        "npz_path": npz_path,
        "summary": summary,
        "features": data["features"],
        "sample_ids": data["sample_ids"],
        "question_ids": data["question_ids"],
        "final_prediction_correct": data["final_prediction_correct"],
        "is_hallucinated": data["is_hallucinated"],
        "mean_entropy": data["mean_entropy"],
        "mean_text_image_similarity": data["mean_text_image_similarity"],
        "mean_update_norm": data["mean_update_norm"],
        "mean_direction_shift": data["mean_direction_shift"],
        "resolved_layer_index": int(data["resolved_layer_index"][0]),
    }


def _add_covariance_ellipse(ax, points: np.ndarray, color: str):
    if points.shape[0] < 2:
        return

    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if np.any(eigvals <= 0):
        return

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2.0 * np.sqrt(eigvals)
    center = points.mean(axis=0)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=1.8,
        alpha=0.95,
    )
    ax.add_patch(ellipse)


def _write_projection_table(artifacts, projected, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "sample_id",
        "question_id",
        "pc1",
        "pc2",
        "mean_entropy",
        "mean_text_image_similarity",
        "mean_update_norm",
        "mean_direction_shift",
        "is_hallucinated",
        "final_prediction_correct",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        start = 0
        for artifact in artifacts:
            count = artifact["features"].shape[0]
            coords = projected[start : start + count]
            for idx in range(count):
                writer.writerow(
                    {
                        "method": artifact["method"],
                        "sample_id": artifact["sample_ids"][idx],
                        "question_id": artifact["question_ids"][idx],
                        "pc1": float(coords[idx, 0]),
                        "pc2": float(coords[idx, 1]),
                        "mean_entropy": float(artifact["mean_entropy"][idx]),
                        "mean_text_image_similarity": float(artifact["mean_text_image_similarity"][idx]),
                        "mean_update_norm": float(artifact["mean_update_norm"][idx]),
                        "mean_direction_shift": float(artifact["mean_direction_shift"][idx]),
                        "is_hallucinated": artifact["is_hallucinated"][idx],
                        "final_prediction_correct": artifact["final_prediction_correct"][idx],
                    }
                )
            start += count


def _plot_embedding_comparison(artifacts, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = np.concatenate([artifact["features"] for artifact in artifacts], axis=0)
    pca = PCA(n_components=2, random_state=0)
    projected = pca.fit_transform(all_features)
    explained = pca.explained_variance_ratio_ * 100.0

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    start = 0
    for artifact in artifacts:
        count = artifact["features"].shape[0]
        points = projected[start : start + count]
        start += count
        method = artifact["method"]
        color = METHOD_COLORS[method]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=11,
            alpha=0.28,
            color=color,
            label=f"{METHOD_DISPLAY[method]} (n={count})",
            rasterized=True,
        )
        _add_covariance_ellipse(ax, points, color)

    ax.set_title("MME Hidden-State PCA Comparison")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "mme_pca_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    _write_projection_table(artifacts, projected, output_dir / "mme_pca_projection.csv")


def _plot_summary_metrics(artifacts, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    metrics = [
        ("mean_entropy", "Mean Entropy"),
        ("mean_text_image_similarity", "Text-Image Similarity"),
        ("mean_update_norm", "State Update Norm"),
        ("mean_direction_shift", "Direction Shift"),
    ]
    methods = [artifact["method"] for artifact in artifacts]
    labels = [METHOD_DISPLAY[method] for method in methods]
    summary_method_colors = {
        "base": "#497fc0",
        "memvr": "#29517c",
        "evo": "#c9393e",
    }
    colors = [summary_method_colors.get(method, METHOD_COLORS.get(method, "#497fc0")) for method in methods]
    x_positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.6))
    axes = axes.flatten()

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        values = []
        for artifact in artifacts:
            value = artifact["summary"].get(metric_key)
            values.append(np.nan if value is None else float(value))

        bars = ax.bar(
            x_positions,
            values,
            width=0.62,
            color=colors,
            edgecolor="#1f1f1f",
            linewidth=0.8,
            zorder=3,
        )
        ax.set_title(metric_title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=0)
        for label in ax.get_xticklabels():
            if label.get_text() == "Evo":
                label.set_color(summary_method_colors["evo"])
                label.set_fontweight("bold")
        finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
        if finite_values.size:
            data_min = float(finite_values.min())
            data_max = float(finite_values.max())
            spread = data_max - data_min
            if spread == 0:
                spread = max(abs(data_max) * 0.05, 1e-6)
            pad = spread * 0.42
            lower = data_min - pad
            upper = data_max + pad
            if data_min >= 0:
                lower = max(0.0, lower)
            ax.set_ylim(lower, upper)
            label_offset = max(spread * 0.06, abs(data_max) * 0.01, 1e-6)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        clip_on=True,
                    )
        ax.grid(axis="y", linestyle="--", alpha=0.82)
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.92, wspace=0.18)
    fig.savefig(output_dir / "mme_pca_feature_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "mme_pca_feature_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def build_and_plot(current_results_dir: Path, method: str, run_tag: str, layer_index: int):
    mme_root, method_dirs = _resolve_method_dirs(current_results_dir, run_tag)
    current_method_dir = method_dirs[method]
    current_artifact = _build_method_features(current_method_dir, method, layer_index)
    if current_artifact is None:
        raise FileNotFoundError(f"No PCA analysis traces found under {current_method_dir / 'analysis'}")

    artifacts = []
    for method_name, method_dir in method_dirs.items():
        artifact = _load_method_artifact(method_dir, method_name)
        if artifact is not None:
            artifacts.append(artifact)

    comparison_dir = mme_root / "pca_compare" / (run_tag or "default")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    overview = {
        "methods_available": [artifact["method"] for artifact in artifacts],
        "comparison_dir": str(comparison_dir),
        "current_method_summary": current_artifact["summary"],
    }

    if len(artifacts) >= 2:
        _plot_embedding_comparison(artifacts, comparison_dir)
        _plot_summary_metrics(artifacts, comparison_dir)
        overview["comparison_ready"] = True
        overview["comparison_files"] = [
            str(comparison_dir / "mme_pca_comparison.png"),
            str(comparison_dir / "mme_pca_feature_summary.png"),
            str(comparison_dir / "mme_pca_feature_summary.pdf"),
            str(comparison_dir / "mme_pca_projection.csv"),
        ]
    else:
        overview["comparison_ready"] = False
        overview["comparison_files"] = []

    overview_path = current_method_dir / "pca" / "pca_overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(json.dumps(overview, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-results-dir", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_DISPLAY))
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--layer-index", type=int, default=-1)
    args = parser.parse_args()

    build_and_plot(
        current_results_dir=args.current_results_dir,
        method=args.method,
        run_tag=args.run_tag,
        layer_index=args.layer_index,
    )


if __name__ == "__main__":
    main()