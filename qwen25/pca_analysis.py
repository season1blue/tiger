import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


METHOD_DISPLAY = OrderedDict(
    [
        ("base", "Vanilla"),
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


def _cleanup_stale_scatter_outputs(output_dir: Path):
    stale_names = [
        "mme_pca_comparison.png",
        "mme_pca_comparison_base_vs_memvr.png",
        "mme_pca_comparison_base_vs_evo.png",
        "mme_pca_comparison_base_vs_memvr.pdf",
        "mme_pca_comparison_base_vs_evo.pdf",
        "mme_pca_projection.csv",
    ]
    for name in stale_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _plot_summary_metrics(artifacts, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("mean_entropy", "(a) Mean Entropy$\\downarrow$"),
        ("mean_text_image_similarity", "(b) Text-Image Similarity$\\downarrow$"),
        ("mean_update_norm", "(c) State Update$\\uparrow$"),
        ("mean_direction_shift", "(d) Direction Shift$\\downarrow$"),
    ]
    methods = [artifact["method"] for artifact in artifacts]
    labels = [METHOD_DISPLAY[method] for method in methods]
    summary_method_colors = {
        "base": "#497fc0",
        "memvr": "#9694e7",
        "evo": "#c9393e",
    }
    colors = [summary_method_colors.get(method, METHOD_COLORS.get(method, "#497fc0")) for method in methods]
    x_positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
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
        ax.set_title(metric_title, fontsize=17)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=0, fontsize=17)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
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
                    label_text = f"{value:.1f}" if metric_key == "mean_update_norm" else f"{value:.3f}"
                    ax.annotate(
                        label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=17,
                        clip_on=True,
                    )
        ax.grid(axis="x", linestyle="--", color="gray", alpha=0.82)
        ax.grid(axis="y", linestyle="--", color="gray", alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.92, wspace=0.28)
    fig.savefig(output_dir / "mme_pca_feature_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    stale_summary_png = output_dir / "mme_pca_feature_summary.png"
    if stale_summary_png.exists():
        stale_summary_png.unlink()


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

    if artifacts:
        _cleanup_stale_scatter_outputs(comparison_dir)
        _plot_summary_metrics(artifacts, comparison_dir)
        overview["comparison_ready"] = True
        overview["comparison_files"] = [
            str(comparison_dir / "mme_pca_feature_summary.pdf"),
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