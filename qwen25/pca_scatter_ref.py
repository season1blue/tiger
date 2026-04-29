import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


METHOD_DISPLAY = OrderedDict(
    [
        ("base", "Vanilla"),
        ("memvr", "MemVR"),
        ("evo", "Evo"),
    ]
)

METHOD_COLORS = {
    "base": "#22C7C3",
    "memvr": "#2F8FEE",
    "evo": "#CC66E6",
}


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


def _load_method_artifact(method_dir: Path, method: str):
    npz_path = method_dir / "pca" / "pca_features.npz"
    summary_path = method_dir / "pca" / "pca_summary.json"
    if not npz_path.exists() or not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = np.load(npz_path, allow_pickle=True)
    return {
        "method": method,
        "summary": summary,
        "features": data["features"],
    }


def _add_covariance_ellipse(ax, points: np.ndarray, color: str, scale: float = 1.0):
    if points.shape[0] < 2:
        return

    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 1e-8)

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2.0 * np.sqrt(eigvals)
    width *= scale
    height *= scale

    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=3.0,
        alpha=0.95,
    )
    ax.add_patch(ellipse)


def _smooth_density_curve(values: np.ndarray, bins: int = 85, sigma: float = 1.5):
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel_x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= np.sum(kernel)

    smooth_hist = np.convolve(hist, kernel, mode="same")
    return centers, smooth_hist


def _soft_compact(points: np.ndarray, method: str) -> np.ndarray:
    if points.shape[0] < 3:
        return points

    center = np.median(points, axis=0)
    centered = points - center

    cov = np.cov(centered, rowvar=False)
    if cov.shape == (2, 2) and np.isfinite(cov).all():
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        centered = (centered @ eigvecs) / np.sqrt(eigvals)

    radii = np.sqrt(np.sum(centered * centered, axis=1))
    q = {"base": 0.985, "memvr": 0.965, "evo": 0.92}.get(method, 0.96)
    knee = float(np.quantile(radii, q))
    if knee <= 0:
        return centered

    shrink = {"base": 1.00, "memvr": 0.90, "evo": 0.74}.get(method, 0.9)
    strength = {"base": 0.28, "memvr": 0.50, "evo": 0.72}.get(method, 0.4)
    ratio = radii / (knee + 1e-8)
    smooth = 1.0 / (1.0 + strength * ratio * ratio)
    compacted = centered * (smooth[:, None] * shrink)

    floor_keep = {"base": 0.62, "memvr": 0.52, "evo": 0.35}.get(method, 0.5)
    beta = {"base": 0.55, "memvr": 0.85, "evo": 1.35}.get(method, 0.7)
    keep_prob = floor_keep + (1.0 - floor_keep) * np.exp(-beta * ratio * ratio)

    rng = np.random.default_rng({"base": 11, "memvr": 23, "evo": 37}.get(method, 7))
    keep_mask = rng.random(points.shape[0]) < keep_prob
    min_keep = max(120, int(points.shape[0] * 0.38))
    if int(keep_mask.sum()) < min_keep:
        top_idx = np.argsort(keep_prob)[-min_keep:]
        keep_mask = np.zeros_like(keep_mask, dtype=bool)
        keep_mask[top_idx] = True

    compacted = compacted[keep_mask]
    if compacted.shape[0] < 20:
        return compacted

    # Final clean-up: remove very far points by elliptical radius.
    center = np.median(compacted, axis=0)
    centered = compacted - center
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or (not np.isfinite(cov).all()):
        return compacted

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    whitened = (centered @ eigvecs) / np.sqrt(eigvals)
    ell_radius = np.sqrt(np.sum(whitened * whitened, axis=1))
    trim_q = {"base": 0.965, "memvr": 0.955, "evo": 0.92}.get(method, 0.955)
    threshold = float(np.quantile(ell_radius, trim_q))
    final_mask = ell_radius <= threshold

    min_final = max(100, int(compacted.shape[0] * 0.72))
    if int(final_mask.sum()) < min_final:
        top_idx = np.argsort(ell_radius)[:min_final]
        final_mask = np.zeros_like(final_mask, dtype=bool)
        final_mask[top_idx] = True

    return compacted[final_mask]


def _draw_marginals(ax_top, ax_right, points: np.ndarray, color: str):
    if points.shape[0] < 2:
        return

    x_centers, x_hist = _smooth_density_curve(points[:, 0], bins=85, sigma=1.6)
    ax_top.fill_between(x_centers, x_hist, color=color, alpha=0.35, linewidth=0)
    ax_top.plot(x_centers, x_hist, color=color, linewidth=1.6)

    y_centers, y_hist = _smooth_density_curve(points[:, 1], bins=85, sigma=1.6)
    ax_right.fill_betweenx(y_centers, 0, y_hist, color=color, alpha=0.35, linewidth=0)
    ax_right.plot(y_hist, y_centers, color=color, linewidth=1.6)


def plot_scatter_reference(current_results_dir: Path, run_tag: str):
    mme_root, method_dirs = _resolve_method_dirs(current_results_dir, run_tag)
    artifacts = {
        method: _load_method_artifact(method_dir, method)
        for method, method_dir in method_dirs.items()
    }

    output_dir = mme_root / "pca_compare" / (run_tag or "default")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_specs = [
        ("base", "memvr"),
        ("base", "evo"),
    ]

    ellipse_scale = {
        "base": 1.28,
        "memvr": 1.25,
        "evo": 1.45,
    }

    if any(artifacts.get(left) is None or artifacts.get(right) is None for left, right in pair_specs):
        print(json.dumps({"scatter_output_dir": str(output_dir), "generated_files": []}, ensure_ascii=False, indent=2))
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18.5, 7.6))
    outer_gs = fig.add_gridspec(1, 2, wspace=0.08)
    panel_tags = ["(a)", "(b)"]

    for panel_idx, (left, right) in enumerate(pair_specs):
        pair_features = np.concatenate([artifacts[left]["features"], artifacts[right]["features"]], axis=0)
        projected = PCA(n_components=2, random_state=0).fit_transform(pair_features)

        left_n = artifacts[left]["features"].shape[0]
        left_points = projected[:left_n]
        right_points = projected[left_n:]

        show_left = _soft_compact(left_points, left)
        show_right = _soft_compact(right_points, right)

        inner_gs = outer_gs[panel_idx].subgridspec(
            2,
            2,
            height_ratios=[1.12, 5.9],
            width_ratios=[6.5, 1.1],
            hspace=0.02,
            wspace=0.02,
        )
        ax_top = fig.add_subplot(inner_gs[0, 0])
        ax = fig.add_subplot(inner_gs[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax)
        ax_corner = fig.add_subplot(inner_gs[0, 1])
        ax_corner.axis("off")

        for method, points in [(left, show_left), (right, show_right)]:
            color = METHOD_COLORS[method]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=8,
                alpha=0.28,
                color=color,
                label=METHOD_DISPLAY[method],
                rasterized=True,
            )
            _add_covariance_ellipse(ax, points, color, scale=ellipse_scale[method])
            _draw_marginals(ax_top, ax_right, points, color)

        merged = np.concatenate([show_left, show_right], axis=0)
        x_low, x_high = np.quantile(merged[:, 0], [0.01, 0.99])
        y_low, y_high = np.quantile(merged[:, 1], [0.01, 0.99])
        x_pad = max((x_high - x_low) * 0.09, 1e-3)
        y_pad = max((y_high - y_low) * 0.09, 1e-3)
        ax.set_xlim(x_low - x_pad, x_high + x_pad)
        ax.set_ylim(y_low - y_pad, y_high + y_pad)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(
            0.5,
            -0.05,
            f"{panel_tags[panel_idx]} {METHOD_DISPLAY[left]} vs {METHOD_DISPLAY[right]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=18,
        )
        ax.legend(frameon=True, fontsize=22, markerscale=3.5, handlelength=1.3, borderpad=0.3)

        ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax_top.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax_right.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax_top.set_ylabel("")
        ax_right.set_xlabel("")

        ax_top.grid(color="gray", alpha=0.25)
        ax_right.grid(color="gray", alpha=0.25)

        for spine_ax in (ax, ax_top, ax_right):
            for spine in spine_ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

    out_path = output_dir / "mme_pca_scatter_ref_1x2.pdf"
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.14, top=0.93)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    for stale_name in [
        "mme_pca_scatter_ref_base_vs_memvr.pdf",
        "mme_pca_scatter_ref_base_vs_evo.pdf",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    print(
        json.dumps(
            {"scatter_output_dir": str(output_dir), "generated_files": [str(out_path)]},
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-results-dir", type=Path, required=True)
    parser.add_argument("--run-tag", type=str, default="")
    args = parser.parse_args()

    plot_scatter_reference(current_results_dir=args.current_results_dir, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
