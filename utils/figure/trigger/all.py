import os

import matplotlib.pyplot as plt
import numpy as np


def _draw_triplet_block(
    fig,
    outer_spec,
    metric_names,
    model_names,
    legend_names,
    values,
    colors,
    hatch_patterns,
    grid_alpha,
    block_tag,
):
    inner = outer_spec.subgridspec(1, 3, wspace=0.24)
    axes = [fig.add_subplot(inner[0, i]) for i in range(3)]

    bar_width = 0.34
    x = np.arange(len(model_names))
    offsets = np.array([-bar_width / 2, bar_width / 2])

    legend_handles = []

    for metric_idx, ax in enumerate(axes):
        metric_values = np.array(
            [
                [values[0, 0, metric_idx], values[0, 1, metric_idx]],
                [values[1, 0, metric_idx], values[1, 1, metric_idx]],
            ]
        )
        y_min = max(0, float(metric_values.min()) * 0.97)
        y_max = float(metric_values.max()) * 1.09
        label_offset = max((y_max - y_min) * 0.015, 5)

        for idx, name in enumerate(legend_names):
            bars = ax.bar(
                x + offsets[idx],
                metric_values[:, idx],
                width=bar_width,
                color=colors[idx],
                edgecolor="#1f1f1f",
                linewidth=0.8,
                label=name if metric_idx == 0 else None,
            )
            if hatch_patterns[idx] is not None:
                for bar in bars:
                    bar.set_hatch(hatch_patterns[idx])

            if metric_idx == 0:
                legend_handles.append(bars[0])

            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + label_offset,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=15,
                )

        ax.set_title(metric_names[metric_idx], fontsize=17)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.tick_params(axis="x")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.grid(axis="y", lw=2, linestyle="--", color="gray", alpha=grid_alpha)
        ax.set_axisbelow(True)

        group_half_span = bar_width
        side_padding = 0.06
        ax.set_xlim(-group_half_span - side_padding, (len(model_names) - 1) + group_half_span + side_padding)
        ax.set_ylim(y_min, y_max)

    if len(legend_handles) == 2:
        axes[1].legend(
            handles=legend_handles,
            labels=legend_names,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=2,
            frameon=False,
            fontsize=16,
        )

    axes[1].text(
        0.5,
        1.22,
        block_tag,
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )


def main() -> None:
    plt.style.use("classic")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
        }
    )

    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    metric_names = ["Overall", "Perception", "Cognition"]
    model_names = ["Qwen2.5-VL", "LLaVA-1.5"]

    feature_names = ["Static feature", "Layer-evolved Dynamic feature"]
    feature_values = np.array(
        [
            [
                [2324.00, 1691.00, 633.00],
                [2346.13, 1703.29, 642.84],
            ],
            [
                [1876.67, 1504.17, 372.50],
                [1892.89, 1513.97, 378.92],
            ],
        ]
    )

    trigger_names = ["Logits Entropy", "Image State Direction"]
    trigger_values = np.array(
        [
            [
                [2316.84, 1686.84, 630.00],
                [2346.13, 1703.29, 642.84],
            ],
            [
                [1854.67, 1506.46, 348.21],
                [1892.89, 1513.97, 378.92],
            ],
        ]
    )

    colors = ["#497fc0", "#c9393e"]
    hatch_patterns = [None, "///"]

    fig = plt.figure(figsize=(20, 4))
    outer = fig.add_gridspec(1, 2, wspace=0.12)

    _draw_triplet_block(
        fig=fig,
        outer_spec=outer[0],
        metric_names=metric_names,
        model_names=model_names,
        legend_names=feature_names,
        values=feature_values,
        colors=colors,
        hatch_patterns=hatch_patterns,
        grid_alpha=0.35,
        block_tag="(a) Revisited Feature",
    )

    _draw_triplet_block(
        fig=fig,
        outer_spec=outer[1],
        metric_names=metric_names,
        model_names=model_names,
        legend_names=trigger_names,
        values=trigger_values,
        colors=colors,
        hatch_patterns=hatch_patterns,
        grid_alpha=0.7,
        block_tag="(b) Strategy Trigger",
    )

    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.11, top=0.90)
    plt.savefig(os.path.join(out_dir, "feature_trigger_comparison_1x2.pdf"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
