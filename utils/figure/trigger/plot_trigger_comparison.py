import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    plt.style.use("classic")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 15,
        }
    )

    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    metric_names = ["Overall", "Perception", "Cognition"]
    model_names = ["Qwen2.5-VL", "LLaVA-1.5"]
    method_names = ["Logits Entropy", "Image State Direction"]

    values = np.array(
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

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)
    bar_width = 0.34
    x = np.arange(len(model_names))
    offsets = np.array([-bar_width / 2, bar_width / 2])

    for metric_idx, ax in enumerate(axes):
        metric_values = np.array(
            [
                [values[0, 0, metric_idx], values[0, 1, metric_idx]],
                [values[1, 0, metric_idx], values[1, 1, metric_idx]],
            ]
        )
        y_min = max(0, float(metric_values.min()) * 0.97)
        y_max = float(metric_values.max()) * 1.04
        label_offset = max((y_max - y_min) * 0.015, 5)

        for method_idx, method_name in enumerate(method_names):
            bars = ax.bar(
                x + offsets[method_idx],
                metric_values[:, method_idx],
                width=bar_width,
                color=colors[method_idx],
                edgecolor="#1f1f1f",
                linewidth=0.8,
                label=method_name if metric_idx == 0 else None,
            )
            if hatch_patterns[method_idx] is not None:
                for bar in bars:
                    bar.set_hatch(hatch_patterns[method_idx])

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + label_offset,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

        ax.set_title(metric_names[metric_idx], fontsize=17)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", lw=2, linestyle="--", color="gray", alpha=0.7)
        ax.set_axisbelow(True)
        group_half_span = bar_width
        side_padding = 0.06
        ax.set_xlim(-group_half_span - side_padding, (len(model_names) - 1) + group_half_span + side_padding)
        ax.set_ylim(y_min, y_max)

        # if metric_idx == 0:
        #     ax.set_ylabel("Time (s)", fontsize=16)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=17,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(os.path.join(out_dir, "trigger_comparison.pdf"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()