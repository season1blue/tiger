import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )

    methods = ["VCD", "OPERA", "ICD", "MemVR", "Ours"]
    qwen_base = 0.134
    llava_base = 0.104
    qwen_times = np.array([1.322, 0.620, 0.266, 0.251, 0.165])
    llava_times = np.array([1.264, 0.599, 0.241, 0.209, 0.141])

    x_positions = np.arange(len(methods))
    bar_width = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8))
    bars_qwen = ax.bar(
        x_positions - bar_width / 2,
        qwen_times,
        width=bar_width,
        color="#497fc0",
        edgecolor="#1f1f1f",
        linewidth=0.8,
        label="Qwen2.5",
        zorder=3,
    )
    bars_llava = ax.bar(
        x_positions + bar_width / 2,
        llava_times,
        width=bar_width,
        color="#c9393e",
        edgecolor="#1f1f1f",
        linewidth=0.8,
        label="LLaVA",
        zorder=3,
    )

    label_offset = max(float(max(qwen_times.max(), llava_times.max())) * 0.012, 0.01)

    ax.axhline(qwen_base, color="#497fc0", linestyle="--", linewidth=1.6, zorder=2)
    ax.axhline(llava_base, color="#c9393e", linestyle="--", linewidth=1.6, zorder=2)

    for bar, value in zip(bars_qwen, qwen_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    # ax.text(len(methods) - 0.5, qwen_base + label_offset * 0.25, f"Vanilla {qwen_base:.3f}", color="#497fc0", ha="right", va="bottom", fontsize=10)
    # ax.text(len(methods) - 0.5, llava_base + label_offset * 0.25, f"Vanilla {llava_base:.3f}", color="#c9393e", ha="right", va="bottom", fontsize=10)
    for bar, value in zip(bars_llava, llava_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    ax.set_ylabel("Time (s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods)
    for label in ax.get_xticklabels():
        if label.get_text() == "Ours":
            label.set_color("#c9393e")
            label.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", alpha=0.82)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(float(qwen_times.max()), float(llava_times.max())) * 1.15)
    ax.set_xlim(-0.55, len(methods) - 0.45)
    legend_handles = [
        bars_qwen[0],
        bars_llava[0],
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.6),
    ]
    ax.legend(
        legend_handles,
        ["Qwen2.5", "LLaVA", "Vanilla (dashed)"],
        loc="upper right",
        frameon=False,
        fontsize=15,
    )

    out_path = os.path.join(os.path.dirname(__file__), "time_comparison.pdf")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
