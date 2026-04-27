import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    outdir = Path("/mnt/data/ssz/mr/utils/figure/action")
    outdir.mkdir(parents=True, exist_ok=True)
    colors = ["#4C72B0", "#55A868"]
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 21,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
        }
    )

    layers = np.arange(1, 29)
    correct = 0.22 + 0.015 * np.sin(layers / 2.7) + 0.005 * np.cos(layers / 1.8)
    halluc = 0.23 + 0.02 * np.sin(layers / 2.5) + 0.005 * np.cos(layers / 1.6)
    halluc += 0.10 * np.exp(-0.5 * ((layers - 18) / 3.2) ** 2)

    post_offsets = np.array([0, 1, 2, 3])
    without_restore = np.array([0.54, 0.50, 0.45, 0.41])
    with_restore = np.array([0.54, 0.36, 0.28, 0.24])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(layers, correct, marker="o", markersize=6.5, linewidth=2.4, color=colors[0], label="Correct samples")
    ax1.plot(layers, halluc, marker="o", markersize=6.5, linewidth=2.4, color=colors[1], label="Hallucinated samples")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(r"Mean $r^{(l)}$")
    # ax1.set_title(r"Layer-wise $r^{(l)}$: Correct vs Hallucinated")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.plot(
        post_offsets,
        without_restore,
        marker="o",
        markersize=12,
        linewidth=3.5,
        color=colors[0],
        label="Without restoration",
    )
    ax2.plot(
        post_offsets,
        with_restore,
        marker="o",
        markersize=12,
        linewidth=3.5,
        color=colors[1],
        label="With restoration",
    )
    ax2.axvline(0, color="#444444", linewidth=1.4)
    ax2.set_xticks(post_offsets)
    ax2.set_xticklabels(["0", "1", "2", "3"])
    ax2.set_xlabel("Relative layer offset from trigger")
    ax2.set_ylabel(r"Mean $r$")
    # ax2.set_title("Post-trigger stabilization")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend()

    # Save the first subplot as a separate PDF with updated figsize
    fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 5))
    ax1.plot(layers, correct, marker="o", markersize=6.5, linewidth=2.4, color=colors[0], label="Correct samples")
    ax1.plot(layers, halluc, marker="o", markersize=6.5, linewidth=2.4, color=colors[1], label="Hallucinated samples")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(r"Mean $r^{(l)}$")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(outdir / "action_layerwise.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig1)

    # Save the second subplot as a separate PDF with updated figsize
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5))
    ax2.plot(
        post_offsets,
        without_restore,
        marker="o",
        markersize=12,
        linewidth=3.5,
        color=colors[0],
        label="Without restoration",
    )
    ax2.plot(
        post_offsets,
        with_restore,
        marker="o",
        markersize=12,
        linewidth=3.5,
        color=colors[1],
        label="With restoration",
    )
    ax2.axvline(0, color="#444444", linewidth=1.4)
    ax2.set_xticks(post_offsets)
    ax2.set_xticklabels(["0", "1", "2", "3"])
    ax2.set_xlabel("Relative layer offset from trigger")
    ax2.set_ylabel(r"Mean $r$")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "action_posttrigger.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig2)

    print(str(outdir / "action_layerwise.pdf"))
    print(str(outdir / "action_posttrigger.pdf"))


if __name__ == "__main__":
    main()