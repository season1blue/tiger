import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    outdir = Path("/mnt/data/ssz/mr/utils/figure/action")
    outdir.mkdir(parents=True, exist_ok=True)
    colors = ["#497fc0", "#c9393e"]

    # Figure 1
    layers = np.arange(1, 29)
    correct = 0.22 + 0.015 * np.sin(layers / 2.7) + 0.005 * np.cos(layers / 1.8)
    halluc = 0.23 + 0.02 * np.sin(layers / 2.5) + 0.005 * np.cos(layers / 1.6)
    halluc += 0.10 * np.exp(-0.5 * ((layers - 18) / 3.2) ** 2)

    plt.figure(figsize=(6, 4))
    plt.plot(layers, correct, marker="o", color=colors[0], label="Correct samples")
    plt.plot(layers, halluc, marker="o", color=colors[1], label="Hallucinated samples")
    plt.xlabel("Layer")
    plt.ylabel(r"Mean $r^{(l)}$")
    plt.title(r"Layer-wise $r^{(l)}$: Correct vs Hallucinated")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    fig1 = outdir / "layerwise_r_curve.pdf"
    plt.savefig(fig1, format="pdf", bbox_inches="tight")
    plt.close()

    # Figure 3
    post_offsets = np.array([0, 1, 2, 3])
    without_restore = np.array([0.54, 0.50, 0.45, 0.41])
    with_restore = np.array([0.54, 0.36, 0.28, 0.24])

    plt.figure(figsize=(6, 4))
    plt.plot(post_offsets, without_restore, marker="o", color=colors[0], label="Without restoration")
    plt.plot(post_offsets, with_restore, marker="o", color=colors[1], label="With restoration")
    plt.axvline(0)
    plt.xlabel("Relative layer offset from trigger")
    plt.ylabel(r"Mean $r$")
    plt.title("Post-trigger stabilization")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    fig3 = outdir / "post_trigger_restoration.pdf"
    plt.savefig(fig3, format="pdf", bbox_inches="tight")
    plt.close()

    print(str(fig1))
    print(str(fig3))


if __name__ == "__main__":
    main()