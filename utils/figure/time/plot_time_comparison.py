import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    methods = ["VCD", "OPERA", "ICD", "MemVR", "Ours", "Base"]
    times = [1.322, 0.62, 0.266, 0.251, 0.165, 0.134]

    colors = ["#B279A2", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#4C78A8"]
    x_positions = [0, 1, 2, 3, 4, 5.0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_positions, times, color=colors, edgecolor="#1f1f1f", linewidth=0.8, width=0.7)
    bars[-1].set_hatch("///")

    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    ax.axvline(4.5, color="#1f1f1f", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods)

    y_max = max(times) * 1.15
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.6, 5.6)

    for bar, value in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

    fig.tight_layout()
    fig.savefig("time_comparison.pdf", format="pdf", dpi=300)


if __name__ == "__main__":
    main()
