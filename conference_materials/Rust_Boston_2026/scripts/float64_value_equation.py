from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    svg_path = output_dir / "float64_value_equation.svg"

    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.axis("off")

    lines = [
        r"$e = 0 \Rightarrow x = (-1)^s \left(\sum_{i=1}^{52} b_i 2^{-i}\right) 2^{-1022}$",
        r"$1 \leq e \leq 2046 \Rightarrow x = (-1)^s \left(1 + \sum_{i=1}^{52} b_i 2^{-i}\right) 2^{e-1023}$",
        r"$e = 2047 \Rightarrow x = \pm\infty \; \mathrm{or} \; \mathrm{NaN}$",
    ]

    y_positions = [0.85, 0.4, 0.1]
    for y, line in zip(y_positions, lines):
        ax.text(
            0.02,
            y,
            line,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=18,
            color="black",
        )

    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
