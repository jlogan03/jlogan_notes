from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    svg_path = output_dir / "float_relative_resolution.svg"
    html_path = output_dir / "float_relative_resolution.html"

    float_specs = {
        "f16": {
            "eps": 2.0**-10,
            "min_normal": 2.0**-14,
            "min_subnormal": 2.0**-24,
            "max": 65504.0,
        },
        "f32": {
            "eps": 2.0**-23,
            "min_normal": 2.0**-126,
            "min_subnormal": 2.0**-149,
            "max": 3.4028235e38,
        },
        "f64": {
            "eps": 2.0**-52,
            "min_normal": 2.0**-1022,
            "min_subnormal": 2.0**-1074,
            "max": 1.7976931348623157e308,
        },
    }

    line_color = "rgb(139, 196, 59)"
    marker_symbols = ["circle-open", "square-open", "diamond-open"]

    fig = go.Figure()
    for idx, (label, spec) in enumerate(float_specs.items()):
        eps = spec["eps"]
        min_normal = spec["min_normal"]
        min_subnormal = spec["min_subnormal"]
        max_value = spec["max"]
        normal_bits = -np.log2(eps)
        subnormal_x = np.logspace(np.log10(min_subnormal), np.log10(min_normal), 320)
        normal_x = np.logspace(np.log10(min_normal), np.log10(max_value), 240)
        x = np.concatenate([subnormal_x[:-1], normal_x])
        bits = np.full_like(x, normal_bits)
        subnormal_mask = (x < min_normal) & (x >= min_subnormal)
        bits[subnormal_mask] = np.floor(np.log2(x[subnormal_mask] / min_subnormal))
        bits[x < min_subnormal] = 0.0
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bits,
                mode="lines",
                line=dict(color=line_color, width=3, shape="hv"),
                name=label,
                showlegend=False,
            )
        )
        marker_count = 8
        if label == "f32":
            marker_count = 4
        elif label == "f16":
            marker_count = 2
        marker_indices = np.linspace(0, len(x) - 1, marker_count, dtype=int)
        fig.add_trace(
            go.Scatter(
                x=x[marker_indices],
                y=bits[marker_indices],
                mode="markers",
                marker=dict(symbol=marker_symbols[idx], size=11, color=line_color),
                name=label,
                hoverinfo="skip",
            )
        )
        ay = -30 + idx * 20
        if label == "f32":
            ay -= 10
        elif label == "f64":
            ay += 10
        fig.add_annotation(
            x=1.0,
            y=normal_bits,
            text=f"epsilon ≈ {eps:.2e}",
            showarrow=True,
            arrowcolor="white",
            font=dict(color="white"),
            ax=40,
            ay=ay,
        )

    fig.update_xaxes(
        type="log",
        title_text="Magnitude",
        tickformat=".1e",
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Resolution (bits)",
        tickformat=".1f",
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title=dict(text="Relative Resolution of IEEE-754 Floats", y=0.97, yanchor="top"),
        height=420,
        margin=dict(t=60, l=70, r=40, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=1.0,
        ),
    )

    fig.write_image(str(svg_path))
    fig.write_html(str(html_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
