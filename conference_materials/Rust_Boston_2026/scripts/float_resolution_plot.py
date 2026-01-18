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
            "dtype": np.float16,
        },
        "f32": {
            "eps": 2.0**-23,
            "min_normal": 2.0**-126,
            "min_subnormal": 2.0**-149,
            "max": 3.4028235e38,
            "dtype": np.float32,
        },
        "f64": {
            "eps": 2.0**-52,
            "min_normal": 2.0**-1022,
            "min_subnormal": 2.0**-1074,
            "max": 1.7976931348623157e308,
            "dtype": np.float64,
        },
    }

    line_color = "#3b76c4"
    marker_symbols = ["circle-open", "square-open", "diamond-open"]

    fig = go.Figure()
    for idx, (label, spec) in enumerate(float_specs.items()):
        eps = spec["eps"]
        min_normal = spec["min_normal"]
        min_subnormal = spec["min_subnormal"]
        max_value = spec["max"]
        dtype = spec["dtype"]
        subnormal_x = np.logspace(
            np.log10(min_subnormal), np.log10(min_normal), 1200
        )
        normal_x = np.logspace(
            np.log10(min_normal), np.log10(max_value), 240
        )
        x = np.concatenate([subnormal_x[:-1], normal_x])
        normal_bits = -np.log2(eps)
        bits = np.full_like(x, normal_bits)
        subnormal_mask = (x < min_normal) & (x >= min_subnormal)
        subnormal_x_vals = x[subnormal_mask].astype(dtype, copy=False)
        min_subnormal_val = np.array(min_subnormal, dtype=dtype)
        subnormal_x_vals = np.maximum(subnormal_x_vals, min_subnormal_val)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            next_up = np.nextafter(subnormal_x_vals, np.array(np.inf, dtype=dtype))
            next_down = np.nextafter(subnormal_x_vals, np.array(-np.inf, dtype=dtype))
            delta_up = (next_up - subnormal_x_vals).astype(np.float64, copy=False)
            delta_down = (subnormal_x_vals - next_down).astype(np.float64, copy=False)
            local_resolution = np.maximum(delta_up, delta_down)
        bits[subnormal_mask] = -np.log2(local_resolution / subnormal_x_vals)
        bits[x <= float(min_subnormal)] = 0.0
        bits[~np.isfinite(bits)] = 0.0
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
            marker_count = 3
        elif label == "f16":
            marker_count = 1
        normal_indices = np.where(x >= min_normal)[0]
        marker_indices = np.linspace(
            normal_indices[0], normal_indices[-1], marker_count, dtype=int
        )
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
            arrowcolor="black",
            font=dict(color="black"),
            ax=40,
            ay=ay,
        )

    fig.update_xaxes(
        type="log",
        title_text="Magnitude",
        tickformat=".1e",
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Resolution (bits)",
        tickformat=".1f",
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title=dict(text="Relative Resolution of IEEE-754 Floats", y=0.97, yanchor="top"),
        height=420,
        margin=dict(t=60, l=70, r=40, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
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
