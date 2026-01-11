from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    svg_path = output_dir / "lagrange_multipliers.svg"
    html_path = output_dir / "lagrange_multipliers.html"

    x = np.linspace(-0.5, 2.0, 300)
    y = np.linspace(-0.5, 2.0, 300)
    xx, yy = np.meshgrid(x, y)

    dark = "rgb(34, 34, 34)"
    light = "rgb(241, 241, 241)"
    green = "rgb(138, 196, 59)"

    # Objective: f(x, y) = x^2 + 2y^2 with constraint y = 1 - x + 0.6(x - 2/3)^2.
    zz = xx**2 + 2.0 * yy**2

    constraint_x = np.linspace(-0.2, 1.6, 300)
    constraint_y = 1.0 - constraint_x + 0.6 * (constraint_x - (2.0 / 3.0)) ** 2

    optimal_x = 2.0 / 3.0
    optimal_y = 1.0 / 3.0
    optimal_value = optimal_x**2 + 2.0 * optimal_y**2
    max_value = float(np.max(zz))

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=zz,
            contours=dict(
                showlabels=False,
                coloring="fill",
                start=optimal_value,
                end=max_value,
                size=(max_value - optimal_value) / 5.0,
            ),
            colorscale=[
                [0.0, dark],
                [1.0, light],
            ],
            line=dict(color=green, width=1),
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=constraint_x,
            y=constraint_y,
            mode="lines",
            line=dict(color=green, width=10),
            name="Constraint: x + y = 1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[optimal_x],
            y=[optimal_y],
            mode="markers",
            marker=dict(color=green, size=40, symbol="x"),
            name="Optimal point",
        )
    )

    fig.update_xaxes(
        range=[-0.5, 2.0],
        showline=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
    )
    fig.update_yaxes(
        range=[-0.5, 2.0],
        showline=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        height=480,
        width=480,
        margin=dict(t=0, l=0, r=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        showlegend=False,
    )

    fig.write_image(str(svg_path))
    fig.write_html(str(html_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
