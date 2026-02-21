from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import NamedTuple

import plotly.graph_objects as go
import torch


class Region(NamedTuple):
    pattern: int
    polygon_xy: list[tuple[float, float]]
    plane: tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceConfig:
    x_min: float = -3.1
    x_max: float = 3.1
    y_min: float = -2.3
    y_max: float = 2.3
    hidden_width: int = 8
    seed: int = 12


@dataclass(frozen=True)
class ReLUParams:
    hidden_w: torch.Tensor  # [hidden, 2]
    hidden_b: torch.Tensor  # [hidden]
    output_w: torch.Tensor  # [hidden]
    output_linear: torch.Tensor  # [2]
    output_b: torch.Tensor  # [1]


def make_relu_params(config: SurfaceConfig) -> ReLUParams:
    gen = torch.Generator().manual_seed(config.seed)
    hidden_w = torch.randn((config.hidden_width, 2), generator=gen) * 1.2
    hidden_b = torch.randn((config.hidden_width,), generator=gen) * 0.8
    output_w = torch.randn((config.hidden_width,), generator=gen) * 1.0
    output_linear = torch.randn((2,), generator=gen) * 0.28
    output_b = torch.randn((1,), generator=gen) * 0.2
    return ReLUParams(
        hidden_w=hidden_w,
        hidden_b=hidden_b,
        output_w=output_w,
        output_linear=output_linear,
        output_b=output_b,
    )


def bbox_polygon(config: SurfaceConfig) -> list[tuple[float, float]]:
    return [
        (config.x_min, config.y_min),
        (config.x_max, config.y_min),
        (config.x_max, config.y_max),
        (config.x_min, config.y_max),
    ]


def polygon_area(poly: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def dedupe_poly(poly: list[tuple[float, float]], eps: float = 1e-9) -> list[tuple[float, float]]:
    if not poly:
        return []

    out: list[tuple[float, float]] = []
    eps2 = eps * eps
    for p in poly:
        if not out:
            out.append(p)
            continue
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        if dx * dx + dy * dy > eps2:
            out.append(p)

    if len(out) > 1:
        dx = out[0][0] - out[-1][0]
        dy = out[0][1] - out[-1][1]
        if dx * dx + dy * dy <= eps2:
            out.pop()
    return out


def clip_with_halfplane(
    poly: list[tuple[float, float]],
    a: float,
    b: float,
    c: float,
    eps: float = 1e-9,
) -> list[tuple[float, float]]:
    if not poly:
        return []

    def eval_side(p: tuple[float, float]) -> float:
        return a * p[0] + b * p[1] + c

    def inside(v: float) -> bool:
        return v >= -eps

    result: list[tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        nxt = poly[(i + 1) % n]
        curr_v = eval_side(curr)
        nxt_v = eval_side(nxt)
        curr_in = inside(curr_v)
        nxt_in = inside(nxt_v)

        if curr_in and nxt_in:
            result.append(nxt)
        elif curr_in and not nxt_in:
            denom = curr_v - nxt_v
            if abs(denom) > eps:
                t = curr_v / denom
                ix = curr[0] + t * (nxt[0] - curr[0])
                iy = curr[1] + t * (nxt[1] - curr[1])
                result.append((ix, iy))
        elif not curr_in and nxt_in:
            denom = curr_v - nxt_v
            if abs(denom) > eps:
                t = curr_v / denom
                ix = curr[0] + t * (nxt[0] - curr[0])
                iy = curr[1] + t * (nxt[1] - curr[1])
                result.append((ix, iy))
            result.append(nxt)

    return dedupe_poly(result, eps=eps)


def plane_for_pattern(pattern: int, params: ReLUParams) -> tuple[float, float, float]:
    ax = float(params.output_linear[0].item())
    ay = float(params.output_linear[1].item())
    c = float(params.output_b[0].item())

    for i in range(params.hidden_w.shape[0]):
        if (pattern >> i) & 1:
            out_w = float(params.output_w[i].item())
            ax += out_w * float(params.hidden_w[i, 0].item())
            ay += out_w * float(params.hidden_w[i, 1].item())
            c += out_w * float(params.hidden_b[i].item())
    return ax, ay, c


def z_for_plane(plane: tuple[float, float, float], x: float, y: float) -> float:
    return plane[0] * x + plane[1] * y + plane[2]


def enumerate_regions(config: SurfaceConfig, params: ReLUParams) -> list[Region]:
    base_poly = bbox_polygon(config)
    hidden = params.hidden_w.shape[0]
    regions: list[Region] = []

    for pattern in range(1 << hidden):
        poly = list(base_poly)

        for i in range(hidden):
            sign = 1.0 if ((pattern >> i) & 1) else -1.0
            a = sign * float(params.hidden_w[i, 0].item())
            b = sign * float(params.hidden_w[i, 1].item())
            c = sign * float(params.hidden_b[i].item())
            poly = clip_with_halfplane(poly, a, b, c)
            if len(poly) < 3:
                break

        poly = dedupe_poly(poly, eps=1e-8)
        if len(poly) < 3:
            continue
        if abs(polygon_area(poly)) < 1e-5:
            continue

        regions.append(
            Region(
                pattern=pattern,
                polygon_xy=poly,
                plane=plane_for_pattern(pattern, params),
            )
        )
    return regions


def edge_key(p0: tuple[float, float], p1: tuple[float, float], ndigits: int = 7) -> tuple[tuple[float, float], tuple[float, float]]:
    a = (round(p0[0], ndigits), round(p0[1], ndigits))
    b = (round(p1[0], ndigits), round(p1[1], ndigits))
    return (a, b) if a <= b else (b, a)


def build_adjacency(regions: list[Region]) -> list[set[int]]:
    adjacency = [set() for _ in regions]
    edge_to_regions: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = defaultdict(list)

    for idx, region in enumerate(regions):
        poly = region.polygon_xy
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 < 1e-12:
                continue
            edge_to_regions[edge_key(p0, p1)].append(idx)

    for owners in edge_to_regions.values():
        if len(owners) < 2:
            continue
        uniq = list(dict.fromkeys(owners))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a = uniq[i]
                b = uniq[j]
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


def color_regions(adjacency: list[set[int]]) -> list[int]:
    n = len(adjacency)
    colors = [-1] * n
    bipartite = True

    for start in range(n):
        if colors[start] != -1:
            continue
        colors[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if colors[v] == -1:
                    colors[v] = 1 - colors[u]
                    q.append(v)
                elif colors[v] == colors[u]:
                    bipartite = False
                    break
            if not bipartite:
                break
        if not bipartite:
            break

    if bipartite:
        return colors

    colors = [-1] * n
    order = sorted(range(n), key=lambda idx: len(adjacency[idx]), reverse=True)
    for u in order:
        used = {colors[v] for v in adjacency[u] if colors[v] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
    return colors


def polygon_mesh_trace(region: Region, color: str, region_idx: int) -> go.Mesh3d:
    x = [p[0] for p in region.polygon_xy]
    y = [p[1] for p in region.polygon_xy]
    z = [z_for_plane(region.plane, p[0], p[1]) for p in region.polygon_xy]

    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for t in range(1, len(region.polygon_xy) - 1):
        i.append(0)
        j.append(t)
        k.append(t + 1)

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.94,
        flatshading=True,
        lighting=dict(ambient=0.58, diffuse=0.3, roughness=1.0, fresnel=0.08),
        lightposition=dict(x=110, y=-90, z=125),
        hovertemplate=f"region={region_idx}<br>pattern={region.pattern}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>f=%{{z:.2f}}<extra></extra>",
        showlegend=False,
    )


def polygon_edges_trace(regions: list[Region]) -> go.Scatter3d:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for region in regions:
        poly = region.polygon_xy
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            z0 = z_for_plane(region.plane, p0[0], p0[1])
            z1 = z_for_plane(region.plane, p1[0], p1[1])
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([z0, z1, None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="rgba(18,18,18,0.68)", width=2),
        hoverinfo="skip",
        showlegend=False,
    )


def lines_trace(
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    color: str,
    width: int,
) -> go.Scatter3d:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for p0, p1 in segments:
        xs.extend([p0[0], p1[0], None])
        ys.extend([p0[1], p1[1], None])
        zs.extend([p0[2], p1[2], None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
    )


def dashed_vertical_segments(
    x: float,
    y: float,
    z0: float,
    z1: float,
    chunks: int = 16,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    segs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    dz = (z1 - z0) / chunks
    for idx in range(chunks):
        if idx % 2 == 0:
            a = z0 + idx * dz
            b = z0 + (idx + 1) * dz
            segs.append(((x, y, a), (x, y, b)))
    return segs


def add_bounding_box(
    fig: go.Figure,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    top = [
        ((x_min, y_min, z_max), (x_max, y_min, z_max)),
        ((x_max, y_min, z_max), (x_max, y_max, z_max)),
        ((x_max, y_max, z_max), (x_min, y_max, z_max)),
        ((x_min, y_max, z_max), (x_min, y_min, z_max)),
    ]
    bottom = [
        ((x_min, y_min, z_min), (x_max, y_min, z_min)),
        ((x_max, y_min, z_min), (x_max, y_max, z_min)),
        ((x_max, y_max, z_min), (x_min, y_max, z_min)),
        ((x_min, y_max, z_min), (x_min, y_min, z_min)),
    ]
    fig.add_trace(lines_trace(top + bottom, color="rgba(15,15,15,0.96)", width=4))

    verts = []
    for x, y in [(x_min, y_min), (x_min, y_max), (x_max, y_max)]:
        verts.extend(dashed_vertical_segments(x, y, z_min, z_max, chunks=18))
    fig.add_trace(lines_trace(verts, color="rgba(39,154,57,0.95)", width=4))


def make_figure(config: SurfaceConfig) -> go.Figure:
    params = make_relu_params(config)
    regions = enumerate_regions(config, params)
    adjacency = build_adjacency(regions)
    color_ids = color_regions(adjacency)

    palette = [
        "rgba(59, 118, 196, 0.86)",
        "rgba(241, 184, 104, 0.86)",
        "rgba(143, 208, 150, 0.86)",
        "rgba(230, 139, 170, 0.86)",
    ]

    fig = go.Figure()
    z_vals: list[float] = []
    for idx, region in enumerate(regions):
        color = palette[color_ids[idx] % len(palette)]
        fig.add_trace(polygon_mesh_trace(region, color, idx))
        for x, y in region.polygon_xy:
            z_vals.append(z_for_plane(region.plane, x, y))

    fig.add_trace(polygon_edges_trace(regions))

    z_min = min(z_vals)
    z_max = max(z_vals)
    z_pad = 0.08 * (z_max - z_min if z_max > z_min else 1.0)
    add_bounding_box(
        fig=fig,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        z_min=z_min - z_pad,
        z_max=z_max + z_pad,
    )

    fig.update_layout(
        title_text="ReLU Activation Regions as Coplanar Polygons",
        title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=44, b=10),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1.58, y=1.15, z=0.62),
            camera=dict(
                eye=dict(x=1.55, y=-1.72, z=0.8),
                up=dict(x=0.0, y=0.0, z=1.0),
                center=dict(x=0.0, y=0.0, z=-0.08),
            ),
        ),
    )
    return fig


def main() -> None:
    seed = 31415926
    output_svg = "relu_stained_glass.svg"
    output_html = "relu_stained_glass.html"
    width = 1280
    height = 760
    scale = 1.0
    show_figure = True

    config = SurfaceConfig(seed=seed)
    fig = make_figure(config)
    fig.write_image(
        output_svg,
        format="svg",
        width=width,
        height=height,
        scale=scale,
    )
    print(f"Wrote {output_svg}")

    if output_html:
        fig.write_html(output_html, include_plotlyjs="cdn")
        print(f"Wrote {output_html}")

    if show_figure:
        fig.show()


if __name__ == "__main__":
    main()
