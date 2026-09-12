"""Simple Plotly views for geometry and StructureOutput. Not a CAD kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import plotly.graph_objects as go

from schemas import GeometryOutput, ImportedCandidateGeometry, StructureOutput

Vec3 = Tuple[float, float, float]


def _desk_mesh(thickness: float, width: float = 220.0, depth: float = 160.0) -> go.Mesh3d:
    x0, x1 = -depth, 0.0
    y0, y1 = -width / 2.0, width / 2.0
    z0, z1 = 0.0, thickness
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 1, 2, 0]
    j = [1, 3, 5, 2, 3, 4]
    k = [2, 2, 6, 5, 7, 7]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color="#c4b7a6", opacity=0.45, name="desk",
        hovertext="desk / support",
    )


def _cylinder(radius: float, x: float, z0: float, z1: float, color: str, name: str) -> go.Scatter3d:
    import math

    theta = [n * 2.0 * math.pi / 24 for n in range(25)]
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for z in (z0, z1):
        for t in theta:
            xs.append(x + radius * 0.15 * math.cos(t))
            ys.append(radius * math.sin(t))
            zs.append(z)
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=6),
        name=name,
    )


def geometry_figure(geometry: GeometryOutput) -> go.Figure:
    desk_t = geometry.environment.desk_thickness_mm or 25.0
    diameter = geometry.payload_object.bottle_diameter_mm or 80.0
    height = geometry.payload_object.bottle_height_mm or 220.0
    protrusion = geometry.design_envelope.max_protrusion_mm or 120.0
    fig = go.Figure()
    fig.add_trace(_desk_mesh(desk_t))
    fig.add_trace(
        _cylinder(diameter / 2.0, protrusion * 0.85, desk_t, desk_t + height * 0.45, "#4c8bf5", "payload")
    )
    for region in geometry.attachment_regions:
        x, y, z = region.position_mm
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z], mode="markers",
                marker=dict(size=10, color="#2ecc71", symbol="square"),
                name=f"attach:{region.name}",
            )
        )
    for region in geometry.load_regions:
        x, y, z = region.position_mm
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z], mode="markers",
                marker=dict(size=10, color="#e74c3c"),
                name=f"load:{region.name}",
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="X mm (out from desk)",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(font=dict(size=10)),
        title="Simplified geometry (not CAD)",
        height=420,
    )
    return fig


def structure_figure(
    structure: StructureOutput, geometry: Optional[GeometryOutput] = None
) -> go.Figure:
    fig = go.Figure()
    if geometry is not None:
        fig.add_trace(_desk_mesh(geometry.environment.desk_thickness_mm or 25.0))
    by_id = {node.id: node.position_mm for node in structure.nodes}
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] for p in by_id.values()],
            y=[p[1] for p in by_id.values()],
            z=[p[2] for p in by_id.values()],
            mode="markers+text",
            text=list(by_id.keys()),
            textposition="top center",
            marker=dict(size=7, color="#1f77b4"),
            name="nodes",
        )
    )
    for member in structure.members:
        a = by_id.get(member.start_node_id)
        b = by_id.get(member.end_node_id)
        if a is None or b is None:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                mode="lines",
                line=dict(color="#444", width=6),
                name=member.id,
            )
        )
    for region in structure.attachment_regions:
        x, y, z = region.position_mm
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z], mode="markers",
                marker=dict(size=11, color="#2ecc71", symbol="diamond"),
                name=f"support:{region.name}",
            )
        )
    if structure.load_cases:
        load = structure.load_cases[0]
        region = next(
            (item for item in structure.load_regions if item.name == load.region_name),
            None,
        )
        if region is not None:
            x, y, z = region.position_mm
            fx, fy, fz = load.force_N
            scale = 4.0
            fig.add_trace(
                go.Scatter3d(
                    x=[x, x + fx * scale],
                    y=[y, y + fy * scale],
                    z=[z, z + fz * scale],
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=8),
                    marker=dict(size=4, color="#e74c3c"),
                    name=f"load {load.load_case_id}",
                )
            )
    fig.update_layout(
        scene=dict(
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Structural concept (nodes / members)",
        height=460,
        legend=dict(font=dict(size=10)),
    )
    return fig


def imported_candidate_mesh_figure(candidate: ImportedCandidateGeometry) -> go.Figure:
    return _mesh_figure_from_path(candidate.mesh_path)


@lru_cache(maxsize=2)
def _mesh_figure_from_path(mesh_path: str) -> go.Figure:
    from imported_candidate import parse_obj_mesh

    vertices, faces = parse_obj_mesh(Path(mesh_path))
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    i = [f[0] for f in faces]
    j = [f[1] for f in faces]
    k = [f[2] for f in faces]
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i,
                j=j,
                k=k,
                color="#6b8cae",
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2),
                name="imported candidate",
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        title="IMPORTED CANDIDATE MESH",
        height=480,
    )
    return fig


def imported_candidate_particle_figure(candidate: ImportedCandidateGeometry) -> go.Figure:
    from imported_candidate import parse_obj_points

    if not candidate.particle_path:
        fig = go.Figure()
        fig.update_layout(title="No particle representation available", height=420)
        return fig
    points = parse_obj_points(Path(candidate.particle_path), limit=8000)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode="markers",
                marker=dict(size=2, color="#334155"),
                name="particles",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        title="IMPORTED CANDIDATE MESH — point / particle representation",
        height=480,
    )
    return fig
