"""Demo: free-standing stapler shelf (100 mm lift, flat top) — TOProblem from dims + point cloud.

Frame (as generated): base on the desk at z ≈ 0, platform top at z = lift height, +Z up.
The platform slab is preserved (the "guaranteed flat top"); the support column is the design region.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..contracts import (
    BoxRegion,
    IntersectionRegion,
    LoadCase,
    Material,
    NearPointsRegion,
    Support,
    TOProblem,
)
from ..ingest.dimensions import parse_dimensions
from ..ingest.point_cloud import load_point_cloud
from ..integration.materials import printed_material
from .cupholder import ASSUMED, _horizontal_faces


@dataclass
class ShelfGeometry:
    z_min: float
    z_top: float
    z_under: float  # platform underside
    z_base_top: float
    platform_x: tuple[float, float]
    platform_y: tuple[float, float]
    base_x: tuple[float, float]
    base_y: tuple[float, float]


def measure_shelf(points: np.ndarray, dims: dict) -> ShelfGeometry:
    z_min, z_top = float(points[:, 2].min()), float(points[:, 2].max())
    t_plat = float(dims.get("platform_thickness", 6.0))
    top = points[points[:, 2] > z_top - 2.5 * t_plat]
    faces = sorted(_horizontal_faces(top[:, 2]))
    below = [f for f in faces if f < z_top - 0.5 * t_plat]
    z_under = below[-1] if below else z_top - t_plat
    plat = points[points[:, 2] > z_under - 0.5]
    base_pts = points[points[:, 2] < z_min + 2.0]
    base = points[points[:, 2] < z_min + 15.0]
    bfaces = sorted(_horizontal_faces(base[:, 2]))
    above = [f for f in bfaces if f > z_min + 2.0]
    z_base_top = above[0] if above else z_min + 8.0
    return ShelfGeometry(
        z_min, z_top, z_under, z_base_top,
        (float(plat[:, 0].min()), float(plat[:, 0].max())), (float(plat[:, 1].min()), float(plat[:, 1].max())),
        (float(base_pts[:, 0].min()), float(base_pts[:, 0].max())), (float(base_pts[:, 1].min()), float(base_pts[:, 1].max())),
    )


def build_shelf_problem(
    dims_path: str | Path,
    points_path: str | Path,
    element_size: float = 3.5,
    volume_fraction: float = 0.15,
    safety_factor: float = 2.5,
    payload_mass_kg: float | None = None,
) -> tuple[TOProblem, dict]:
    dims = parse_dimensions(dims_path)
    points = load_point_cloud(points_path)
    points_path = str(Path(points_path).resolve())
    g = measure_shelf(points, dims)
    h = element_size
    pad = h + 1.0
    lo, hi = points.min(0), points.max(0)
    mass = payload_mass_kg if payload_mass_kg is not None else 0.5  # a desk stapler; not stated in the concept
    near = NearPointsRegion(path=points_path, tol=h)
    px, py = g.platform_x, g.platform_y

    problem = TOProblem(
        material=printed_material("PLA"),
        design_domain=BoxRegion(
            min=(float(lo[0] - pad), float(lo[1] - pad), g.z_min - 0.5),
            max=(float(hi[0] + pad), float(hi[1] + pad), g.z_top + 0.5),  # nothing above the flat top
        ),
        warm_start=[near],
        preserve=[
            BoxRegion(min=(px[0] - 0.5, py[0] - 0.5, g.z_under - 0.5), max=(px[1] + 0.5, py[1] + 0.5, g.z_top + 1.0)),  # platform slab
            IntersectionRegion(regions=[near, BoxRegion(min=(g.base_x[0] - 1, g.base_y[0] - 1, g.z_min - 1), max=(g.base_x[1] + 1, g.base_y[1] + 1, g.z_min + 3.0))]),  # foot skin
        ],
        void=[],
        supports=[
            Support(id="ground", region=BoxRegion(min=(g.base_x[0], g.base_y[0], g.z_min - 1.0), max=(g.base_x[1], g.base_y[1], g.z_min + 1.5)), confidence=ASSUMED),
        ],
        load_cases=[
            LoadCase(id="static_gravity", region=BoxRegion(min=(px[0] + 5, py[0] + 5, g.z_top - h), max=(px[1] - 5, py[1] - 5, g.z_top + 1)), force_N=(0.0, 0.0, -9.81 * mass), confidence=ASSUMED),
            LoadCase(id="stapling_press", region=BoxRegion(min=(px[1] - 30.0, py[0] + 10, g.z_top - h), max=(px[1] - 4.0, py[1] - 10, g.z_top + 1)), force_N=(0.0, 0.0, -30.0), weight=1.0, confidence=ASSUMED),
        ],
        safety_factor=safety_factor,
        volume_fraction=volume_fraction,
        target_element_size=h,
        filter_radius=1.5 * h,
        max_iters=50,
        notes=f"Demo stapler shelf. Payload {mass} kg (assumed) + 30 N stapling press (assumed). Levels measured from the scan: {g}.",
    )
    return problem, {"shelf": asdict(g), "dims": dims, "payload_mass_kg": mass}
