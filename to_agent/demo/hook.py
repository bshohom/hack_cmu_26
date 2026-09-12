"""Demo: desk bag hook (5 kg at 100 mm reach) — TOProblem from its dimension file + point cloud.

Frame (as generated): desk front edge at x ≈ 0 with the clamp arms at x < 0, hook toward +x,
desk underside at z ≈ 0 (bottom arm top face), +Z up.
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
class HookGeometry:
    x_min: float
    x_max: float
    x_back_in: float  # desk-facing face of the back plate (desk front edge)
    x_back_out: float
    z_bot_arm: tuple[float, float]
    z_top_arm: tuple[float, float]
    z_rib_lo: float  # top of the lower contact ribs (desk underside)
    z_rib_hi: float  # bottom of the upper contact ribs (desk top)
    z_seat: float  # top face of the horizontal hook arm (strap seat)
    x_seat: tuple[float, float]
    x_tip: tuple[float, float]
    z_tip_top: float
    half_w: float
    hook_half_w: float
    measured_gap: float


def measure_hook(points: np.ndarray, dims: dict) -> HookGeometry:
    t_arm = float(dims.get("top_bottom_arm_thickness", 10.0))
    half_w = float(dims.get("clamp_back_plate_overall_width", 50.0)) / 2.0
    hook_half_w = float(dims.get("hook_arm_width", 20.0)) / 2.0
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())

    arms = points[points[:, 0] < x_min + 0.6 * abs(x_min)]  # clamp arms, clear of the back plate
    z_lo, z_hi = float(arms[:, 2].min()), float(arms[:, 2].max())
    z_bot_arm = (z_lo, z_lo + t_arm)
    z_top_arm = (z_hi - t_arm, z_hi)
    gap = arms[(arms[:, 2] > z_bot_arm[1] + 0.3) & (arms[:, 2] < z_top_arm[0] - 0.3)]
    mid = 0.5 * (z_bot_arm[1] + z_top_arm[0])
    lower = gap[gap[:, 2] < mid]
    upper = gap[gap[:, 2] > mid]
    z_rib_lo = float(lower[:, 2].max()) if len(lower) else z_bot_arm[1]
    z_rib_hi = float(upper[:, 2].min()) if len(upper) else z_top_arm[0]

    plate = points[(np.abs(points[:, 1]) > half_w - 4.0) & (points[:, 2] > z_rib_lo + 1.0) & (points[:, 2] < z_rib_hi - 1.0)]
    plate = plate[plate[:, 0] > x_min + 0.6 * abs(x_min)]
    x_back_in, x_back_out = float(plate[:, 0].min()), float(plate[:, 0].max())

    tip = points[(points[:, 2] > z_top_arm[0]) & (points[:, 0] > x_back_out + 20.0)]
    x_tip = (float(tip[:, 0].min()), x_max)
    z_tip_top = float(tip[:, 2].max())

    seat_pts = points[(points[:, 0] > x_back_out + 30.0) & (points[:, 0] < x_tip[0] - 3.0) & (np.abs(points[:, 1]) < hook_half_w)]
    faces = sorted(_horizontal_faces(seat_pts[:, 2]))
    z_seat = faces[-1] if faces else float(seat_pts[:, 2].max())
    # The seat runs all the way to the tip: the strap bears on the whole horizontal arm and
    # is retained by the tip, so seat and tip must stay one connected body.
    x_seat = (x_back_out + 25.0, x_max)
    return HookGeometry(
        x_min, x_max, x_back_in, x_back_out, z_bot_arm, z_top_arm, z_rib_lo, z_rib_hi,
        z_seat, x_seat, x_tip, z_tip_top, half_w, hook_half_w, z_rib_hi - z_rib_lo,
    )


def desk_edge_to_mesh_transform(geometry: HookGeometry) -> list[list[float]]:
    """4x4 taking desk_edge_frame points into this mesh's candidate_mesh_frame.

    Both frames already share axes and units (mm, +X off the desk, +Y along the
    edge, +Z up). The mesh origin is the print/scan origin. The generator measures
    two named planes that desk_edge_frame defines as x=0 and z=0:

    - desk front edge  -> x = geometry.x_back_in
    - desk underside   -> z = geometry.z_rib_lo

    The translation is those two plane offsets. It is not a fit to any load point.
    """
    return [
        [1.0, 0.0, 0.0, float(geometry.x_back_in)],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, float(geometry.z_rib_lo)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def hook_registration(dims_path: str | Path, points_path: str | Path) -> dict:
    """Canned-demo registration payload: transform + landmarks in both frames."""
    geometry = measure_hook(load_point_cloud(points_path), parse_dimensions(dims_path))
    matrix = desk_edge_to_mesh_transform(geometry)
    tx, tz = float(geometry.x_back_in), float(geometry.z_rib_lo)

    def to_desk(x: float, y: float, z: float) -> list[float]:
        return [x - tx, y, z - tz]

    seat_x = 0.5 * (geometry.x_seat[0] + geometry.x_seat[1])
    named = {
        "strap_seat": to_desk(seat_x, 0.0, geometry.z_seat),
        "mount_contact": to_desk(
            0.5 * (geometry.x_min + geometry.x_back_in),
            0.0,
            0.5 * (geometry.z_rib_lo + geometry.z_rib_hi),
        ),
    }
    return {
        "from_frame": "desk_edge_frame",
        "to_frame": "candidate_mesh_frame",
        "units": "mm",
        "matrix": matrix,
        "basis": (
            "translation aligning the measured desk front-edge plane "
            f"(x={tx:.4f}) and desk-underside plane (z={tz:.4f}) with "
            "desk_edge_frame x=0 / z=0; axes already coincide"
        ),
        "landmarks_desk_edge_frame": named,
        "named_regions": named,
    }


def build_hook_problem(
    dims_path: str | Path,
    points_path: str | Path,
    element_size: float = 3.0,
    volume_fraction: float = 0.2,
    safety_factor: float = 2.5,
    payload_mass_kg: float | None = None,
) -> tuple[TOProblem, dict]:
    dims = parse_dimensions(dims_path)
    points = load_point_cloud(points_path)
    points_path = str(Path(points_path).resolve())
    g = measure_hook(points, dims)
    h = element_size
    pad = h + 1.0
    lo, hi = points.min(0), points.max(0)
    mass = payload_mass_kg if payload_mass_kg is not None else float(dims.get("nominal_concept_target_load", 5.0))
    near = NearPointsRegion(path=points_path, tol=h)

    def solid(box: BoxRegion) -> IntersectionRegion:
        return IntersectionRegion(regions=[near, box])

    problem = TOProblem(
        material=printed_material("PLA"),
        design_domain=BoxRegion(min=tuple(float(v) for v in lo - pad), max=tuple(float(v) for v in hi + pad)),
        warm_start=[near],
        preserve=[
            solid(BoxRegion(min=(g.x_min - 1, -g.half_w - 1, g.z_rib_hi - 3.0), max=(g.x_back_in, g.half_w + 1, g.z_top_arm[0] + h))),  # top jaw
            solid(BoxRegion(min=(g.x_min - 1, -g.half_w - 1, g.z_bot_arm[1] - h), max=(g.x_back_in, g.half_w + 1, g.z_rib_lo + 3.0))),  # bottom jaw
            solid(BoxRegion(min=(g.x_tip[0] - 1, -g.hook_half_w - 1, g.z_bot_arm[0] - 1), max=(g.x_tip[1] + 1, g.hook_half_w + 1, g.z_tip_top + 1))),  # hook tip
            solid(BoxRegion(min=(g.x_seat[0], -g.hook_half_w - 1, g.z_seat - h - 1), max=(g.x_seat[1], g.hook_half_w + 1, g.z_seat + 0.5))),  # strap seat
        ],
        void=[
            BoxRegion(min=(float(lo[0] - pad), float(lo[1] - pad), g.z_rib_lo), max=(g.x_back_in, float(hi[1] + pad), g.z_rib_hi)),  # the desk
            BoxRegion(min=(g.x_seat[0], -g.hook_half_w - 3.0, g.z_seat + 1.0), max=(g.x_tip[0] - 1.0, g.hook_half_w + 3.0, g.z_tip_top - 1.0)),  # strap opening
        ],
        supports=[
            Support(id="top_jaw", region=BoxRegion(min=(g.x_min + 2, -g.half_w, g.z_rib_hi - 0.5), max=(g.x_back_in - 2, g.half_w, g.z_rib_hi + h)), confidence=ASSUMED),
            Support(id="bottom_jaw", region=BoxRegion(min=(g.x_min + 2, -g.half_w, g.z_rib_lo - h), max=(g.x_back_in - 2, g.half_w, g.z_rib_lo + 0.5)), confidence=ASSUMED),
        ],
        load_cases=[
            # Strap weight, distributed over the whole horizontal arm it rests on.
            LoadCase(id="static_gravity", region=BoxRegion(min=(g.x_seat[0], -g.hook_half_w, g.z_seat - h), max=(g.x_seat[1], g.hook_half_w, g.z_seat + 1.0)), force_N=(0.0, 0.0, -9.81 * mass), confidence=ASSUMED),
            # Strap pulling outward against the tip: this is what makes the tip load-bearing
            # (preserved-but-unloaded geometry gets disconnected by the optimizer).
            LoadCase(id="tip_retention", region=BoxRegion(min=(g.x_tip[0] - 1.0, -g.hook_half_w, g.z_seat - h), max=(g.x_tip[1], g.hook_half_w, g.z_tip_top)), force_N=(0.4 * 9.81 * mass, 0.0, 0.0), weight=1.0, confidence=ASSUMED),
            LoadCase(id="side_swing", region=BoxRegion(min=(g.x_seat[0], -g.hook_half_w, g.z_seat - h), max=(g.x_seat[1], g.hook_half_w, g.z_seat + 1.0)), force_N=(0.0, 0.3 * 9.81 * mass, 0.0), weight=0.5, confidence=ASSUMED),
        ],
        safety_factor=safety_factor,
        volume_fraction=volume_fraction,
        target_element_size=h,
        filter_radius=1.5 * h,
        max_iters=50,
        notes=f"Demo desk bag hook. Payload {mass} kg (concept target). Clamp levels measured from the scan: {g}.",
    )
    return problem, {"hook": asdict(g), "dims": dims, "payload_mass_kg": mass}
