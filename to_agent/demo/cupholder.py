"""Demo: build a TOProblem for the desk-clamp cupholder from its dimension file + point cloud.

Everything part-specific lives here. The output is an ordinary TOProblem made of generic
region primitives; the solver never sees this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import (
    BoxRegion,
    CylinderRegion,
    LoadCase,
    Material,
    NearPointsRegion,
    Support,
    TOProblem,
)
from ..ingest.dimensions import parse_dimensions
from ..ingest.point_cloud import fit_circle, load_point_cloud

ASSUMED = "assumed"


@dataclass
class ClampGeometry:
    """Levels of the clamp measured from the point cloud (mm)."""

    x_far: float  # outer end of the clamp
    x_spine: float  # vertical member's outer face (desk slot starts here)
    plate_top: float
    plate_bottom: float  # top jaw contact plane
    hook_top: float  # bottom jaw contact plane
    hook_bottom: float
    measured_gap: float


def _horizontal_faces(z: np.ndarray, bin_mm: float = 1.0, factor: float = 3.0) -> list[float]:
    """z-levels of horizontal faces = histogram bins well above the median count."""
    edges = np.arange(z.min(), z.max() + bin_mm, bin_mm)
    counts, _ = np.histogram(z, bins=edges)
    strong = counts > factor * max(np.median(counts), 1)
    faces: list[float] = []
    i = 0
    while i < len(strong):
        if strong[i]:
            j = i
            while j + 1 < len(strong) and strong[j + 1]:
                j += 1
            sel = (z >= edges[i]) & (z < edges[j + 1])
            faces.append(float(np.median(z[sel])))
            i = j + 1
        else:
            i += 1
    return faces


def measure_clamp(points: np.ndarray, r_out: float, desk_gap: float, hook_t: float) -> ClampGeometry:
    arm = points[points[:, 0] > r_out + 1.0]
    x_far = float(arm[:, 0].max())
    far = arm[arm[:, 0] > x_far - 20.0]  # far column: plate + hook faces, no neck
    faces = sorted(_horizontal_faces(far[:, 2]))
    if len(faces) < 3:
        raise ValueError(f"expected >=3 horizontal faces in clamp region, found {faces}")
    plate_top = faces[-1]
    below = [f for f in faces if f < plate_top - 4.0]
    plate_bottom = below[-1]
    target_hook_top = plate_bottom - desk_gap
    hook_top = min((f for f in faces if f < plate_bottom - 4.0), key=lambda f: abs(f - target_hook_top))
    hook_bottom = hook_top - hook_t
    # vertical member: widest x reached in the middle of the slot band, excluding the far lip
    mid = arm[(arm[:, 2] > hook_top + 0.4 * desk_gap) & (arm[:, 2] < plate_bottom - 4.0) & (arm[:, 0] < x_far - 20.0)]
    x_spine = float(mid[:, 0].max())
    return ClampGeometry(x_far, x_spine, plate_top, plate_bottom, hook_top, hook_bottom, plate_bottom - hook_top)


def validate_scan(points: np.ndarray, r_in: float, r_out: float, tol_mm: float = 1.5) -> dict:
    """Cross-check the point cloud against the dimension file (cup centre and radii)."""
    body = points[np.abs(points[:, 0]) < r_out + 5.0]
    r = np.linalg.norm(body[:, :2], axis=1)
    ring = body[(r > r_in - 3.0) & (r < r_out + 3.0)]
    cx, cy, _ = fit_circle(ring[:, :2])
    r_ring = np.linalg.norm(ring[:, :2] - [cx, cy], axis=1)
    hist, edges = np.histogram(r_ring, bins=np.arange(r_in - 3.0, r_out + 3.5, 0.5))
    order = np.argsort(hist)[::-1]
    peaks = sorted(float(edges[i] + 0.25) for i in order[:6])
    r_in_found = min(peaks, key=lambda p: abs(p - r_in))
    r_out_found = min(peaks, key=lambda p: abs(p - r_out))
    ok = abs(cx) < tol_mm and abs(cy) < tol_mm and abs(r_in_found - r_in) < tol_mm and abs(r_out_found - r_out) < tol_mm
    return {
        "center_xy": [round(cx, 2), round(cy, 2)],
        "r_in_found": r_in_found,
        "r_out_found": r_out_found,
        "r_in_spec": r_in,
        "r_out_spec": r_out,
        "consistent": bool(ok),
    }


def build_cupholder_problem(
    dims_path: str | Path,
    points_path: str | Path,
    element_size: float = 4.0,
    volume_fraction: float = 0.2,
    safety_factor: float = 2.5,
) -> tuple[TOProblem, dict]:
    dims = parse_dimensions(dims_path)
    points = load_point_cloud(points_path)
    points_path = str(Path(points_path).resolve())

    r_in = dims["inner_diameter"] / 2.0
    r_out = dims["outer_diameter"] / 2.0
    base_t = dims["base_thickness"]
    cup_top = base_t + dims["holder_height"]
    half_w = dims["arm_width"] / 2.0
    desk_gap = dims["nominal_desk_gap"]
    hook_t = dims["lower_hook_thickness"]

    scan = validate_scan(points, r_in, r_out)
    clamp = measure_clamp(points, r_out, desk_gap, hook_t)
    h = element_size
    lo, hi = points.min(0), points.max(0)
    pad = h + 1.0

    problem = TOProblem(
        material=Material(name="PLA", E_MPa=2300.0, nu=0.35, density_kg_m3=1240.0, yield_MPa=50.0, confidence=ASSUMED),
        design_domain=BoxRegion(
            min=(float(lo[0] - pad), float(lo[1] - pad), float(lo[2] - pad)),
            max=(float(hi[0] + pad), float(hi[1] + pad), float(hi[2] + pad)),
        ),
        warm_start=[NearPointsRegion(path=points_path, tol=h)],
        preserve=[
            CylinderRegion(center=(0, 0, 0), r_min=r_in, r_max=r_out, along=(0.0, cup_top)),  # cup wall
            CylinderRegion(center=(0, 0, 0), r_min=0.0, r_max=r_out, along=(0.0, base_t)),  # cup floor
            BoxRegion(  # top jaw: plate material right above the desk
                min=(clamp.x_spine, -half_w, clamp.plate_bottom),
                max=(clamp.x_far, half_w, clamp.plate_bottom + h + 1.0),
            ),
            BoxRegion(  # bottom jaw: hook material right below the desk
                min=(clamp.x_spine, -half_w, clamp.hook_top - h - 1.0),
                max=(clamp.x_far, half_w, clamp.hook_top),
            ),
        ],
        void=[
            CylinderRegion(center=(0, 0, 0), r_min=0.0, r_max=r_in, along=(base_t + 1.0, hi[2] + pad)),  # cup cavity
            BoxRegion(  # the desk itself
                min=(clamp.x_spine, float(lo[1] - pad), clamp.hook_top),
                max=(float(hi[0] + pad), float(hi[1] + pad), clamp.plate_bottom),
            ),
        ],
        supports=[
            Support(
                id="top_jaw",
                region=BoxRegion(
                    min=(clamp.x_spine + 2.0, -half_w, clamp.plate_bottom),
                    max=(clamp.x_far, half_w, clamp.plate_bottom + h),
                ),
                confidence=ASSUMED,
            ),
            Support(
                id="bottom_jaw",
                region=BoxRegion(
                    min=(clamp.x_spine + 2.0, -half_w, clamp.hook_top - h),
                    max=(clamp.x_far, half_w, clamp.hook_top),
                ),
                confidence=ASSUMED,
            ),
        ],
        load_cases=[
            LoadCase(  # ~1 kg drink resting on the cup floor
                id="drink_weight",
                region=CylinderRegion(center=(0, 0, 0), r_min=0.0, r_max=r_in, along=(0.0, base_t + h)),
                force_N=(0.0, 0.0, -10.0),
                confidence=ASSUMED,
            ),
            LoadCase(  # lateral knock on the rim, away from the clamp
                id="rim_bump",
                region=CylinderRegion(center=(0, 0, 0), r_min=r_in - 1.0, r_max=r_out + 1.0, along=(cup_top - h, cup_top + 1.0)),
                force_N=(30.0, 0.0, 0.0),
                confidence=ASSUMED,
            ),
        ],
        safety_factor=safety_factor,
        volume_fraction=volume_fraction,
        target_element_size=h,
        filter_radius=1.5 * h,
        max_iters=60,
        notes=(
            "Demo cupholder. Loads and material are ASSUMED placeholders. "
            f"Clamp levels measured from the scan: {clamp}."
        ),
    )
    report = {"scan_check": scan, "clamp": clamp.__dict__, "dims": dims}
    return problem, report
