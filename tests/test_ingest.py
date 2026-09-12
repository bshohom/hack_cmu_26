from pathlib import Path

import numpy as np

from to_agent.ingest.dimensions import parse_dimensions
from to_agent.ingest.point_cloud import bbox, fit_circle, load_point_cloud

ROOT = Path(__file__).resolve().parent.parent


def test_parse_dimensions():
    d = parse_dimensions(ROOT / "cupholder_dimensions.txt")
    assert d["inner_diameter"] == 70.0
    assert d["compatible_desk_range_from_concept"] == (20.0, 35.0)
    assert d["units"] == "mm"


def test_load_point_cloud_and_fit():
    pts = load_point_cloud(ROOT / "cupholder_surface_particles.obj")
    assert pts.shape == (50000, 3)
    lo, hi = bbox(pts)
    assert lo[0] < -40 and hi[0] > 100
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circ = np.column_stack([3 + 7 * np.cos(theta), -2 + 7 * np.sin(theta)])
    cx, cy, r = fit_circle(circ)
    assert abs(cx - 3) < 1e-6 and abs(cy + 2) < 1e-6 and abs(r - 7) < 1e-6
