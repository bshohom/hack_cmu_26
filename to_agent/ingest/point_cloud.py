"""Point-cloud loading and simple geometric fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load an (N,3) point array from OBJ (vertex-only or meshed), PLY, XYZ/CSV/TXT."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"point cloud file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".obj":
        pts = [
            line.split()[1:4]
            for line in path.read_text().splitlines()
            if line.startswith("v ")
        ]
        if not pts:
            raise ValueError(f"no 'v x y z' lines in {path}")
        return np.asarray(pts, dtype=float)
    if suffix in (".xyz", ".txt", ".csv"):
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)[:, :3]
    import trimesh

    obj = trimesh.load(path)
    return np.asarray(obj.vertices, dtype=float)


def bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit. Returns (cx, cy, r)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    (cx2, cy2, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = cx2 / 2, cy2 / 2
    r = float(np.sqrt(c + cx**2 + cy**2))
    return float(cx), float(cy), r


def radius_histogram(xy: np.ndarray, center: tuple[float, float], bin_mm: float = 0.5):
    """Histogram of radial distances from `center`; useful to find shell radii."""
    r = np.linalg.norm(xy - np.asarray(center), axis=1)
    edges = np.arange(0.0, r.max() + bin_mm, bin_mm)
    counts, _ = np.histogram(r, bins=edges)
    return edges[:-1], counts
