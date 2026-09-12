"""Point-membership tests and bounds for every Region primitive."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import contracts as c

_AXIS = {"x": 0, "y": 1, "z": 2}


@lru_cache(maxsize=None)
def _points(path: str) -> np.ndarray:
    from .ingest.point_cloud import load_point_cloud

    return load_point_cloud(path)


@lru_cache(maxsize=None)
def _tree(path: str) -> cKDTree:
    return cKDTree(_points(path))


@lru_cache(maxsize=None)
def _mesh(path: str) -> trimesh.Trimesh:
    from .ingest.mesh import load_mesh

    return load_mesh(path)


def contains(region, pts: np.ndarray) -> np.ndarray:
    """Boolean mask of which points (N,3) lie in `region`."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    t = region.type
    if t == "box":
        lo, hi = np.asarray(region.min), np.asarray(region.max)
        return np.all((pts >= lo) & (pts <= hi), axis=1)
    if t == "cylinder":
        ax = _AXIS[region.axis]
        tr = [i for i in range(3) if i != ax]
        d = pts[:, tr] - np.asarray(region.center)[tr]
        r = np.linalg.norm(d, axis=1)
        a = pts[:, ax]
        return (r >= region.r_min) & (r <= region.r_max) & (a >= region.along[0]) & (a <= region.along[1])
    if t == "sphere":
        return np.linalg.norm(pts - np.asarray(region.center), axis=1) <= region.radius
    if t == "halfspace":
        return (pts - np.asarray(region.point)) @ np.asarray(region.normal) >= 0.0
    if t == "near_points":
        d, _ = _tree(region.path).query(pts, distance_upper_bound=region.tol)
        return np.isfinite(d)
    if t == "inside_mesh":
        m = _mesh(region.path)
        if not m.is_watertight:
            raise ValueError(
                f"inside_mesh requires a watertight mesh; {region.path} is not. "
                "Use near_mesh with a tolerance instead, or repair the mesh."
            )
        return m.contains(pts)
    if t == "near_mesh":
        m = _mesh(region.path)
        _, dist, _ = trimesh.proximity.closest_point(m, pts)
        return dist <= region.tol
    if t == "union":
        return contains_any(region.regions, pts)
    if t == "intersection":
        if not region.regions:
            return np.zeros(len(pts), dtype=bool)
        return np.all([contains(r, pts) for r in region.regions], axis=0)
    if t == "difference":
        return contains(region.a, pts) & ~contains(region.b, pts)
    raise ValueError(f"unknown region type {t!r}")


def contains_any(regions: Sequence, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    if not regions:
        return np.zeros(len(pts), dtype=bool)
    return np.any([contains(r, pts) for r in regions], axis=0)


def bounds(region) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned bounds (lo, hi) of a region, or None if unbounded (halfspace)."""
    t = region.type
    if t == "box":
        return np.asarray(region.min, float), np.asarray(region.max, float)
    if t == "cylinder":
        ax = _AXIS[region.axis]
        lo = np.asarray(region.center, float) - region.r_max
        hi = np.asarray(region.center, float) + region.r_max
        lo[ax], hi[ax] = region.along
        return lo, hi
    if t == "sphere":
        cen = np.asarray(region.center, float)
        return cen - region.radius, cen + region.radius
    if t == "halfspace":
        return None
    if t == "near_points":
        p = _points(region.path)
        return p.min(0) - region.tol, p.max(0) + region.tol
    if t in ("inside_mesh", "near_mesh"):
        b = _mesh(region.path).bounds
        tol = getattr(region, "tol", 0.0)
        return b[0] - tol, b[1] + tol
    if t in ("union", "intersection"):
        bs = [bounds(r) for r in region.regions]
        bs = [b for b in bs if b is not None]
        if not bs:
            return None
        los, his = zip(*bs)
        if t == "union":
            return np.min(los, axis=0), np.max(his, axis=0)
        return np.max(los, axis=0), np.min(his, axis=0)
    if t == "difference":
        return bounds(region.a)
    raise ValueError(f"unknown region type {t!r}")


def bounds_of(regions: Sequence) -> tuple[np.ndarray, np.ndarray] | None:
    bs = [bounds(r) for r in regions]
    bs = [b for b in bs if b is not None]
    if not bs:
        return None
    los, his = zip(*bs)
    return np.min(los, axis=0), np.max(his, axis=0)


def resolve_domain(problem: c.TOProblem) -> c.BoxRegion:
    """The design domain box: explicit, or padded bounds of the warm-start regions."""
    if problem.design_domain is not None:
        return problem.design_domain
    b = bounds_of(problem.warm_start)
    if b is None:
        raise ValueError("cannot infer design_domain: warm_start regions are unbounded")
    lo, hi = b[0] - problem.domain_padding, b[1] + problem.domain_padding
    return c.BoxRegion(min=tuple(lo.tolist()), max=tuple(hi.tolist()))
