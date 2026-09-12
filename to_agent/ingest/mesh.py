"""Triangle-mesh loading (warm starts, attachment-point STLs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mesh file not found: {path}")
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(
            f"{path} has no faces (is it a point cloud? use near_points instead of a mesh region)"
        )
    return mesh


def describe_mesh(mesh: trimesh.Trimesh) -> dict:
    lo, hi = mesh.bounds
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
        "bounds_min": np.round(lo, 2).tolist(),
        "bounds_max": np.round(hi, 2).tolist(),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }
