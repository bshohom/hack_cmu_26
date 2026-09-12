"""Export density fields as VTK image data and as an STL isosurface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from ..meshing.voxel_backend import HexMesh


def rho_to_grid(mesh: HexMesh, rho: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) array of element densities."""
    grid = np.zeros(mesh.shape, dtype=float)
    i, j, k = mesh.elem_ijk.T
    grid[i, j, k] = rho
    return grid


def nodal_to_grid(mesh: HexMesh, values: np.ndarray) -> np.ndarray:
    """(nx+1, ny+1, nz+1, c) array of nodal values."""
    shape = tuple(int(n) + 1 for n in mesh.shape)
    out = np.zeros((*shape, values.shape[1]), dtype=float)
    i, j, k = mesh.node_ijk.T
    out[i, j, k] = values
    return out


def image_data(mesh: HexMesh, rho: np.ndarray, pad: int = 0, fill: float = 0.0) -> pv.ImageData:
    """pyvista ImageData with cell data 'rho' (optionally padded by `pad` cells of `fill`)."""
    grid = rho_to_grid(mesh, rho)
    if pad:
        grid = np.pad(grid, pad, constant_values=fill)
    dims = tuple(int(n) + 1 for n in grid.shape)
    origin = mesh.origin - pad * mesh.spacing
    img = pv.ImageData(dimensions=dims, spacing=tuple(mesh.spacing), origin=tuple(origin))
    img.cell_data["rho"] = grid.ravel(order="F")
    return img


def save_vti(mesh: HexMesh, rho: np.ndarray, path: str | Path, u: np.ndarray | None = None) -> Path:
    img = image_data(mesh, rho)
    if u is not None:
        img.point_data["u"] = nodal_to_grid(mesh, u).reshape(-1, 3, order="F")
        img.point_data["u_mag"] = np.linalg.norm(img.point_data["u"], axis=1)
    path = Path(path)
    img.save(path)
    return path


def isosurface(mesh: HexMesh, rho: np.ndarray, threshold: float = 0.5) -> pv.PolyData:
    """Closed isosurface of the density field (padded so boundaries close)."""
    img = image_data(mesh, rho, pad=1, fill=0.0)
    return img.cell_data_to_point_data().contour([threshold], scalars="rho")


def save_stl(mesh: HexMesh, rho: np.ndarray, path: str | Path, threshold: float = 0.5) -> dict:
    surf = isosurface(mesh, rho, threshold)
    path = Path(path)
    if surf.n_points == 0:
        raise ValueError(f"no material above rho={threshold}; nothing to export")
    surf = surf.triangulate().clean()
    surf.save(path)
    info = {"path": str(path), "n_faces": int(surf.n_cells), "n_points": int(surf.n_points)}
    try:
        import trimesh

        tm = trimesh.Trimesh(np.asarray(surf.points), surf.faces.reshape(-1, 4)[:, 1:], process=True)
        info["watertight"] = bool(tm.is_watertight)
        info["components"] = int(len(tm.split(only_watertight=False)))
        if tm.is_watertight:
            info["volume_mm3"] = float(abs(tm.volume))
    except Exception as exc:  # trimesh is optional here
        info["check_error"] = str(exc)
    return info
