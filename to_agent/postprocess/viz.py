"""Convergence plots and a rendered view of the optimized topology."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..meshing.voxel_backend import HexMesh  # noqa: E402
from .export import isosurface, rho_to_grid  # noqa: E402


def save_history_png(compliance: list[float], volume: list[float], path: str | Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(compliance, marker="o", ms=3)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("compliance (N·mm)")
    ax[0].set_yscale("log")
    ax[0].grid(alpha=0.3)
    ax[1].plot(volume, marker="o", ms=3, color="tab:orange")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("volume fraction (design)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _slices_png(mesh: HexMesh, rho: np.ndarray, path: Path) -> Path:
    grid = rho_to_grid(mesh, rho)
    nx, ny, nz = grid.shape
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        (grid[:, ny // 2, :].T, "xz (mid y)", "x", "z"),
        (grid[nx // 2, :, :].T, "yz (mid x)", "y", "z"),
        (grid[:, :, nz // 2].T, "xy (mid z)", "x", "y"),
    ]
    for a, (img, title, xl, yl) in zip(ax, panels):
        a.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        a.set_title(title)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_render_png(mesh: HexMesh, rho: np.ndarray, path: str | Path, threshold: float = 0.5) -> tuple[Path, str]:
    """Off-screen pyvista render of the isosurface; falls back to matplotlib slices."""
    path = Path(path)
    try:
        import pyvista as pv

        surf = isosurface(mesh, rho, threshold)
        if surf.n_points == 0:
            raise ValueError("empty isosurface")
        pl = pv.Plotter(off_screen=True, window_size=(1000, 750))
        pl.add_mesh(surf, color="lightsteelblue", smooth_shading=False)
        pl.add_axes()
        pl.show_bounds(grid=False, location="outer")
        pl.view_isometric()
        pl.screenshot(str(path))
        pl.close()
        return path, "pyvista"
    except Exception as exc:  # no OpenGL context, etc.
        _slices_png(mesh, rho, path)
        return path, f"matplotlib slices (pyvista failed: {type(exc).__name__}: {exc})"
