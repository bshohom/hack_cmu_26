"""Reduce a density field to one connected body, and check it is actually usable.

A SIMP result can converge with the load path split into several pieces (the saved desk-hook
run shipped a watertight STL made of two disconnected bodies). Two things were wrong with
that: the STL was unprintable as a single part, and the post-check still reported a healthy
factor of safety, because it solves the voxel grid where "removed" elements keep a small
residual stiffness (VOID_STIFFNESS) that quietly bridges the gap.

Trimming the *density field* rather than the exported mesh fixes both at once: the STL and
the FE model are then the same body. Trimming is only safe if the surviving body still
carries every support and every load, so that is checked rather than assumed.
"""

from __future__ import annotations

import numpy as np

from ..meshing.masks import Masks
from ..meshing.voxel_backend import HexMesh
from .export import rho_to_grid


def keep_largest_component(
    mesh: HexMesh, rho: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, dict]:
    """Zero every solid element outside the largest body of the *exported* isosurface.

    Connectivity has to be judged on the same representation that is shipped. Labelling the
    thresholded voxels is not equivalent: the isosurface is contoured from cell data
    interpolated to points, so two voxel regions that share a face can still contour into
    separate shells where the interpolated field dips below the threshold. The saved
    desk-hook run is exactly that case — one 6-connected voxel body, two STL bodies.

    So the surface is contoured first, its largest connected shell taken, and the density
    field trimmed to the elements inside it. Both the STL and the post-check FE then
    describe that one body.

    Returns (rho, info); `rho` is unchanged when the surface is already a single body.
    """
    from .export import isosurface

    info: dict = {"components_before": None, "trimmed": False}
    surf = isosurface(mesh, rho, threshold)
    if surf.n_points == 0:
        return rho, info

    split = surf.connectivity("all")
    region_ids = np.asarray(split.point_data["RegionId"])
    n = int(region_ids.max()) + 1 if region_ids.size else 0
    info["components_before"] = n
    if n <= 1:
        return rho, info

    largest = split.connectivity("largest").triangulate().clean()
    try:
        import trimesh

        tm = trimesh.Trimesh(
            np.asarray(largest.points), largest.faces.reshape(-1, 4)[:, 1:], process=True
        )
        inside = tm.contains(mesh.centroids)
    except Exception as exc:  # noqa: BLE001 — without containment we cannot trim safely
        info["trim_error"] = f"{type(exc).__name__}: {exc}"
        return rho, info

    solid = rho >= threshold
    drop = solid & ~inside
    if not drop.any():
        return rho, info
    out = rho.copy()
    out[drop] = 0.0
    info.update(
        trimmed=True,
        kept_elements=int((solid & inside).sum()),
        removed_elements=int(drop.sum()),
        removed_fraction=float(drop.sum() / max(solid.sum(), 1)),
    )
    return out, info


def carries_boundary_conditions(
    mesh: HexMesh,
    rho: np.ndarray,
    masks: Masks,
    load_case_ids: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """Does solid material actually reach the supports and every load region?

    A body that is single and watertight but has shed a mount is not a usable part. Node
    membership is taken through each solid element's nodes, matching how the solver applies
    constraints and forces.
    """
    solid = rho >= threshold
    reached = np.zeros(mesh.n_nodes, dtype=bool)
    if solid.any():
        reached[np.unique(mesh.elements.numpy()[solid])] = True

    constrained = masks.constraints.any(dim=1).numpy()
    supports_ok = bool((reached & constrained).any())

    detached: list[str] = []
    for i, F in enumerate(masks.forces):
        loaded = (F.abs().sum(dim=1) > 0).numpy()
        if not bool((reached & loaded).any()):
            name = load_case_ids[i] if load_case_ids and i < len(load_case_ids) else f"case_{i}"
            detached.append(name)
    return {
        "supports_attached": supports_ok,
        "loads_attached": not detached,
        "detached_loads": detached,
    }
