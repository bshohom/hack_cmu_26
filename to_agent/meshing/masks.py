"""Element and node masks from a TOProblem on a HexMesh."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..contracts import TOProblem
from ..regions import bounds, contains, contains_any
from .voxel_backend import HexMesh

_AXIS = {"x": 0, "y": 1, "z": 2}


class ProblemSetupError(ValueError):
    """Raised with an agent-readable message when a problem cannot be set up on the grid."""


def select_nodes_for_region(
    region,
    nodes: np.ndarray,
    h: float,
    solid_centroids: np.ndarray | None = None,
) -> np.ndarray:
    """Select mesh nodes for a physical support/load region.

    Exact occupancy first. If a zero-thickness face lies between grid planes,
    snap to the nearest nodes within one element spacing that still project onto
    the region's tangential footprint. Does not relocate or enlarge the physical
    support. Returns an all-false mask when the mesh cannot resolve the region.
    """
    pts = np.asarray(nodes, dtype=float).reshape(-1, 3)
    exact = contains(region, pts)
    if exact.any():
        return _touching_solid(exact, pts, solid_centroids, h)
    box = bounds(region)
    if box is None or h <= 0:
        return exact
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    thickness = np.maximum(hi - lo, 0.0)
    pad = np.where(thickness < h, 0.5 * h, 0.0)
    if not np.any(pad > 0):
        return exact
    snapped = np.all((pts >= lo - pad) & (pts <= hi + pad), axis=1)
    for axis in range(3):
        if pad[axis] <= 0:
            snapped &= (pts[:, axis] >= lo[axis]) & (pts[:, axis] <= hi[axis])
    return _touching_solid(snapped, pts, solid_centroids, h)


def _touching_solid(
    mask: np.ndarray,
    pts: np.ndarray,
    solid_centroids: np.ndarray | None,
    h: float,
) -> np.ndarray:
    if solid_centroids is None or not mask.any() or h <= 0:
        return mask
    solid = np.asarray(solid_centroids, float).reshape(-1, 3)
    if solid.size == 0:
        return mask
    chosen = np.where(mask)[0]
    ok = np.array([np.linalg.norm(solid - pts[i], axis=1).min() <= h for i in chosen])
    if not ok.any():
        return np.zeros(len(pts), dtype=bool)
    out = np.zeros(len(pts), dtype=bool)
    out[chosen[ok]] = True
    return out


@dataclass
class Masks:
    design: np.ndarray  # elements the optimizer may change
    preserve: np.ndarray  # elements held at rho = 1
    void: np.ndarray  # elements held at rho = rho_min
    warm: np.ndarray  # design elements inside the warm-start geometry
    constraints: torch.Tensor  # [n_nodes, 3] bool
    forces: list[torch.Tensor]  # per load case, [n_nodes, 3], already scaled by safety_factor
    weights: list[float]
    report: dict = field(default_factory=dict)


def build_masks(problem: TOProblem, mesh: HexMesh) -> Masks:
    cen = mesh.centroids
    void = contains_any(problem.void, cen)
    preserve_raw = contains_any(problem.preserve, cen)
    overlap_mask = void & preserve_raw
    # Required BCs are nodal; they are not on problem.preserve. Overlap here is
    # optional candidate skins (jaws, seat, tip) clipped by keep-out / envelope.
    required = np.zeros(len(cen), dtype=bool)
    for item in (*problem.supports, *problem.load_cases):
        required |= contains(item.region, cen)
    optional_clip = overlap_mask & ~required
    required_clip = overlap_mask & required
    preserve = preserve_raw & ~void
    design = ~void & ~preserve
    warm = contains_any(problem.warm_start, cen) & design
    if design.sum() == 0:
        raise ProblemSetupError("no design elements left after applying preserve/void regions")

    nodes = mesh.nodes.numpy()
    constraints = torch.zeros((mesh.n_nodes, 3), dtype=torch.bool)
    support_counts: dict[str, int] = {}
    for s in problem.supports:
        sel = select_nodes_for_region(s.region, nodes, mesh.h, solid_centroids=cen[~void])
        n = int(sel.sum())
        if n == 0:
            raise ProblemSetupError(
                f"support '{s.id}' cannot be resolved at element size {mesh.h:.3g}: "
                "the physical region is smaller than the mesh can resolve, or it "
                "does not meet the grid. Reduce target_element_size."
            )
        support_counts[s.id] = n
        for d in s.fixed_dofs:
            constraints[torch.as_tensor(sel), _AXIS[d]] = True

    forces: list[torch.Tensor] = []
    weights: list[float] = []
    load_counts: dict[str, int] = {}
    for lc in problem.load_cases:
        sel = select_nodes_for_region(lc.region, nodes, mesh.h, solid_centroids=cen[~void])
        n = int(sel.sum())
        if n == 0:
            raise ProblemSetupError(
                f"load case '{lc.id}' cannot be resolved at element size {mesh.h:.3g}: "
                "the physical region is smaller than the mesh can resolve, or it "
                "does not meet the grid. Reduce target_element_size."
            )
        load_counts[lc.id] = n
        F = torch.zeros((mesh.n_nodes, 3))
        F[torch.as_tensor(sel)] = torch.as_tensor(lc.force_N) * problem.safety_factor / n
        forces.append(F)
        weights.append(float(lc.weight))

    optional_n = int(optional_clip.sum())
    required_n = int(required_clip.sum())
    report = {
        "n_elem": mesh.n_elem,
        "n_nodes": mesh.n_nodes,
        "n_dof": mesh.n_dof,
        "grid": list(mesh.shape),
        "spacing": np.round(mesh.spacing, 3).tolist(),
        "design_elems": int(design.sum()),
        "preserve_elems": int(preserve.sum()),
        "void_elems": int(void.sum()),
        "warm_start_elems": int(warm.sum()),
        "warm_start_fraction_of_design": float(warm.sum() / max(design.sum(), 1)),
        "support_nodes": support_counts,
        "load_nodes": load_counts,
        "safety_factor": problem.safety_factor,
        "optional_candidate_clipped_by_keepout": optional_n,
        "required_preserve_clipped_by_keepout": required_n,
    }
    return Masks(design, preserve, void, warm, constraints, forces, weights, report)
