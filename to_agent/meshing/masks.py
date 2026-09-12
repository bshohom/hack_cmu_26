"""Element and node masks from a TOProblem on a HexMesh."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from ..contracts import TOProblem
from ..regions import contains, contains_any
from .voxel_backend import HexMesh

_AXIS = {"x": 0, "y": 1, "z": 2}


class ProblemSetupError(ValueError):
    """Raised with an agent-readable message when a problem cannot be set up on the grid."""


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
    overlap = int(np.sum(void & preserve_raw))
    if overlap:
        warnings.warn(f"{overlap} elements are both preserve and void; void wins", stacklevel=2)
    preserve = preserve_raw & ~void
    design = ~void & ~preserve
    warm = contains_any(problem.warm_start, cen) & design
    if design.sum() == 0:
        raise ProblemSetupError("no design elements left after applying preserve/void regions")

    nodes = mesh.nodes.numpy()
    constraints = torch.zeros((mesh.n_nodes, 3), dtype=torch.bool)
    support_counts: dict[str, int] = {}
    for s in problem.supports:
        sel = contains(s.region, nodes)
        n = int(sel.sum())
        if n == 0:
            raise ProblemSetupError(
                f"support '{s.id}' selects no nodes at element size {mesh.h:.3g}; "
                "enlarge its region or reduce target_element_size"
            )
        support_counts[s.id] = n
        for d in s.fixed_dofs:
            constraints[torch.as_tensor(sel), _AXIS[d]] = True

    forces: list[torch.Tensor] = []
    weights: list[float] = []
    load_counts: dict[str, int] = {}
    for lc in problem.load_cases:
        sel = contains(lc.region, nodes)
        n = int(sel.sum())
        if n == 0:
            raise ProblemSetupError(
                f"load case '{lc.id}' selects no nodes at element size {mesh.h:.3g}; "
                "enlarge its region or reduce target_element_size"
            )
        load_counts[lc.id] = n
        F = torch.zeros((mesh.n_nodes, 3))
        F[torch.as_tensor(sel)] = torch.as_tensor(lc.force_N) * problem.safety_factor / n
        forces.append(F)
        weights.append(float(lc.weight))

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
    }
    return Masks(design, preserve, void, warm, constraints, forces, weights, report)
