"""Linear FE check of the optimized design at nominal (unfactored) load.

Thresholds the density field (rho >= threshold -> solid, else void), solves each load case
once on the same hex grid and reports max displacement, max von Mises stress and the factor
of safety against the material yield. A voxel-level linear check, not a certification.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..contracts import TOProblem
from ..meshing.masks import Masks
from ..meshing.voxel_backend import HexMesh
from ..solver.simp import make_model, solve

VOID_STIFFNESS = 1e-4  # relative stiffness of removed material (keeps K well posed)


def von_mises(sigma: torch.Tensor) -> torch.Tensor:
    """Per-element von Mises stress from a [n_elem, ..., 3, 3] stress tensor (integration
    points, if present, are averaged)."""
    while sigma.dim() > 3:
        sigma = sigma.mean(dim=1)
    s = sigma - torch.einsum("nii->n", sigma)[:, None, None] / 3.0 * torch.eye(3, device=sigma.device, dtype=sigma.dtype)
    return torch.sqrt(1.5 * torch.einsum("nij,nij->n", s, s))


def post_check(
    problem: TOProblem,
    mesh: HexMesh,
    masks: Masks,
    rho: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    solid = rho >= threshold
    model, tdev, mode = make_model(mesh, masks, problem, device)
    C0 = model.material.C.clone()
    scale = torch.as_tensor(np.where(solid, 1.0, VOID_STIFFNESS), device=tdev, dtype=C0.dtype)
    model.material.C = torch.einsum("n,nijkl->nijkl", scale, C0)
    solid_t = torch.as_tensor(solid, device=tdev)
    sf = problem.safety_factor or 1.0
    yield_mpa: Optional[float] = problem.material.yield_MPa

    cases = []
    for lc, F in zip(problem.load_cases, masks.forces):
        model.forces = (F / sf).to(tdev)  # nominal load
        u, f, sigma, *_ = solve(model, device, tdev)
        u = u.to(tdev)
        disp = torch.linalg.norm(u, dim=1)
        vm = von_mises(sigma.to(tdev))
        vm_solid = vm[solid_t] if solid_t.any() else vm
        vm_max = float(vm_solid.max().item())
        cases.append(
            {
                "load_case_id": lc.id,
                "load_force_N": [float(v) for v in lc.force_N],
                "max_displacement_mm": float(disp.max().item()),
                "max_von_mises_MPa": vm_max,
                "factor_of_safety": (yield_mpa / vm_max) if (yield_mpa and vm_max > 0) else None,
            }
        )
    worst = min(cases, key=lambda c: c["factor_of_safety"] if c["factor_of_safety"] is not None else float("inf"))
    return {
        "threshold": threshold,
        "solid_elements": int(solid.sum()),
        "nominal_load_note": f"loads divided by safety_factor {sf} (nominal); yield {yield_mpa} MPa ({problem.material.name})",
        "solver": f"torch-fem linear hex8 on the thresholded design ({mode}, h={mesh.h:.2f} mm)",
        "cases": cases,
        "worst_case": worst,
        "disclaimer": (
            "Linear elastic voxel FE on the rho>=0.5 isosurface at nominal load. "
            "Not a physical safety certification; verify by print and test."
        ),
    }


def to_analysis_output(check: dict) -> dict:
    """Map a post_check() result to the interface's AnalysisOutput fields."""
    w = check["worst_case"]
    return {
        "load_case_id": w["load_case_id"],
        "load_force_N": w["load_force_N"],
        "is_mock": False,
        "max_displacement_mm": w["max_displacement_mm"],
        "max_stress_pa": w["max_von_mises_MPa"] * 1e6,
        "factor_of_safety": w["factor_of_safety"],
        "reaction_forces_N": [],
        "is_safety_validation": False,
        "solver": check["solver"],
        "solver_status": "computed",
        "disclaimer": check["disclaimer"],
    }
