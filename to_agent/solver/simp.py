"""SIMP compliance minimization on torch-fem (optimality-criteria update).

Ported from torch-fem's examples/optimization/solid/topology.ipynb with: a sparse density
filter, updates restricted to design elements, preserve/void elements pinned, and compliance
summed over load cases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .. import device as _device  # noqa: F401
import torch
from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa

from ..contracts import TOProblem
from ..meshing.masks import Masks
from ..meshing.voxel_backend import HexMesh

LogFn = Callable[[str], None]


@dataclass
class SIMPResult:
    rho: np.ndarray
    compliance: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    change: list[float] = field(default_factory=list)
    iter_seconds: list[float] = field(default_factory=list)
    u: list[np.ndarray] = field(default_factory=list)  # final displacement per load case
    wall_time: float = 0.0
    device: str = "cpu"
    solver_mode: str = ""
    iters: int = 0
    converged: bool = False


# ----------------------------------------------------------------------------- model
def make_material(E: float, nu: float, device: torch.device) -> IsotropicElasticity3D:
    return IsotropicElasticity3D(E=torch.tensor(E, device=device), nu=torch.tensor(nu, device=device))


@lru_cache(maxsize=None)
def model_on_device_works(device_type: str) -> bool:
    """Can a Solid live entirely on this device? Probed once on a tiny mesh."""
    if device_type == "cpu":
        return True
    try:
        dev = torch.device(device_type)
        nodes, elements = cube_hexa(3, 3, 3, 1.0, 1.0, 1.0)
        model = Solid(nodes.to(dev), elements.to(dev), make_material(1.0, 0.3, dev))
        model.constraints[nodes.to(dev)[:, 0] == 0] = True
        model.forces[nodes.to(dev)[:, 0] == 1.0, 2] = -1.0
        model.solve()
        return True
    except Exception:
        return False


def make_model(mesh: HexMesh, masks: Masks, problem: TOProblem, device: torch.device) -> tuple[Solid, torch.device, str]:
    """Return (model, tensor_device, mode). mode is 'model-on-device' or 'cpu-model+device-solver'."""
    if model_on_device_works(device.type):
        tdev = device
        mode = "model-on-device"
    else:
        tdev = torch.device("cpu")
        mode = f"cpu-model+{device.type}-solver"
    model = Solid(mesh.nodes.to(tdev), mesh.elements.to(tdev), make_material(problem.material.E_MPa, problem.material.nu, tdev))
    model.constraints = masks.constraints.to(tdev)
    return model, tdev, mode


def solve(model: Solid, device: torch.device, tdev: torch.device):
    if tdev == device:
        return model.solve()
    return model.solve(device=device.type)


# ----------------------------------------------------------------------------- filter
def sparse_filter(centroids: np.ndarray, radius: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear 'hat' density filter H (sparse, n x n) and its row sums."""
    n = len(centroids)
    tree = cKDTree(centroids)
    neighbours = tree.query_ball_point(centroids, radius)
    counts = np.fromiter((len(p) for p in neighbours), dtype=int, count=n)
    rows = np.repeat(np.arange(n), counts)
    cols = np.concatenate(neighbours).astype(int)
    vals = radius - np.linalg.norm(centroids[rows] - centroids[cols], axis=1)
    H = torch.sparse_coo_tensor(
        torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long),
        torch.as_tensor(vals, dtype=torch.get_default_dtype()),
        (n, n),
    ).coalesce().to(device)
    Hs = torch.sparse.mm(H, torch.ones((n, 1), device=device))[:, 0]
    return H, Hs


# ----------------------------------------------------------------------------- OC update
def oc_update(
    rho: torch.Tensor,
    sens: torch.Tensor,
    design: torch.Tensor,
    v_target: float,
    rho_min: float,
    move: float,
) -> torch.Tensor:
    """Optimality-criteria step on the design elements with a volume constraint."""
    rd, sd = rho[design], sens[design]
    lower = torch.clamp(rd * (1 - move), min=rho_min)
    upper = torch.clamp(rd * (1 + move), max=1.0)
    B = torch.clamp(-sd, min=0.0)

    def step(mu: float) -> torch.Tensor:
        return torch.clamp(rd * torch.sqrt(B / mu), lower, upper)

    lo, hi = 1e-20, 1e20
    if step(lo).sum().item() < v_target:  # cannot fill enough within move limits
        return step(lo)
    if step(hi).sum().item() > v_target:  # cannot empty enough within move limits
        return step(hi)
    for _ in range(200):
        mid = float(np.sqrt(lo * hi))
        if step(mid).sum().item() > v_target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-8:
            break
    return step(float(np.sqrt(lo * hi)))


# ----------------------------------------------------------------------------- driver
def initial_density(problem: TOProblem, masks: Masks) -> np.ndarray:
    rho = np.full(len(masks.design), problem.rho_min, dtype=float)
    if masks.warm.any():
        rho[masks.warm] = 1.0
    else:
        rho[masks.design] = problem.volume_fraction
    rho[masks.preserve] = 1.0
    rho[masks.void] = problem.rho_min
    return rho


def optimize(
    problem: TOProblem,
    mesh: HexMesh,
    masks: Masks,
    device: torch.device,
    log: LogFn | None = print,
    max_iters: int | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> SIMPResult:
    t_start = time.time()
    model, tdev, mode = make_model(mesh, masks, problem, device)
    p, rho_min, move = problem.penal, problem.rho_min, problem.move
    radius = problem.filter_radius or 1.5 * mesh.h
    iters = max_iters or problem.max_iters

    design = torch.as_tensor(masks.design, device=tdev)
    preserve = torch.as_tensor(masks.preserve, device=tdev)
    void = torch.as_tensor(masks.void, device=tdev)
    n_design = int(masks.design.sum())
    v_target = problem.volume_fraction * n_design

    k0 = model.k0()
    C0 = model.material.C.clone()
    H, Hs = sparse_filter(mesh.centroids, radius, tdev)
    forces = [F.to(tdev) for F in masks.forces]
    elements = model.elements

    rho = torch.as_tensor(initial_density(problem, masks), device=tdev)
    result = SIMPResult(rho=rho.cpu().numpy(), device=str(device), solver_mode=mode)
    if log:
        log(
            f"SIMP: {mesh.n_elem} elems ({n_design} design), {mesh.n_dof} dof, "
            f"{len(forces)} load case(s), filter R={radius:.2f}, mode={mode}"
        )

    for it in range(iters):
        t0 = time.time()
        model.material.C = torch.einsum("n,nijkl->nijkl", rho**p, C0)
        compliance = 0.0
        sens = torch.zeros_like(rho)
        u_cases: list[np.ndarray] = []
        for F, w in zip(forces, masks.weights):
            model.forces = F
            u, f, *_ = solve(model, device, tdev)
            u = u.to(tdev)
            f = f.to(tdev)
            compliance += w * torch.inner(f.ravel(), u.ravel()).item()
            u_e = u[elements].reshape(model.n_elem, -1)
            w_e = torch.einsum("...i,...ij,...j", u_e, k0, u_e)
            sens += w * (-p * rho ** (p - 1.0) * w_e)
            u_cases.append(u.cpu().numpy())
        sens = torch.sparse.mm(H, (rho * sens)[:, None])[:, 0] / Hs / rho

        rho_new = rho.clone()
        rho_new[design] = oc_update(rho, sens, design, v_target, rho_min, move)
        rho_new[preserve] = 1.0
        rho_new[void] = rho_min
        change = (rho_new - rho).abs().max().item()
        rho = rho_new
        vol = rho[design].mean().item()
        dt = time.time() - t0

        result.compliance.append(float(compliance))
        result.volume.append(vol)
        result.change.append(change)
        result.iter_seconds.append(dt)
        result.u = u_cases
        result.iters = it + 1
        if log:
            log(f"  it {it:3d}  C={compliance:12.5g}  vol={vol:.3f}  change={change:.3f}  {dt:5.2f}s")
        if progress:
            progress(it + 1, iters, float(compliance))
        plateau = (
            it >= 10
            and abs(result.compliance[-6] - compliance) < 1e-3 * abs(compliance)
        )
        if it >= 10 and (change < problem.change_tol or plateau):
            result.converged = True
            break

    result.rho = rho.cpu().numpy()
    result.wall_time = time.time() - t_start
    return result
