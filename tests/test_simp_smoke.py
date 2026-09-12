import numpy as np
import torch

from to_agent.contracts import BoxRegion, LoadCase, Support, TOProblem
from to_agent.meshing.masks import build_masks
from to_agent.meshing.voxel_backend import build_hex_grid
from to_agent.postprocess.export import isosurface
from to_agent.solver.simp import optimize


def cantilever() -> TOProblem:
    return TOProblem(
        design_domain=BoxRegion(min=(0, 0, 0), max=(8, 4, 4)),
        preserve=[BoxRegion(min=(0, 0, 0), max=(1, 4, 4))],
        supports=[Support(id="wall", region=BoxRegion(min=(-0.1, -1, -1), max=(0.1, 5, 5)))],
        load_cases=[
            LoadCase(id="tip", region=BoxRegion(min=(7.9, -1, -0.1), max=(8.1, 5, 0.1)), force_N=(0, 0, -1.0)),
            LoadCase(id="side", region=BoxRegion(min=(7.9, -1, -0.1), max=(8.1, 5, 0.1)), force_N=(0, 0.5, 0), weight=0.5),
        ],
        volume_fraction=0.4,
        target_element_size=1.0,
        max_iters=4,
        safety_factor=1.0,
    )


def test_simp_smoke_cpu():
    problem = cantilever()
    mesh = build_hex_grid(problem.design_domain, problem.target_element_size)
    masks = build_masks(problem, mesh)
    assert mesh.n_elem == 8 * 4 * 4
    result = optimize(problem, mesh, masks, torch.device("cpu"), log=None)
    assert result.iters == 4
    assert all(np.isfinite(result.compliance))
    assert result.compliance[-1] <= result.compliance[0] * 1.05
    assert abs(result.volume[-1] - 0.4) < 0.02
    assert np.all(result.rho[masks.preserve] == 1.0)
    assert result.rho.min() >= problem.rho_min - 1e-12 and result.rho.max() <= 1.0 + 1e-12
    surf = isosurface(mesh, result.rho, 0.5)
    assert surf.n_points > 0
