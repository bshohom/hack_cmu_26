"""A shipped STL must be one body, and the FE check must describe that same body.

The saved desk-hook run exported a watertight STL made of two disconnected pieces and
reported a factor of safety of 77.6, because the post-check solves the voxel grid where
"removed" elements keep a residual stiffness that bridges the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from to_agent.contracts import BoxRegion, LoadCase, Support, TOProblem
from to_agent.integration.run import prepare
from to_agent.postprocess.connectivity import carries_boundary_conditions, keep_largest_component
from to_agent.postprocess.export import save_stl


def _two_bar_problem() -> TOProblem:
    return TOProblem(
        design_domain=BoxRegion(min=(0, 0, 0), max=(60, 10, 30)),
        supports=[Support(id="s", region=BoxRegion(min=(0, 0, 0), max=(4, 10, 30)))],
        load_cases=[LoadCase(id="l", region=BoxRegion(min=(56, 0, 0), max=(60, 10, 30)), force_N=(0, 0, -10))],
        target_element_size=2.0,
    )


def _mesh_and_rho():
    problem = _two_bar_problem()
    mesh, masks = prepare(problem)
    cen = mesh.centroids
    # Two separated slabs: z in [0,8] and z in [20,30]. Nothing joins them.
    rho = np.zeros(mesh.n_elem)
    rho[(cen[:, 2] <= 8.0)] = 1.0
    rho[(cen[:, 2] >= 20.0)] = 1.0
    return problem, mesh, masks, rho


def test_two_bodies_are_reduced_to_one(tmp_path):
    _, mesh, _, rho = _mesh_and_rho()
    before = save_stl(mesh, rho, tmp_path / "before.stl")
    assert before["components"] == 2, "fixture should start disconnected"

    trimmed, info = keep_largest_component(mesh, rho)
    assert info["trimmed"] is True
    assert info["components_before"] == 2
    assert info["removed_elements"] > 0

    after = save_stl(mesh, trimmed, tmp_path / "after.stl")
    assert after["components"] == 1
    assert after["watertight"] is True


def test_single_body_is_left_alone(tmp_path):
    _, mesh, _, rho = _mesh_and_rho()
    rho[:] = 0.0
    rho[mesh.centroids[:, 2] <= 8.0] = 1.0
    trimmed, info = keep_largest_component(mesh, rho)
    assert info["components_before"] == 1
    assert info["trimmed"] is False
    assert np.array_equal(trimmed, rho)


def test_trim_that_sheds_a_load_region_is_reported(tmp_path):
    """Trimming must never quietly produce a part with nothing at the load."""
    problem, mesh, masks, rho = _mesh_and_rho()
    # Keep only the lower slab, which reaches the support but the load patch spans both.
    rho_low = np.zeros(mesh.n_elem)
    rho_low[mesh.centroids[:, 2] <= 8.0] = 1.0
    status = carries_boundary_conditions(mesh, rho_low, masks, [lc.id for lc in problem.load_cases])
    assert status["supports_attached"] is True
    assert status["loads_attached"] is True  # the load box spans z, so it is still reached

    # Nothing solid at all: both must be reported as detached.
    empty = carries_boundary_conditions(mesh, np.zeros(mesh.n_elem), masks, ["l"])
    assert empty["supports_attached"] is False
    assert empty["loads_attached"] is False
    assert empty["detached_loads"] == ["l"]
