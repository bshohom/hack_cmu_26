import numpy as np
import pytest

from to_agent.demo.registry import DEMO_FILES, get_builder
from to_agent.meshing.masks import build_masks
from to_agent.meshing.voxel_backend import build_hex_grid
from to_agent.regions import contains_any, resolve_domain


def _build(task):
    dims, pts = DEMO_FILES[task]
    problem, report = get_builder(task)(dims, pts)
    mesh = build_hex_grid(resolve_domain(problem), problem.target_element_size)
    return problem, report, mesh, build_masks(problem, mesh)


def _check_masks(masks):
    r = masks.report
    assert r["design_elems"] > 0 and r["preserve_elems"] > 0
    assert all(n > 0 for n in r["support_nodes"].values())
    assert all(n > 0 for n in r["load_nodes"].values())
    assert not (masks.design & masks.preserve).any()
    assert not (masks.design & masks.void).any()
    assert 0.05 < r["warm_start_fraction_of_design"] < 0.8


def test_hook_geometry_and_masks():
    problem, report, mesh, masks = _build("desk_bag_hook")
    g = report["hook"]
    assert 18 < g["measured_gap"] < 24  # nominal clamp gap 22 between rib tips ~18
    assert g["z_bot_arm"][1] < g["z_rib_lo"] < g["z_rib_hi"] < g["z_top_arm"][0]
    assert -2 < g["x_back_in"] < 2 and 8 < g["x_back_out"] < 16
    assert 85 < g["x_tip"][0] < 100 and g["z_tip_top"] > 30
    assert 3 < g["z_seat"] < 7
    _check_masks(masks)
    assert masks.report["void_elems"] > 0
    probe = np.array([[-30.0, 0.0, 11.0], [65.0, 0.0, 22.0]])  # inside the desk; inside the strap opening
    assert contains_any(problem.void, probe).all()
    assert abs(problem.load_cases[0].force_N[2] + 9.81 * 5.0) < 1e-6


def test_shelf_geometry_and_masks():
    problem, report, mesh, masks = _build("stapler_shelf")
    g = report["shelf"]
    assert 92 < g["z_under"] < 96 and abs(g["z_top"] - 100.0) < 0.5
    assert 5 < g["z_base_top"] < 11
    _check_masks(masks)
    assert contains_any(problem.preserve, np.array([[0.0, 0.0, 97.0]])).all()
    assert not contains_any(problem.preserve, np.array([[60.0, 0.0, 50.0]])).any()
    assert problem.design_domain.max[2] <= 100.6
