from pathlib import Path

import pytest

from to_agent.demo.cupholder import build_cupholder_problem
from to_agent.meshing.masks import build_masks
from to_agent.meshing.voxel_backend import build_hex_grid
from to_agent.regions import resolve_domain

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cupholder():
    return build_cupholder_problem(ROOT / "cupholder_dimensions.txt", ROOT / "cupholder_surface_particles.obj")


def test_scan_matches_dimensions(cupholder):
    _, report = cupholder
    scan = report["scan_check"]
    assert scan["consistent"], scan
    assert abs(scan["r_in_found"] - 35.0) < 1.5 and abs(scan["r_out_found"] - 40.0) < 1.5


def test_clamp_levels(cupholder):
    _, report = cupholder
    clamp = report["clamp"]
    assert 25 < clamp["measured_gap"] < 35  # nominal desk gap 30
    assert clamp["hook_top"] < clamp["plate_bottom"] < clamp["plate_top"]
    assert clamp["x_spine"] < clamp["x_far"]


def test_masks_on_grid(cupholder):
    problem, _ = cupholder
    mesh = build_hex_grid(resolve_domain(problem), problem.target_element_size)
    masks = build_masks(problem, mesh)
    r = masks.report
    assert r["design_elems"] > 0 and r["void_elems"] > 0 and r["preserve_elems"] > 0
    assert all(n > 0 for n in r["support_nodes"].values())
    assert all(n > 0 for n in r["load_nodes"].values())
    assert 0.05 < r["warm_start_fraction_of_design"] < 0.6
    assert not (masks.design & masks.preserve).any()
    assert not (masks.design & masks.void).any()
    assert not (masks.preserve & masks.void).any()
