"""The solver must build its boundary conditions from the agent's regions, not an archetype.

These tests exist because the adapter used to read only `attachment_method` and load case 0,
so a better reasoning layer changed nothing downstream. They pin the contract: agent regions
win, fallbacks are declared, and `payload_kind` no longer decides where anything goes.
"""

from __future__ import annotations

import pytest

from to_agent.contracts import BoxRegion, LoadCase
from to_agent.integration.agent_regions import merge_load_cases
from to_agent.integration.from_requirements import build_from_requirements

BASE = {
    "material": "PLA",
    "desk_thickness_mm": 20.0,
    "payload_size_mm": 40.0,
    "attachment_method": "clamp",
    "envelope": {"max_protrusion_mm": 90.0, "max_width_mm": 40.0, "max_height_mm": 60.0},
}

MOUNTS = [
    {"name": "upper_mount", "position_mm": [-20.0, 0.0, 22.0], "normal": [0, 0, 1], "area_mm2": 400.0},
    {"name": "lower_mount", "position_mm": [-20.0, 0.0, -2.0], "normal": [0, 0, -1], "area_mm2": 400.0},
]
LOAD_REGIONS = [
    {"name": "strap_seat", "position_mm": [60.0, 0.0, -20.0], "direction": [0, 0, -1]},
    {"name": "tip", "position_mm": [75.0, 0.0, -14.0], "direction": [1, 0, 0]},
]


def _input(**over) -> dict:
    data = {**BASE, "fixed_regions": MOUNTS, "load_regions": LOAD_REGIONS}
    data["loads"] = [
        {"load_case_id": "static_gravity", "region_name": "strap_seat", "force_N": [0.0, 0.0, -49.05]},
        {"load_case_id": "tip_retention", "region_name": "tip", "force_N": [19.62, 0.0, 0.0]},
    ]
    data.update(over)
    return data


def test_merge_keeps_retention_not_unmarked_template_loads():
    box = BoxRegion(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0))
    agent = [LoadCase(id="static_gravity", region=box, force_N=(0.0, 0.0, -1.0), provenance="user")]
    template = [
        LoadCase(id="tip_retention", region=box, force_N=(1.0, 0.0, 0.0), role="retention"),
        LoadCase(id="side_swing", region=box, force_N=(0.0, 1.0, 0.0)),
    ]
    merged, kept = merge_load_cases(agent, template)
    assert [c.id for c in merged] == ["static_gravity", "tip_retention"]
    assert [c.id for c in kept] == ["tip_retention"]


def test_agent_regions_drive_supports_and_loads():
    problem, report = build_from_requirements(_input(), element_size=6.0)

    assert [s.id for s in problem.supports] == ["upper_mount", "lower_mount"]
    assert report["supports_from"] == "agent fixed_regions"
    # each support patch is centred on the position the agent gave
    upper = next(s for s in problem.supports if s.id == "upper_mount")
    cz = (upper.region.min[2] + upper.region.max[2]) / 2.0
    assert upper.region.min[0] == pytest.approx(-30.0) and upper.region.max[0] == pytest.approx(-10.0)
    assert cz == pytest.approx(22.0)

    assert [c.id for c in problem.load_cases] == ["static_gravity", "tip_retention"]
    assert problem.load_cases[1].force_N == (19.62, 0.0, 0.0)
    # nothing was invented
    assert problem.assumptions == []


def test_every_load_case_scales_with_payload():
    """Finding 10: changing the payload must move all cases, not only the primary."""
    light = build_from_requirements(_input(), element_size=6.0)[0]
    heavy_in = _input()
    heavy_in["loads"] = [
        {**heavy_in["loads"][0], "force_N": [0.0, 0.0, -490.5]},
        {**heavy_in["loads"][1], "force_N": [196.2, 0.0, 0.0]},
    ]
    heavy = build_from_requirements(heavy_in, element_size=6.0)[0]

    for a, b in zip(light.load_cases, heavy.load_cases):
        assert b.force_N == pytest.approx(tuple(10.0 * v for v in a.force_N))


def test_payload_kind_does_not_change_boundary_conditions():
    """Same regions, different archetype label -> identical problem."""
    strap = build_from_requirements(_input(payload_kind="strap"), element_size=6.0)[0]
    box = build_from_requirements(_input(payload_kind="box"), element_size=6.0)[0]

    assert [s.model_dump() for s in strap.supports] == [s.model_dump() for s in box.supports]
    assert [c.model_dump() for c in strap.load_cases] == [c.model_dump() for c in box.load_cases]
    assert strap.design_domain.model_dump() == box.design_domain.model_dump()


def test_domain_contains_the_agent_regions():
    problem, _ = build_from_requirements(_input(), element_size=6.0)
    dom = problem.design_domain
    for region in [s.region for s in problem.supports] + [c.region for c in problem.load_cases]:
        for i in range(3):
            assert dom.min[i] <= region.min[i] and region.max[i] <= dom.max[i], f"axis {i}"


def test_fallback_fires_and_is_declared_when_regions_are_missing():
    data = _input(fixed_regions=[], load_regions=[], loads=[
        {"load_case_id": "static_gravity", "force_N": [0.0, 0.0, -49.05]},
    ])
    problem, report = build_from_requirements(data, element_size=6.0)

    assert report["supports_from"] == "'clamp' archetype"
    fields = {a.field for a in problem.assumptions}
    assert "supports" in fields and "load_cases" in fields
    assert problem.supports and problem.load_cases  # still solvable, just declared


def test_named_placeholder_regions_are_not_treated_as_the_origin():
    """A region with no position_mm carries no intent; using the (0,0,0) default put
    supports and loads inside the desk and silently produced a zero-load solve."""
    data = _input(
        fixed_regions=[{"name": "mount_contact"}],
        load_regions=[{"name": "cup_cavity"}],
        loads=[{"load_case_id": "static_gravity", "region_name": "cup_cavity", "force_N": [0.0, 0.0, -10.79]}],
    )
    problem, report = build_from_requirements(data, element_size=6.0)

    assert report["supports_from"] == "'clamp' archetype"
    bases = " ".join(a.basis for a in problem.assumptions)
    assert "no position_mm" in bases
    assert all(s.id != "mount_contact" for s in problem.supports)


def test_envelope_bounds_the_domain():
    """max_protrusion/max_width are hard bounds; max_height is applied about the desk top."""
    problem, _ = build_from_requirements(_input(), element_size=6.0)
    dom = problem.design_domain
    assert dom.max[0] == pytest.approx(90.0)
    assert dom.min[1] == pytest.approx(-20.0) and dom.max[1] == pytest.approx(20.0)
    assert dom.max[2] <= 20.0 + 60.0 + 1e-9
    assert dom.min[2] >= 20.0 - 60.0 - 1e-9


def test_impossible_height_envelope_is_rejected_not_silently_widened():
    with pytest.raises(ValueError, match="max_height_mm"):
        build_from_requirements(
            _input(envelope={"max_protrusion_mm": 90.0, "max_width_mm": 40.0, "max_height_mm": 2.0}),
            element_size=6.0,
        )
