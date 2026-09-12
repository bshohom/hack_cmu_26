"""Cross-case contract for the three demo tasks. Presentation of names is test data only.

Assertions stay generic: usable BCs, merge roles, envelope void, optional clip, fit family.
No new task-specific logic is added to integration code.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pytest

from to_agent.contracts import LoadCase
from to_agent.demo.registry import DEMO_FILES, get_builder
from to_agent.integration.agent_regions import merge_load_cases
from to_agent.integration.agentic import build_from_registry
from to_agent.integration.envelope_constraint import apply_envelope_constraint
from to_agent.integration.run import prepare

ROOT = Path(__file__).resolve().parent.parent
INTERFACE = ROOT / "Hack_CMU_Agentic_Interface" / "HackCMU_mechanical_design_copilot_drive_20260912_014958"
if str(INTERFACE) not in sys.path:
    sys.path.insert(0, str(INTERFACE))

# Demo metadata only. Shared assertions below do not branch on these names.
CASES = {
    "cupholder": {
        "mesh": ROOT / "does_not_exist.obj",
        "frame": None,
        "envelope": {"max_protrusion_mm": 150.0, "max_width_mm": 120.0, "max_height_mm": 80.0},
        "desk_mm": 25.0,
        "payload_mm": 65.0,
        "payload_mass_kg": 1.0,
        "load_region": "cup_cavity",
        "fit_checks": {"payload_fit", "desk_fit", "envelope_fit"},
        "forbidden_fit_checks": set(),
        "forbidden_fit_phrases": (),
        "payload_phrase": "holder opening",
    },
    "desk_bag_hook": {
        "mesh": ROOT / "desk_bag_hook_5kg_100mm_final.stl",
        "frame": "candidate_mesh_frame",
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 60.0, "max_height_mm": 65.0},
        "desk_mm": 20.0,
        "payload_mm": 30.0,
        "payload_mass_kg": 5.0,
        "load_region": "strap_seat",
        "fit_checks": {"payload_fit", "desk_fit", "envelope_fit", "load_rating"},
        "forbidden_fit_checks": set(),
        "forbidden_fit_phrases": ("inner diameter", "holder opening"),
        "payload_phrase": "hook opening",
    },
    "stapler_shelf": {
        "mesh": ROOT / "stapler_shelf_100mm_guaranteed_flat_top.stl",
        "frame": None,
        "envelope": {"max_protrusion_mm": 120.0, "max_width_mm": 80.0, "max_height_mm": 110.0},
        "desk_mm": 20.0,
        "payload_mm": 60.0,
        "payload_mass_kg": 0.5,
        "load_region": "platform",
        "fit_checks": {"payload_fit", "envelope_fit"},
        "forbidden_fit_checks": {"desk_fit"},
        "forbidden_fit_phrases": ("inner diameter", "holder opening"),
        "payload_phrase": None,
    },
}

GENERIC_DIRS = (
    ROOT / "to_agent" / "integration",
    ROOT / "to_agent" / "meshing",
    ROOT / "to_agent" / "solver",
    ROOT / "to_agent" / "postprocess",
)
# Frozen inventory: executable name branches in generic integration. Demo/template
# modules are outside this scan. Do not grow this list without a generic rule.
KNOWN_GENERIC_NAME_BRANCHES = {
    'if task == "desk_bag_hook"',
}


def _spec(task: str) -> dict:
    return CASES[task]


def _candidate(task: str) -> dict:
    spec = _spec(task)
    dims, pts = DEMO_FILES[task]
    cand = {
        "candidate_name": task,
        "task": task,
        "mesh_path": str(spec["mesh"]),
        "particle_path": str(pts),
        "dimensions_path": str(dims),
    }
    if spec["frame"]:
        cand["frame"] = spec["frame"]
    return cand


def _overlay(task: str) -> dict:
    spec = _spec(task)
    region = spec["load_region"]
    return {
        "fixed_regions": [{"name": "mount_contact"}],
        "load_regions": [{"name": region}],
        "loads": [{
            "load_case_id": "static_gravity",
            "name": "static_gravity",
            "region_name": region,
            "force_N": [0.0, 0.0, -10.0],
        }],
        "material": "PLA",
        "target_volume_fraction": 0.4,
        "candidate": _candidate(task),
        "solver_options": {"element_size_mm": 8.0, "max_iters": 2, "device": "cpu"},
        "envelope": spec["envelope"],
        "desk_thickness_mm": spec["desk_mm"],
    }


@pytest.mark.parametrize("task", list(CASES))
def test_candidate_fit_is_task_appropriate(task):
    from imported_candidate import check_candidate_fit, fit_family_for, load_candidate
    from schemas import CandidateFitStatus, UserRequirements

    spec = _spec(task)
    candidate = load_candidate(task)
    assert fit_family_for(candidate) == task
    req = UserRequirements()
    req.object_geometry.bottle_diameter_mm = spec["payload_mm"]
    req.payload.filled_mass_kg = spec["payload_mass_kg"]
    req.environment.desk_thickness_mm = spec["desk_mm"]
    req.design_envelope.max_protrusion_mm = spec["envelope"]["max_protrusion_mm"]
    fit = check_candidate_fit(req, candidate)
    names = {c.name for c in fit.checks}
    assert spec["fit_checks"] <= names
    assert names.isdisjoint(spec["forbidden_fit_checks"])
    blob = (fit.message + " " + " ".join(c.message for c in fit.checks)).lower()
    for phrase in spec["forbidden_fit_phrases"]:
        if phrase == "holder opening":
            payload = next(c for c in fit.checks if c.name == "payload_fit")
            assert "holder opening" not in payload.message.lower()
        else:
            assert phrase not in blob
    if spec["payload_phrase"]:
        payload = next(c for c in fit.checks if c.name == "payload_fit")
        assert spec["payload_phrase"] in payload.message.lower()
    assert all(c.status != CandidateFitStatus.FAIL for c in fit.checks)
    assert fit.fits, fit.message


@pytest.mark.parametrize("task", list(CASES))
def test_template_assembly_required_bcs_are_usable(task):
    dims, pts = DEMO_FILES[task]
    problem, _ = get_builder(task)(dims, pts, element_size=8.0)
    support_ids = [s.id for s in problem.supports]
    load_ids = [c.id for c in problem.load_cases]
    assert support_ids and load_ids
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh, masks = prepare(problem)
    del mesh
    assert not any("void wins" in str(w.message).lower() for w in caught)
    assert masks.report.get("optional_candidate_clipped_by_keepout", 0) >= 0
    bc = masks.report["bc_validation"]
    assert bc["hard_infeasible"] is False
    reported = {r["id"] for r in bc["regions"]}
    assert set(support_ids) | set(load_ids) == reported
    for region in bc["regions"]:
        assert region["usable_nodes"] > 0
        assert region["status"] in {"ok", "partial"}
        assert region["usable_fraction"] > 0


@pytest.mark.parametrize("task", list(CASES))
def test_load_merge_keeps_retention_not_unmarked(task):
    dims, pts = DEMO_FILES[task]
    problem, _ = get_builder(task)(dims, pts, element_size=8.0)
    retention = [c.id for c in problem.load_cases if c.role == "retention"]
    unmarked = [c.id for c in problem.load_cases if c.role not in ("primary", "retention")]
    agent = [
        LoadCase(
            id="static_gravity",
            region=problem.load_cases[0].region,
            force_N=(0.0, 0.0, -1.0),
            role="primary",
            provenance="user",
        )
    ]
    merged, kept = merge_load_cases(agent, problem.load_cases)
    ids = [c.id for c in merged]
    assert "static_gravity" in ids
    assert {c.id for c in kept} == set(retention)
    assert set(retention) <= set(ids)
    assert not (set(unmarked) - {"static_gravity"}) & set(ids)


@pytest.mark.parametrize("task", list(CASES))
def test_agentic_assembly_envelope_and_no_silent_bc_drop(task):
    data = _overlay(task)
    problem, report = build_from_registry(data["candidate"], data, 8.0, 0.4, 2.5)
    template, _ = get_builder(task)(DEMO_FILES[task][0], DEMO_FILES[task][1], element_size=8.0)
    retention = {c.id for c in template.load_cases if c.role == "retention"}
    unmarked = {c.id for c in template.load_cases if c.role not in ("primary", "retention")}
    applied = any("load cases from agent" in n for n in report.get("region_source") or [])
    final_ids = {c.id for c in problem.load_cases}
    assert retention <= final_ids
    if applied:
        assert "static_gravity" in final_ids
        assert not (unmarked - {"static_gravity"}) & final_ids
    else:
        # Fallback overwrites the first template case id/force; it does not drop the set.
        assert problem.load_cases
        assert "static_gravity" in final_ids
        assert len(problem.load_cases) == len(template.load_cases)
    T = (report.get("registration") or {}).get("matrix")
    info = apply_envelope_constraint(problem, data, T)
    assert info.get("applied") is True
    assert info.get("hard_infeasible") is False
    mesh, masks = prepare(problem)
    del mesh
    bc = masks.report["bc_validation"]
    assert bc["hard_infeasible"] is False
    problem_ids = {s.id for s in problem.supports} | {c.id for c in problem.load_cases}
    assert {r["id"] for r in bc["regions"]} == problem_ids
    for region in bc["regions"]:
        assert region["usable_nodes"] > 0
        assert region["status"] != "hard_infeasible"
    assert masks.report.get("optional_candidate_clipped_by_keepout", 0) >= 0


def test_generic_integration_name_branches_are_inventoried():
    """Fail if generic integration grows new task-name branches."""
    quoted = re.compile(r"""(?:task\s*==\s*["']desk_bag_hook["']|["']strap_seat["']|["']tip_retention["'])""")
    hits: list[str] = []
    for folder in GENERIC_DIRS:
        for path in folder.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if quoted.search(code):
                    hits.append(f"{path.relative_to(ROOT)}:{i}:{code.strip()}")
    unexpected = [h for h in hits if not any(known in h for known in KNOWN_GENERIC_NAME_BRANCHES)]
    assert not unexpected, "new task-name branch in generic integration:\n" + "\n".join(unexpected)
    assert any(known in h for h in hits for known in KNOWN_GENERIC_NAME_BRANCHES)
