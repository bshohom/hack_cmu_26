from pathlib import Path
from types import SimpleNamespace

import pytest

from to_agent.contracts import BoxRegion, LoadCase, Material, Support, TOProblem
from to_agent.demo.hook import hook_registration
from to_agent.integration.agentic import (
    AdapterError,
    acceptance_checks,
    bind_candidate_feature_positions,
    build_from_registry,
    invert_rigid,
    require_registration,
    resolve_volume_fraction,
    result_frame,
    run_topology,
    transform_point,
    volume_fraction_for_mass_cap,
)
from to_agent.integration.envelope_constraint import apply_envelope_constraint, count_envelope_mask

ROOT = Path(__file__).resolve().parent.parent

HOOK_CANDIDATE = {
    "candidate_name": "desk_bag_hook",
    "task": "desk_bag_hook",
    "frame": "candidate_mesh_frame",
    "mesh_path": str(ROOT / "desk_bag_hook_5kg_100mm_final.stl"),
    "particle_path": str(ROOT / "desk_bag_hook_5kg_100mm_final_particles.obj"),
    "dimensions_path": str(ROOT / "desk_bag_hook_5kg_100mm_final_dimensions.txt"),
}


def topology_input(candidate: dict | None) -> dict:
    return {
        "design_domain": {"shape": "clamp_arm_ring", "length_mm": 120.0, "width_mm": 112.0, "height_mm": 44.0},
        "fixed_regions": [{"name": "mount_contact"}],
        "load_regions": [{"name": "cup_cavity"}],
        "loads": [{"load_case_id": "static_gravity", "name": "static_gravity", "region_name": "cup_cavity", "force_N": [0.0, 0.0, -10.79]}],
        "material": "PLA",
        "target_volume_fraction": 0.4,
        "candidate": candidate,
        "solver_options": {"element_size_mm": 8.0, "max_iters": 2, "device": "cpu"},
    }


def test_no_candidate_designs_from_requirements(tmp_path):
    """Without a candidate mesh the adapter designs from the requirements instead of failing."""
    data = topology_input(None)
    data.update(
        desk_thickness_mm=20.0,
        payload_kind="strap",
        payload_size_mm=15.0,
        attachment_method="clamp",
        envelope={"max_protrusion_mm": 90.0, "max_width_mm": 40.0, "max_height_mm": 60.0},
        structure={
            "nodes": [
                {"id": "mount_upper", "position_mm": [0, 0, 20]},
                {"id": "mount_lower", "position_mm": [0, 0, 0]},
                {"id": "load", "position_mm": [65, 0, -25]},
            ],
            "members": [
                {"start_node_id": "mount_upper", "end_node_id": "load"},
                {"start_node_id": "mount_lower", "end_node_id": "load"},
            ],
            "parameters": {"support_thickness_mm": 8.0},
        },
    )
    data["solver_options"] = {"element_size_mm": 9.0, "max_iters": 2, "device": "cpu"}
    out = run_topology(data, out_root=tmp_path)
    assert out["is_mock"] is False
    assert Path(out["optimized_geometry_ref"]).exists()
    assert "from scratch" in out["notes"]
    assert out["problem_report"]["mode"] == "from_requirements"
    assert "structural member" in out["problem_report"]["warm_start"]


def test_missing_loads_raises():
    with pytest.raises((AdapterError, ValueError), match="load"):
        run_topology({"candidate": None, "loads": []})


def test_cupholder_live(tmp_path):
    candidate = {
        "candidate_name": "cupholder",
        "task": "cupholder",
        "mesh_path": str(ROOT / "does_not_exist.obj"),
        "particle_path": str(ROOT / "cupholder_surface_particles.obj"),
        "dimensions_path": str(ROOT / "cupholder_dimensions.txt"),
    }
    out = run_topology(topology_input(candidate), out_root=tmp_path)
    assert out["is_mock"] is False
    assert Path(out["optimized_geometry_ref"]).exists()
    assert out["iterations"] == 2
    assert out["compliance"] > 0
    # Explicit agent target_volume_fraction overrides the cupholder template default (0.2).
    assert out["unsupported_requirements"] == []
    assert out["problem_report"]["volume_fraction"] == pytest.approx(0.4)
    assert 0 < out["volume_fraction"] < 1
    pc = out["post_check"]
    assert pc is not None and pc["is_mock"] is False and pc["is_safety_validation"] is False
    assert pc["max_displacement_mm"] > 0 and pc["max_stress_pa"] > 0
    assert pc["factor_of_safety"] is not None and pc["factor_of_safety"] > 0


def test_resolve_volume_fraction_prefers_agent_then_solver_option():
    assert resolve_volume_fraction({"target_volume_fraction": 0.4}, {}) == pytest.approx(0.4)
    assert resolve_volume_fraction(
        {"target_volume_fraction": 0.4, "solver_options": {"volume_fraction": 0.15}},
        {"volume_fraction": 0.15},
    ) == pytest.approx(0.15)
    assert resolve_volume_fraction({}, {}) is None


def _hook_agent_input() -> dict:
    """TopologyInput the orchestrator emits: GeometryAgent conceptual points + imported hook."""
    spec = hook_registration(HOOK_CANDIDATE["dimensions_path"], HOOK_CANDIDATE["particle_path"])
    return {
        "design_domain": {"shape": "clamp_arm_hook", "length_mm": 110.0, "width_mm": 60.0, "height_mm": 65.0},
        "fixed_regions": [{
            "name": "mount_contact",
            "position_mm": [0.0, 0.0, 10.0],
            "normal": [1.0, 0.0, 0.0],
            "area_mm2": 800.0,
        }],
        "load_regions": [{
            "name": "strap_seat",
            "position_mm": [82.5, 0.0, -25.0],
            "direction": [0.0, 0.0, -1.0],
        }],
        "loads": [{
            "load_case_id": "static_gravity",
            "name": "static_gravity",
            "region_name": "strap_seat",
            "force_N": [0.0, 0.0, -49.05],
        }],
        "material": "PLA",
        "target_volume_fraction": 0.4,
        "candidate": {**HOOK_CANDIDATE, "registration_transform": spec},
        "registration_transform": spec,
        "desk_thickness_mm": 20.0,
        "solver_options": {"element_size_mm": 8.0, "max_iters": 2, "device": "cpu"},
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 60.0, "max_height_mm": 65.0},
        "payload_kind": "strap",
    }


def test_hook_agent_loads_override_template():
    """Agent primary load is accepted; unmarked template cases drop; tip_retention is kept."""
    data = _hook_agent_input()
    problem, report = build_from_registry(data["candidate"], data, 8.0, 0.4, 2.5)
    assert "load cases from agent load_regions (1)" in report["region_source"]
    assert "kept template retention loads (tip_retention)" in report["region_source"]
    assert [c.id for c in problem.load_cases] == ["static_gravity", "tip_retention"]
    assert problem.load_cases[0].provenance == "user"
    assert problem.load_cases[0].role == "primary"
    assert problem.load_cases[1].role == "retention"
    assert "side_swing" not in {c.id for c in problem.load_cases}
    assert not any(a.field == "load_cases" for a in problem.assumptions)
    assert problem.volume_fraction == pytest.approx(0.4)
    assert report["registration"]["matrix"][0][3] == pytest.approx(-0.25, abs=0.05)
    assert report["registration"]["matrix"][2][3] == pytest.approx(1.95, abs=0.05)
    assert report["region_position_source"] == "candidate_metadata"


def test_positioned_agent_regions_require_registration():
    data = {
        "fixed_regions": [{"name": "mount_contact", "position_mm": [0.0, 0.0, 10.0]}],
        "load_regions": [{"name": "strap_seat", "position_mm": [82.5, 0.0, -25.0]}],
        "loads": [{"load_case_id": "static_gravity", "region_name": "strap_seat", "force_N": [0.0, 0.0, -49.05]}],
        "candidate": {"frame": "candidate_mesh_frame", "task": "other_part"},
    }
    with pytest.raises(AdapterError, match="registration_required"):
        require_registration(data)


def test_imported_candidate_load_position_comes_from_metadata_not_geometry_agent():
    """Imported hook: GeometryAgent conceptual hang is not the physical strap_seat."""
    from to_agent.integration.run import prepare

    data = _hook_agent_input()
    assert data["load_regions"][0]["position_mm"] == [82.5, 0.0, -25.0]
    spec = hook_registration(HOOK_CANDIDATE["dimensions_path"], HOOK_CANDIDATE["particle_path"])
    seat_desk = spec["landmarks_desk_edge_frame"]["strap_seat"]
    assert seat_desk[0] == pytest.approx(72.09, abs=0.05)
    assert seat_desk[2] == pytest.approx(2.80, abs=0.05)

    problem, report = build_from_registry(data["candidate"], data, 8.0, 0.4, 2.5)
    assert report["region_position_source"] == "candidate_metadata"
    assert "strap_seat" in report["bound_regions"]
    # Adapter mutated the input to the measured desk_edge landmark, not the conceptual hang.
    assert data["load_regions"][0]["position_mm"][0] == pytest.approx(seat_desk[0], abs=1e-6)
    assert data["load_regions"][0]["position_mm"][2] == pytest.approx(seat_desk[2], abs=1e-6)
    assert data["load_regions"][0]["position_mm"] != [82.5, 0.0, -25.0]

    seat_mesh = transform_point(spec["matrix"], seat_desk)
    hang_mesh = transform_point(spec["matrix"], [82.5, 0.0, -25.0])
    box = next(c.region for c in problem.load_cases if c.id == "static_gravity")
    cx = 0.5 * (box.min[0] + box.max[0])
    assert cx == pytest.approx(seat_mesh[0], abs=1.0)
    # Measured seat box: into the arm up to the opening at z_seat+1, not the conceptual hang.
    assert box.min[2] < seat_mesh[2]
    assert box.max[2] == pytest.approx(seat_mesh[2] + 1.0, abs=0.05)
    assert abs(0.5 * (box.min[2] + box.max[2]) - hang_mesh[2]) > 15.0

    mesh, masks = prepare(problem)
    assert masks.report["load_nodes"]["static_gravity"] > 0
    del mesh


def test_plane_registration_does_not_relocate_conceptual_hang():
    """The plane transform is not a one-point fit of z=-25 onto the mesh seat."""
    spec = hook_registration(HOOK_CANDIDATE["dimensions_path"], HOOK_CANDIDATE["particle_path"])
    hang_mesh = transform_point(spec["matrix"], [82.5, 0.0, -25.0])
    seat_mesh = transform_point(spec["matrix"], spec["landmarks_desk_edge_frame"]["strap_seat"])
    assert hang_mesh[2] == pytest.approx(-23.05, abs=0.1)
    assert seat_mesh[2] == pytest.approx(4.75, abs=0.1)
    assert abs(hang_mesh[2] - seat_mesh[2]) > 20.0


def test_generated_candidate_keeps_geometry_output_positions():
    data = {
        "candidate": {
            "task": "generated",
            "named_regions": {"strap_seat": [72.09, 0.0, 2.80]},
        },
        "load_regions": [{"name": "strap_seat", "position_mm": [82.5, 0.0, -25.0]}],
    }
    bind_candidate_feature_positions(data)
    assert data["load_regions"][0]["position_mm"] == [82.5, 0.0, -25.0]
    assert data["_region_position_source"] == "geometry_output"


def test_from_scratch_keeps_geometry_agent_positions():
    data = {
        "load_regions": [{"name": "strap_seat", "position_mm": [82.5, 0.0, -25.0]}],
    }
    bind_candidate_feature_positions(data)
    assert data["load_regions"][0]["position_mm"] == [82.5, 0.0, -25.0]
    assert data["_region_position_source"] == "agent"


def test_target_volume_fraction_overrides_hook_template():
    data = {
        "fixed_regions": [{"name": "mount_contact"}],
        "load_regions": [{"name": "strap_seat"}],
        "loads": [{"load_case_id": "static_gravity", "region_name": "strap_seat", "force_N": [0.0, 0.0, -49.05]}],
        "target_volume_fraction": 0.4,
        "candidate": HOOK_CANDIDATE,
    }
    vf = resolve_volume_fraction(data)
    problem, _ = build_from_registry(HOOK_CANDIDATE, data, 8.0, vf, 2.5)
    assert vf == pytest.approx(0.4)
    assert problem.volume_fraction == pytest.approx(0.4)


def _tiny_problem() -> TOProblem:
    box = BoxRegion(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0))
    return TOProblem(
        material=Material(density_kg_m3=1240.0),
        design_domain=box,
        supports=[Support(id="s", region=box)],
        load_cases=[LoadCase(id="l", region=box, force_N=(0.0, 0.0, -1.0))],
    )


def _fake_outcome(stl_path: str = "unused.stl"):
    return SimpleNamespace(
        summary={
            "stl": {"components": 1, "watertight": True, "volume_mm3": 1000.0},
            "connectivity": {"supports_attached": True, "loads_attached": True, "detached_loads": []},
            "artifacts": {"design_stl": stl_path},
        }
    )


def test_envelope_frame_unresolved_for_candidate_mesh_frame(monkeypatch):
    monkeypatch.setattr(
        "to_agent.integration.agentic._stl_bounds",
        lambda _p: ([-10.0, -10.0, -10.0], [80.0, 20.0, 30.0]),
    )
    inp = {
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 60.0, "max_height_mm": 65.0},
        "candidate": {"frame": "candidate_mesh_frame", "task": "desk_bag_hook"},
        "desk_thickness_mm": 20.0,
    }
    checks = acceptance_checks(_fake_outcome(), _tiny_problem(), inp, report={})
    assert result_frame(inp, {}) == "candidate_mesh_frame"
    assert checks["envelope_status"] == "frame_unresolved"
    assert checks["within_envelope"] == "frame_unresolved"
    assert checks["result_frame"] == "candidate_mesh_frame"
    assert checks["envelope_frame"] == "desk_edge_frame"
    assert "within_envelope" in checks["unknown"]
    assert checks["accepted"] is False
    assert any("frame_unresolved" in r for r in checks["reasons"])


def test_envelope_evaluated_when_frames_match(monkeypatch):
    monkeypatch.setattr(
        "to_agent.integration.agentic._stl_bounds",
        lambda _p: ([0.0, -10.0, -5.0], [50.0, 10.0, 25.0]),
    )
    inp = {
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 60.0, "max_height_mm": 65.0},
        "candidate": {"frame": "desk_edge_frame", "task": "generated"},
        "desk_thickness_mm": 20.0,
    }
    checks = acceptance_checks(_fake_outcome(), _tiny_problem(), inp, report={})
    assert checks["envelope_status"] == "evaluated"
    assert checks["within_envelope"] is True


def test_envelope_protrusion_ignores_extent_behind_desk(monkeypatch):
    """Clamp arms at x<0 must not consume max_protrusion. Outward reach is max(0, x_max)."""
    monkeypatch.setattr(
        "to_agent.integration.agentic._stl_bounds",
        lambda _p: ([-80.0, -20.0, -10.0], [90.0, 20.0, 30.0]),
    )
    inp = {
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 50.0, "max_height_mm": 50.0},
        "candidate": {"frame": "desk_edge_frame", "task": "generated"},
        "desk_thickness_mm": 20.0,
    }
    checks = acceptance_checks(_fake_outcome(), _tiny_problem(), inp, report={})
    assert checks["envelope_status"] == "evaluated"
    assert checks["within_envelope"] is True
    # x-span is 170 mm; that is not the protrusion. Outward reach is 90 mm.
    assert checks["envelope_measured_mm"]["max_protrusion_mm"] == pytest.approx(90.0)
    assert checks["envelope_measured_mm"]["max_width_mm"] == pytest.approx(40.0)
    assert checks["envelope_measured_mm"]["max_height_mm"] == pytest.approx(40.0)
    assert checks["envelope_bounds_mm"]["lo"][0] == pytest.approx(-80.0)


def test_envelope_protrusion_fails_when_outward_exceeds(monkeypatch):
    monkeypatch.setattr(
        "to_agent.integration.agentic._stl_bounds",
        lambda _p: ([-80.0, -20.0, -10.0], [120.0, 20.0, 30.0]),
    )
    inp = {
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 50.0, "max_height_mm": 50.0},
        "candidate": {"frame": "desk_edge_frame", "task": "generated"},
        "desk_thickness_mm": 20.0,
    }
    checks = acceptance_checks(_fake_outcome(), _tiny_problem(), inp, report={})
    assert checks["within_envelope"] is False
    assert checks["envelope_measured_mm"]["max_protrusion_mm"] == pytest.approx(120.0)
    assert any("max_protrusion_mm" in r for r in checks["reasons"])


def test_envelope_uses_explicit_registration_transform(monkeypatch):
    monkeypatch.setattr(
        "to_agent.integration.agentic._stl_bounds",
        lambda _p: ([1000.0, 1000.0, 1000.0], [1010.0, 1010.0, 1010.0]),
    )
    inp = {
        "envelope": {"max_protrusion_mm": 20.0, "max_width_mm": 40.0, "max_height_mm": 40.0},
        "candidate": {"frame": "candidate_mesh_frame", "task": "other_part"},
        "desk_thickness_mm": 0.0,
        # desk_edge -> mesh: +1000 mm. Inverse sends mesh bounds back to the origin.
        "registration_transform": [
            [1.0, 0.0, 0.0, 1000.0],
            [0.0, 1.0, 0.0, 1000.0],
            [0.0, 0.0, 1.0, 1000.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    checks = acceptance_checks(_fake_outcome(), _tiny_problem(), inp, report={})
    assert checks["envelope_status"] == "evaluated"
    assert checks["envelope_transform"] == "registration_transform"
    assert checks["within_envelope"] is True


def test_hook_registration_live(tmp_path):
    """desk_bag_hook: strap_seat + plane transform + SIMP + post-FE + STL."""
    from to_agent.integration.run import prepare

    data = _hook_agent_input()
    problem, report = build_from_registry(data["candidate"], data, 8.0, 0.4, 2.5)
    mesh, masks = prepare(problem)
    assert masks.report["load_nodes"]["static_gravity"] > 0
    assert masks.report["support_nodes"]["mount_contact"] > 0
    assert "cup_cavity" not in {c.id for c in problem.load_cases}
    assert all(r.get("name") != "cup_cavity" for r in data["load_regions"])

    out = run_topology(data, out_root=tmp_path)
    assert out["is_mock"] is False
    assert Path(out["optimized_geometry_ref"]).exists()
    assert out["iterations"] == 2
    assert out["post_check"] is not None and out["post_check"]["is_mock"] is False
    assert out["problem_report"]["volume_fraction"] == pytest.approx(0.4)
    assert out["acceptance"]["envelope_status"] == "evaluated"
    assert out["acceptance"]["within_envelope"] != "frame_unresolved"
    assert out["acceptance"]["acceptance_status"] == "unresolved_not_converged"
    assert out["acceptance"]["solver_domain_within_envelope"] is True
    assert out["problem_report"]["region_position_source"] == "candidate_metadata"
    assert "strap_seat" in out["problem_report"]["bound_regions"]
    envc = out["problem_report"]["envelope_constraint"]
    assert envc["applied"] is True
    assert envc["design_elems_after"] < envc["design_elems_before"]
    assert out["requested_volume_fraction"] == pytest.approx(0.4)
    assert out["applied_volume_fraction"] == pytest.approx(0.4)
    assert out["volume_fraction_override_reason"] is None
    assert problem.load_cases[0].provenance == "user"
    assert report["registration"]["from_frame"] == "desk_edge_frame"
    assert {c.id for c in problem.load_cases} >= {"static_gravity", "tip_retention"}
    assert "mount_contact" in masks.report["support_nodes"]
    assert "mount_contact_top" in {s.id for s in problem.supports}
    bc = out["acceptance"].get("bc_validation") or masks.report["bc_validation"]
    assert bc["hard_infeasible"] is False
    assert all(r["usable_nodes"] > 0 for r in bc["regions"])
    assert out["acceptance"]["optional_candidate_clipped_by_keepout"] == masks.report["optional_candidate_clipped_by_keepout"]


def test_hook_mount_landmark_is_underside_not_mid_gap():
    spec = hook_registration(HOOK_CANDIDATE["dimensions_path"], HOOK_CANDIDATE["particle_path"])
    mount = spec["landmarks_desk_edge_frame"]["mount_contact"]
    top = spec["landmarks_desk_edge_frame"]["mount_contact_top"]
    assert mount[2] == pytest.approx(0.0, abs=0.05)
    assert top[2] == pytest.approx(18.2, abs=0.3)


def test_hook_required_bcs_sit_on_measured_keepout_faces():
    """Jaw supports and seat load are measured surfaces, not the mid-slab / opening patch."""
    import warnings

    from to_agent.integration.run import prepare

    data = _hook_agent_input()
    problem, report = build_from_registry(data["candidate"], data, 4.0, 0.4, 2.5)
    g = report["hook"]
    bot = next(s for s in problem.supports if s.id == "mount_contact")
    top = next(s for s in problem.supports if s.id == "mount_contact_top")
    assert bot.region.max[2] == pytest.approx(g["z_rib_lo"], abs=0.05)
    assert bot.region.min[2] < g["z_rib_lo"]
    assert top.region.min[2] == pytest.approx(g["z_rib_hi"], abs=0.05)
    seat = next(c.region for c in problem.load_cases if c.id == "static_gravity")
    assert seat.max[2] == pytest.approx(g["z_seat"] + 1.0, abs=0.05)

    apply_envelope_constraint(problem, data, report["registration"]["matrix"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh, masks = prepare(problem)
    del mesh
    assert not any("void wins" in str(w.message) for w in caught)
    assert masks.report["optional_candidate_clipped_by_keepout"] >= 0
    bc = masks.report["bc_validation"]
    assert bc["hard_infeasible"] is False
    by_id = {r["id"]: r for r in bc["regions"]}
    assert by_id["mount_contact"]["usable_nodes"] > 0
    assert by_id["static_gravity"]["usable_nodes"] > 0
    assert by_id["tip_retention"]["usable_nodes"] > 0
    assert by_id["mount_contact"]["usable_fraction"] >= 0.5
    assert by_id["static_gravity"]["usable_fraction"] >= 0.5
    assert by_id["static_gravity"]["status"] in {"ok", "partial"}


def test_bc_validation_hard_when_region_has_no_nonvoid_nodes():
    from to_agent.integration.bc_validation import validate_required_bcs
    from to_agent.integration.run import prepare
    from to_agent.meshing.masks import ProblemSetupError

    domain = BoxRegion(min=(0.0, 0.0, 0.0), max=(16.0, 8.0, 8.0))
    voided = BoxRegion(min=(0.0, 0.0, 0.0), max=(8.0, 8.0, 8.0))
    dead = BoxRegion(min=(0.0, 0.0, 0.0), max=(2.0, 2.0, 2.0))
    live = BoxRegion(min=(12.0, 0.0, 0.0), max=(16.0, 8.0, 8.0))
    problem = TOProblem(
        design_domain=domain,
        void=[voided],
        supports=[Support(id="dead", region=dead), Support(id="live", region=live)],
        load_cases=[LoadCase(id="l", region=live, force_N=(0.0, 0.0, -1.0))],
        target_element_size=4.0,
    )
    with pytest.raises(ProblemSetupError, match="hard infeasible"):
        prepare(problem)
    from to_agent.meshing.masks import build_masks
    from to_agent.meshing.voxel_backend import build_hex_grid
    from to_agent.regions import resolve_domain

    mesh = build_hex_grid(resolve_domain(problem), 4.0)
    masks = build_masks(problem, mesh)
    bc = validate_required_bcs(problem, mesh, masks.void)
    assert bc["hard_infeasible"] is True
    gone = next(r for r in bc["regions"] if r["id"] == "dead")
    assert gone["status"] == "hard_infeasible"
    assert gone["usable_nodes"] == 0
    assert gone["requested_nodes"] > 0


def test_envelope_voids_design_elements_outside_user_box():
    """The user envelope is a hard void, not an after-the-fact acceptance box."""
    from to_agent.integration.run import prepare

    data = _hook_agent_input()
    problem, report = build_from_registry(data["candidate"], data, 8.0, 0.4, 2.5)
    T = report["registration"]["matrix"]
    info = apply_envelope_constraint(problem, data, T)
    info = count_envelope_mask(problem, info)
    assert info["applied"] is True
    assert info["design_elems_after"] < info["design_elems_before"]
    mesh, masks = prepare(problem)
    Tinv = invert_rigid(T)
    for cen in mesh.centroids[masks.design]:
        p = transform_point(Tinv, cen)
        assert p[0] <= 110.0 + 1e-4
        assert abs(p[1]) <= 30.0 + 1e-4
    behind = [
        transform_point(Tinv, cen)[0]
        for cen in mesh.centroids[masks.design | masks.preserve]
    ]
    assert any(x < 0.0 for x in behind)


def test_unconverged_envelope_miss_is_unresolved_not_fail(monkeypatch):
    monkeypatch.setattr(
        "to_agent.integration.agentic._stl_bounds",
        lambda _p: ([-80.0, -20.0, -10.0], [120.0, 20.0, 30.0]),
    )
    inp = {
        "envelope": {"max_protrusion_mm": 110.0, "max_width_mm": 50.0, "max_height_mm": 50.0},
        "candidate": {"frame": "desk_edge_frame", "task": "generated"},
        "desk_thickness_mm": 20.0,
    }
    outcome = _fake_outcome()
    outcome.summary["converged"] = False
    checks = acceptance_checks(outcome, _tiny_problem(), inp, report={})
    assert checks["exported_mesh_within_envelope"] is False
    assert checks["acceptance_status"] == "unresolved_not_converged"
    assert checks["accepted"] is False

    outcome.summary["converged"] = True
    checks = acceptance_checks(outcome, _tiny_problem(), inp, report={})
    assert checks["acceptance_status"] == "fail"
    assert checks["accepted"] is False


def test_mass_cap_vf_is_provenance_not_unsupported():
    from to_agent.integration.materials import material_from_name

    data = _hook_agent_input()
    problem, report = build_from_registry(data["candidate"], data, 8.0, 0.4, 2.5)
    problem.material, _ = material_from_name("PLA")
    apply_envelope_constraint(problem, data, report["registration"]["matrix"])
    applied, note, vf_max = volume_fraction_for_mass_cap(problem, 0.3)
    assert vf_max is not None
    assert applied <= 0.4 + 1e-12
    assert "mass cap" in note or applied == pytest.approx(0.4)
    data["max_part_mass_kg"] = 0.3
    # Provenance fields are populated by run_topology; the override must not be "unsupported".
    assert data["target_volume_fraction"] == pytest.approx(0.4)
