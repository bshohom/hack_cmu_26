from pathlib import Path

import pytest

from to_agent.integration.agentic import AdapterError, run_topology

ROOT = Path(__file__).resolve().parent.parent


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
    assert "requested volume fraction 0.4" in out["notes"]
    assert 0 < out["volume_fraction"] < 1
    pc = out["post_check"]
    assert pc is not None and pc["is_mock"] is False and pc["is_safety_validation"] is False
    assert pc["max_displacement_mm"] > 0 and pc["max_stress_pa"] > 0
    assert pc["factor_of_safety"] is not None and pc["factor_of_safety"] > 0
