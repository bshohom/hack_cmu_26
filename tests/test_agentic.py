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


def test_no_candidate_raises():
    with pytest.raises(AdapterError, match="no candidate"):
        run_topology(topology_input(None))


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
