import json
from pathlib import Path

from to_agent.ingest.surfcap import measurements_from_path, target_to_measurements

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "HackCMU" / "examples" / "target.example.json"
SCENE_DESK = ROOT / "HackCMU" / "Generated_Scene_meshes" / "desk.ply"


def test_scene_mesh_thickness():
    if not SCENE_DESK.exists():
        return
    m = measurements_from_path(SCENE_DESK)
    assert m["units"] == "mm" and m["thickness_source"].startswith("slab")
    assert 15.0 <= m["desk_thickness_mm"] <= 40.0
    assert 600 <= m["mount_extent_mm"][0] <= 700
    # surfcap closes slabs by mirroring the top face at a supplied/default thickness, so
    # the gap between the two faces is the input echoed back, not a measurement. It must
    # not pre-fill until surfcap reports that a bottom plane was actually observed.
    assert m["thickness_provenance"] == "assumed"
    assert m["prefill"] is False
    assert any("not measured" in n for n in m["notes"])


def test_scene_mesh_with_observed_underside_may_prefill():
    """Once surfcap reports a real bottom plane, the same number becomes usable."""
    if not SCENE_DESK.exists():
        return
    from to_agent.ingest.surfcap import scene_mesh_to_measurements

    m = scene_mesh_to_measurements(SCENE_DESK, mesh_stats={"mirrored_underside": False})
    assert m["thickness_provenance"] == "observed"
    assert m["prefill"] is True


def test_default_thickness_never_prefills(tmp_path):
    """The audit's reproduction: thickness_source='default' scored confidence 1.0."""
    target = {
        "frame": {"units": "m"},
        "scale": {"reliable": True, "rms_mm": 0.9, "n_views_with_ref": 9},
        "surfaces": [],
        "cloud": {"postprocess": {"thickness_m": 0.018, "thickness_source": "default"}},
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(target))
    m = target_to_measurements(p)
    assert m["desk_thickness_mm"] == 18.0
    assert m["thickness_provenance"] == "assumed"
    assert m["prefill"] is False


def test_example_target_is_confident():
    if not EXAMPLE.exists():
        return  # CV repo not checked out
    m = target_to_measurements(EXAMPLE)
    assert m["units"] == "mm"
    assert m["desk_thickness_mm"] == 30.0  # front face extent 0.03 m
    assert m["confidence"] >= 0.7 and m["prefill"] is True
    assert m["mount_extent_mm"] == [1200.0, 700.0]
    assert m["front_edge_mm"]["distance_from_card_mm"] == 350.0


def test_relative_units_are_unusable(tmp_path):
    target = {"frame": {"units": "relative"}, "scale": {"reliable": False}, "surfaces": [], "obb": {"extents_m": [1, 1, 0.02]}}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(target))
    m = target_to_measurements(p)
    assert m["confidence"] == 0.0 and m["prefill"] is False
