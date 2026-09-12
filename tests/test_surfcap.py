import json
from pathlib import Path

from to_agent.ingest.surfcap import target_to_measurements

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "HackCMU" / "examples" / "target.example.json"


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
