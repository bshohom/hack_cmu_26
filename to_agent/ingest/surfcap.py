"""Bridge from the surfcap registration output (HackCMU/target.json) to design measurements.

surfcap: units metres, Z-up, origin = reference-card centre on the mount plane. We convert the
few scalars the design workflow needs to mm and attach a confidence so the agent can decide
whether to pre-fill a question with the registered value or ask the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CONFIDENCE_PREFILL = 0.7  # >= this: pre-fill the question with the registered value


def _confidence(target: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    frame = target.get("frame", {})
    scale = target.get("scale", {})
    if frame.get("units") != "m":
        return 0.0, ["frame.units is not metres (relative scale); measurements unusable"]
    conf = 1.0
    if not scale.get("reliable", False):
        conf *= 0.4
        reasons.append("scale.reliable is false")
    rms = scale.get("rms_mm")
    if rms is not None:
        if rms > 3.0:
            conf *= 0.6
            reasons.append(f"scale rms {rms} mm > 3 mm")
        elif rms > 1.5:
            conf *= 0.85
            reasons.append(f"scale rms {rms} mm")
    n_views = scale.get("n_views_with_ref", 0)
    if n_views < 3:
        conf *= 0.6
        reasons.append(f"reference seen in only {n_views} views")
    src = (target.get("cloud", {}).get("postprocess") or {}).get("thickness_source")
    if src in ("obb", "obb_estimate"):
        conf *= 0.7
        reasons.append("thickness from OBB (over-estimates)")
    return round(conf, 3), reasons


def target_to_measurements(path: str | Path) -> dict[str, Any]:
    """Return {desk_thickness_mm, mount_normal, mount_extent_mm, front_edge_mm, confidence, source, notes}."""
    target = json.loads(Path(path).read_text())
    conf, reasons = _confidence(target)
    surfaces = target.get("surfaces", [])
    top = next((s for s in surfaces if s.get("role") == "top"), None)
    front = next((s for s in surfaces if s.get("role") == "front"), None)
    bottom = next((s for s in surfaces if s.get("role") == "bottom"), None)

    thickness_mm = None
    thickness_source = "none"
    post = (target.get("cloud", {}).get("postprocess") or {})
    if post.get("thickness_m"):
        thickness_mm = float(post["thickness_m"]) * 1000.0
        thickness_source = post.get("thickness_source", "postprocess")
    elif top is not None and bottom is not None:
        thickness_mm = abs(float(top["centroid"][2]) - float(bottom["centroid"][2])) * 1000.0
        thickness_source = "top-bottom planes"
    elif front is not None and front.get("extent_m"):
        thickness_mm = float(min(front["extent_m"])) * 1000.0
        thickness_source = "front face extent"
    elif target.get("obb", {}).get("extents_m"):
        thickness_mm = float(min(target["obb"]["extents_m"])) * 1000.0
        thickness_source = "obb"
        conf = round(conf * 0.7, 3)
        reasons.append("thickness from OBB minor extent")

    out: dict[str, Any] = {
        "source": "registration",
        "target_json": str(path),
        "units": "mm",
        "confidence": conf if thickness_mm is not None else 0.0,
        "prefill": thickness_mm is not None and conf >= CONFIDENCE_PREFILL,
        "desk_thickness_mm": round(thickness_mm, 2) if thickness_mm is not None else None,
        "thickness_source": thickness_source,
        "mount_normal": top.get("normal") if top else None,
        "mount_extent_mm": [round(float(v) * 1000.0, 1) for v in top["extent_m"]] if top and top.get("extent_m") else None,
        "front_edge_mm": None,
        "frame": {
            "origin_desc": target.get("frame", {}).get("origin_desc"),
            "up": target.get("frame", {}).get("up"),
            "note": "surfcap frame: origin at the reference card on the mount plane; not the candidate mesh frame",
        },
        "notes": reasons + list(target.get("warnings", [])),
    }
    if front is not None and front.get("centroid") is not None:
        c = np.asarray(front["centroid"], float) * 1000.0
        n = np.asarray(front.get("normal", [0, 0, 0]), float)
        out["front_edge_mm"] = {"point": np.round(c, 1).tolist(), "outward_normal": n.tolist(),
                                "distance_from_card_mm": round(float(abs(np.dot(c, n))), 1)}
    return out
