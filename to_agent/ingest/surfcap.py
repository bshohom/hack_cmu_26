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
MESH_SUFFIXES = {".ply", ".stl", ".obj", ".glb"}

# Where a thickness number actually came from. surfcap can synthesize an underside from a
# supplied or default thickness and then mesh it watertight, so "the pipeline produced a
# number" says nothing about whether anything was measured. Only an observed or explicitly
# entered thickness may pre-fill a question; a synthesized one must be asked about.
#   observed  - measured from surfaces the reconstruction actually saw
#   user      - supplied by a person (CLI --thickness)
#   assumed   - a default, an OBB over-estimate, or a mirrored/invented underside
_THICKNESS_PROVENANCE = {
    "top-bottom planes": "observed",
    "front face extent": "observed",
    "slab top/bottom faces": "observed",
    "arg": "user",
    "default": "assumed",
    "obb": "assumed",
    "obb_estimate": "assumed",
    "z extent": "assumed",
    "postprocess": "assumed",  # source not stated by surfcap: not evidence of observation
    "none": "assumed",
}


def thickness_provenance(source: str | None) -> str:
    """Unknown sources are 'assumed': absence of provenance is not provenance."""
    return _THICKNESS_PROVENANCE.get((source or "none").strip().lower(), "assumed")


def measurements_from_path(path: str | Path) -> dict[str, Any]:
    """Dispatch: surfcap target.json, or a registered scene mesh (metres, Z-up, mount face at z≈0)."""
    path = Path(path)
    if path.suffix.lower() in MESH_SUFFIXES:
        return scene_mesh_to_measurements(path)
    return target_to_measurements(path)


def scene_mesh_to_measurements(path: str | Path, mesh_stats: dict | None = None) -> dict[str, Any]:
    """Desk/table thickness and extent from a plane-fitted slab mesh (surfcap Generated_Scene_meshes).

    Thickness = median height of upward-facing faces minus median height of downward-facing
    faces; a slab needs both. Units are assumed metres when the extent is < 5 (surfcap
    convention), else mm. Confidence is lower than a target.json because there is no scale
    metadata: 0.75 for a watertight slab with both faces, 0.4 otherwise.
    """
    import trimesh

    path = Path(path)
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"{path} has no faces")
    lo, hi = mesh.bounds
    scale = 1000.0 if float(max(hi - lo)) < 5.0 else 1.0  # metres -> mm
    normals = mesh.face_normals
    centroids = mesh.triangles_center * scale
    up = normals[:, 2] > 0.9
    down = normals[:, 2] < -0.9
    notes: list[str] = []
    thickness = None
    if up.sum() >= 3 and down.sum() >= 3:
        z_top = float(np.median(centroids[up, 2]))
        z_bot = float(np.median(centroids[down, 2]))
        thickness = z_top - z_bot
        source = "slab top/bottom faces"
    else:
        thickness = float((hi - lo)[2] * scale)
        source = "z extent"
        notes.append("no clear top/bottom face pair; thickness from z extent (upper bound)")
    ok = thickness is not None and 3.0 <= thickness <= 120.0
    # A scene mesh carries no provenance metadata, and surfcap currently closes slabs by
    # mirroring the top face at a supplied or default thickness. Measuring the gap between
    # those two faces then just returns the number that was used to build them, so this is
    # only an observation once surfcap reports that a bottom plane was actually seen.
    # (Handed to the surfcap owner: emit `mirrored_underside` in the postprocess stats.)
    observed_bottom = bool(mesh_stats and mesh_stats.get("mirrored_underside") is False)
    provenance = thickness_provenance(source) if observed_bottom else "assumed"
    confidence = 0.75 if (ok and mesh.is_watertight and provenance == "observed") else 0.4
    if not ok:
        notes.append(f"thickness {thickness:.1f} mm is not a plausible mounting slab")
        confidence = 0.0
    if not mesh.is_watertight:
        notes.append("mesh not watertight")
    if provenance != "observed":
        notes.append(
            f"thickness is {provenance}, not measured ({source}); it will not pre-fill"
        )
    ext = (hi - lo) * scale
    return {
        "source": "registration",
        "target_json": str(path),
        "units": "mm",
        "confidence": confidence,
        # A watertight mesh proves nothing when the underside was synthesized to make it so.
        "prefill": bool(ok and confidence >= CONFIDENCE_PREFILL and provenance in ("observed", "user")),
        "desk_thickness_mm": round(float(thickness), 1) if thickness is not None else None,
        "thickness_source": source,
        "thickness_provenance": provenance,
        "mount_normal": [0.0, 0.0, 1.0],
        "mount_extent_mm": [round(float(ext[0]), 1), round(float(ext[1]), 1)],
        "front_edge_mm": None,
        "frame": {"origin_desc": "surfcap scene mesh (metres, Z-up, mount face near z=0)", "up": [0, 0, 1],
                  "note": "scene mesh frame; not the candidate mesh frame"},
        "notes": notes,
    }


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

    provenance = thickness_provenance(thickness_source)
    if thickness_mm is not None and provenance != "observed":
        reasons.append(
            f"thickness is {provenance}, not measured (source: {thickness_source}); "
            "it will not pre-fill"
        )
    out: dict[str, Any] = {
        "source": "registration",
        "target_json": str(path),
        "units": "mm",
        "confidence": conf if thickness_mm is not None else 0.0,
        # Confidence alone is not enough: a default thickness scored 1.0 and pre-filled a
        # number nobody measured. Provenance gates the pre-fill.
        "prefill": (
            thickness_mm is not None
            and conf >= CONFIDENCE_PREFILL
            and provenance in ("observed", "user")
        ),
        "desk_thickness_mm": round(thickness_mm, 2) if thickness_mm is not None else None,
        "thickness_source": thickness_source,
        "thickness_provenance": provenance,
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
