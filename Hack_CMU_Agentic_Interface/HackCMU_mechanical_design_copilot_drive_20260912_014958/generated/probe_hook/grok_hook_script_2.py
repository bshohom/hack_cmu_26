import numpy as np
import trimesh
from trimesh.creation import box

PARAMS = {
    "desk_thickness_mm": 25.0,
    "clamp_gap_mm": 27.0,
    "clamp_depth_mm": 30.0,
    "clamp_width_mm": 25.0,
    "clamp_thickness_mm": 8.0,
    "vertical_drop_mm": 40.0,
    "arm_length_mm": 100.0,
    "arm_width_mm": 20.0,
    "arm_thickness_mm": 8.0,
    "tip_height_mm": 12.0,
    "wall_thickness_mm": 5.0,
}

def build(params):
    d = params
    dt = d["desk_thickness_mm"]
    cg = d["clamp_gap_mm"]
    cd = d["clamp_depth_mm"]
    cw = d["clamp_width_mm"]
    ct = d["clamp_thickness_mm"]
    vd = d["vertical_drop_mm"]
    al = d["arm_length_mm"]
    aw = d["arm_width_mm"]
    at = d["arm_thickness_mm"]
    th = d["tip_height_mm"]
    wt = d["wall_thickness_mm"]

    meshes = []

    # Upper clamp arm (overlaps vertical)
    upper = box(extents=[cd, cw, ct], transform=trimesh.transformations.translation_matrix([-cd/2 + 2, 0, dt + ct/2]))
    meshes.append(upper)

    # Lower clamp arm (overlaps vertical)
    lower = box(extents=[cd, cw, ct], transform=trimesh.transformations.translation_matrix([-cd/2 + 2, 0, -ct/2]))
    meshes.append(lower)

    # Vertical connector at edge (overlaps both clamps and descent)
    vert = box(extents=[wt + 4, cw, dt + cg + ct + 4], transform=trimesh.transformations.translation_matrix([-wt/2 + 2, 0, (dt + cg)/2 - ct/2]))
    meshes.append(vert)

    # Descent arm (overlaps vertical)
    descent = box(extents=[wt, aw, vd + 4], transform=trimesh.transformations.translation_matrix([-wt/2, 0, -vd/2 - ct - 2]))
    meshes.append(descent)

    # Horizontal arm (overlaps descent)
    horiz = box(extents=[al + 4, aw, at], transform=trimesh.transformations.translation_matrix([al/2 + 2, 0, -vd - at/2 - ct - 2]))
    meshes.append(horiz)

    # Raised tip (overlaps horiz)
    tip = box(extents=[wt + 2, aw, th], transform=trimesh.transformations.translation_matrix([al - wt/2 + 1, 0, -vd - at - th/2 - ct - 2]))
    meshes.append(tip)

    part = trimesh.boolean.union(meshes, engine="manifold")
    return part

DIMENSIONS = {
    "Protrusion": 100.0,
    "Clamp gap": 27.0,
    "Hook arm length": 100.0,
    "Total height below desk": 60.0,
}

REGIONS = {
    "load": {"min": [40.0, -8.0, -55.0], "max": [70.0, 8.0, -53.0]},
    "mounts": [
        {"name": "upper", "box": {"min": [-28.0, -12.5, 25.0], "max": [-5.0, 12.5, 33.0]}},
        {"name": "lower", "box": {"min": [-28.0, -12.5, -8.0], "max": [-5.0, 12.5, 0.0]}},
    ],
    "keep_out": [
        {"name": "throat", "box": {"min": [0.0, -9.0, -55.0], "max": [85.0, 9.0, -47.0]}},
    ],
}

NOTES = [
    "Hook for 5kg bag strap under desk edge.",
    "Clamp gap 27mm for 25mm desk.",
    "Horizontal arm 100mm out with raised tip.",
    "All walls >=5mm for FDM PLA.",
]
