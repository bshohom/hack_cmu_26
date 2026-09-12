import numpy as np
import trimesh
import math
PARAMS = {
    "desk_thickness": 25.0,
    "clamp_gap": 26.5,
    "bottle_dia": 85.0,
    "holder_id": 89.0,
    "holder_wall": 4.5,
    "holder_height": 55.0,
    "protrusion": 130.0,
    "clamp_width": 48.0,
    "clamp_depth": 42.0,
    "arm_thickness": 6.5,
    "support_thickness": 5.0
}
def build(params):
    dt = params["desk_thickness"]
    cg = params["clamp_gap"]
    hid = params["holder_id"]
    hw = params["holder_wall"]
    hh = params["holder_height"]
    pr = params["protrusion"]
    cw = params["clamp_width"]
    cd = params["clamp_depth"]
    at = params["arm_thickness"]
    st = params["support_thickness"]
    hod = hid + 2 * hw
    top_arm = trimesh.creation.box(extents=[cd, cw, at], transform=trimesh.transformations.translation_matrix([-cd/2, 0, dt + cg - at/2]))
    bot_arm = trimesh.creation.box(extents=[cd, cw, at], transform=trimesh.transformations.translation_matrix([-cd/2, 0, -at/2]))
    back = trimesh.creation.box(extents=[at, cw, dt + cg + at], transform=trimesh.transformations.translation_matrix([-at/2, 0, (dt + cg)/2]))
    holder_outer = trimesh.creation.cylinder(radius=hod/2, height=hh, sections=64, transform=trimesh.transformations.translation_matrix([pr - hod/2, 0, hh/2]))
    holder_inner = trimesh.creation.cylinder(radius=hid/2, height=hh + 2, sections=64, transform=trimesh.transformations.translation_matrix([pr - hod/2, 0, hh/2]))
    floor = trimesh.creation.cylinder(radius=hod/2, height=st, sections=64, transform=trimesh.transformations.translation_matrix([pr - hod/2, 0, st/2]))
    bridge = trimesh.creation.box(extents=[pr - hod/2 + hod/2 - cd + at, cw, st], transform=trimesh.transformations.translation_matrix([(pr - hod/2 + hod/2 - cd + at)/2, 0, st/2]))
    body = trimesh.boolean.union([top_arm, bot_arm, back, holder_outer, floor, bridge], engine="manifold")
    part = trimesh.boolean.difference([body, holder_inner], engine="manifold")
    return part
DIMENSIONS = {
    "Holder inner diameter": 89.0,
    "Holder height": 55.0,
    "Max protrusion": 130.0,
    "Clamp gap": 26.5,
    "Wall thickness": 4.5
}
REGIONS = {
    "load": {"min": [85.5, -22.25, 0.0], "max": [174.5, 22.25, 4.5]},
    "mounts": [
        {"name": "top_clamp", "min": [-42.0, -24.0, 25.0], "max": [-6.5, 24.0, 31.5]},
        {"name": "bottom_clamp", "min": [-42.0, -24.0, -6.5], "max": [-6.5, 24.0, 0.0]}
    ],
    "keep_out": [
        {"name": "bottle_cavity", "min": [85.5, -44.5, 0.0], "max": [174.5, 44.5, 55.0]}
    ]
}
NOTES = ["Clamp gap 26.5mm for 25mm desk + clearance", "4.5mm walls for PLA strength", "Single connected watertight body"]
