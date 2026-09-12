import numpy as np
import trimesh
PARAMS = {
    "desk_thickness": 25.0,
    "bottle_diameter": 85.0,
    "holder_clearance": 3.0,
    "holder_height": 60.0,
    "wall": 5.0,
    "clamp_depth": 40.0,
    "clamp_height": 30.0,
    "protrusion": 120.0,
    "gap_extra": 2.0
}
def build(params):
    d = {**PARAMS, **params}
    r_in = d["bottle_diameter"] / 2 + d["holder_clearance"]
    r_out = r_in + d["wall"]
    h_hold = d["holder_height"]
    wall = d["wall"]
    desk_t = d["desk_thickness"]
    gap = desk_t + d["gap_extra"]
    clamp_d = d["clamp_depth"]
    clamp_h = d["clamp_height"]
    prot = d["protrusion"]
    # holder cylinder
    holder = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=h_hold,
        transform=trimesh.transformations.translation_matrix([prot - r_out, 0, 0]))
    # holder bottom
    bottom = trimesh.creation.cylinder(radius=r_out, height=wall,
        transform=trimesh.transformations.translation_matrix([prot - r_out, 0, -wall]))
    # clamp top arm (x<0)
    top_arm = trimesh.creation.box(extents=[clamp_d, r_out*2, wall],
        transform=trimesh.transformations.translation_matrix([-clamp_d/2, 0, desk_t + wall]))
    # clamp bottom arm
    bot_arm = trimesh.creation.box(extents=[clamp_d, r_out*2, wall],
        transform=trimesh.transformations.translation_matrix([-clamp_d/2, 0, -gap - wall]))
    # vertical connector at edge
    connector = trimesh.creation.box(extents=[wall, r_out*2, desk_t + gap + 2*wall],
        transform=trimesh.transformations.translation_matrix([-wall/2, 0, (desk_t - gap)/2]))
    # join holder to connector
    join = trimesh.creation.box(extents=[prot - r_out + wall/2, wall, wall],
        transform=trimesh.transformations.translation_matrix([(prot - r_out - wall/2)/2, 0, 0]))
    parts = [holder, bottom, top_arm, bot_arm, connector, join]
    mesh = trimesh.boolean.union(parts, engine="manifold")
    return mesh
DIMENSIONS = {"Holder ID": 91.0, "Protrusion": 120.0, "Clamp gap": 27.0, "Wall thickness": 5.0}
REGIONS = {
    "load": {"min": [75.0, -45.5, -5.0], "max": [115.0, 45.5, 0.0]},
    "mounts": [
        {"name": "top", "min": [-40.0, -45.5, 25.0], "max": [0.0, 45.5, 30.0]},
        {"name": "bottom", "min": [-40.0, -45.5, -27.0], "max": [0.0, 45.5, -22.0]}
    ],
    "keep_out": [
        {"name": "bottle", "min": [75.0, -42.5, 0.0], "max": [115.0, 42.5, 60.0]},
        {"name": "desk", "min": [-40.0, -50.0, 0.0], "max": [0.0, 50.0, 25.0]}
    ]
}
NOTES = ["FDM PLA, >=4mm walls, 1.1kg bottle, desk 25mm"]
