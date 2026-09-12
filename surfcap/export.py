"""target.ply / target.glb / target.json export, per plan (e).

Input point clouds/surfaces are assumed already in the world frame: metres,
Z-up, table-top at z ~= 0 (per surfcap conventions, see surfcap/types.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from surfcap.types import Surface, Target

_ROLE_COLOURS = {
    "top": (60, 180, 75, 128),
    "front": (60, 120, 220, 128),
    "side": (240, 150, 30, 128),
    "bottom": (150, 60, 200, 128),
    "other": (150, 150, 150, 128),
}

_MAX_GLB_BYTES = 25 * 1024 * 1024


# --------------------------------------------------------------------------
# PLY
# --------------------------------------------------------------------------

def write_ply(pcd: o3d.geometry.PointCloud, path) -> int:
    """Write `pcd` (with colours, if present) to `path` as PLY. Returns n_points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    return int(np.asarray(pcd.points).shape[0])


# --------------------------------------------------------------------------
# GLB helpers
# --------------------------------------------------------------------------

def _cloud_points_geom(pcd: o3d.geometry.PointCloud) -> trimesh.points.PointCloud:
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pcd.has_colors():
        colours = np.asarray(pcd.colors, dtype=np.float64)
        colours_u8 = np.clip(colours * 255.0, 0, 255).astype(np.uint8)
        colours_u8 = np.concatenate(
            [colours_u8, np.full((len(colours_u8), 1), 255, dtype=np.uint8)], axis=1
        )
        return trimesh.points.PointCloud(vertices=pts, colors=colours_u8)
    return trimesh.points.PointCloud(vertices=pts)


def _downsampled_box_mesh(
    pcd: o3d.geometry.PointCloud, max_cloud_pts: int, box_size: float = 0.002
) -> "trimesh.Trimesh | None":
    """Tiny boxes at (a subsample of) the cloud's points, merged into one mesh.

    Point primitives can be invisible in some three.js-based GLB viewers, so
    this gives every viewer *something* visibly solid to render.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return None
    if n > max_cloud_pts:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_cloud_pts, replace=False)
        pts = pts[idx]
    else:
        idx = np.arange(n)

    if pcd.has_colors():
        colours = np.asarray(pcd.colors, dtype=np.float64)[idx]
        colours_u8 = np.clip(colours * 255.0, 0, 255).astype(np.uint8)
    else:
        colours_u8 = None

    boxes = []
    for i, p in enumerate(pts):
        b = trimesh.creation.box(extents=(box_size, box_size, box_size))
        b.apply_translation(p)
        if colours_u8 is not None:
            c = colours_u8[i]
            b.visual.face_colors = np.tile(
                np.concatenate([c, [255]]), (len(b.faces), 1)
            )
        boxes.append(b)
    return trimesh.util.concatenate(boxes)


def _surface_mesh(surf: Surface, thickness: float = 0.002) -> "trimesh.Trimesh | None":
    """A thin extrusion of `surf.polygon_3d` in its own plane, coloured by role."""
    poly3d = np.asarray(surf.polygon_3d, dtype=np.float64)
    if len(poly3d) < 3:
        return None

    normal = np.asarray(surf.normal, dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    if norm_len == 0:
        return None
    normal = normal / norm_len

    centroid = poly3d.mean(axis=0)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(tmp, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    poly2d = np.stack(
        [(poly3d - centroid) @ u, (poly3d - centroid) @ v], axis=1
    )

    mesh = None
    try:
        import shapely.geometry as shg

        polygon = shg.Polygon(poly2d)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_valid and polygon.area > 0:
            mesh = trimesh.creation.extrude_polygon(polygon, height=thickness)
            mesh.apply_translation([0.0, 0.0, -thickness / 2.0])
    except Exception:
        mesh = None

    if mesh is None:
        # fan-triangulate the (assumed convex) 2D polygon into a two-sided
        # thin mesh ourselves.
        k = len(poly2d)
        verts2d_top = poly2d.copy()
        verts2d_bot = poly2d.copy()
        verts = np.concatenate(
            [
                np.concatenate([verts2d_top, np.full((k, 1), thickness / 2.0)], axis=1),
                np.concatenate([verts2d_bot, np.full((k, 1), -thickness / 2.0)], axis=1),
            ],
            axis=0,
        )
        faces = []
        for i in range(1, k - 1):
            faces.append([0, i, i + 1])  # top, CCW
            faces.append([k, k + i + 1, k + i])  # bottom, CW (flipped)
        # side walls
        for i in range(k):
            j = (i + 1) % k
            faces.append([i, j, k + j])
            faces.append([i, k + j, k + i])
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)

    # transform from local (u, v, n) plane coords back to world
    R = np.stack([u, v, normal], axis=1)  # columns u,v,n
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centroid
    mesh.apply_transform(T)

    colour = _ROLE_COLOURS.get(surf.role, _ROLE_COLOURS["other"])
    mesh.visual.face_colors = np.tile(colour, (len(mesh.faces), 1)).astype(np.uint8)
    return mesh


def _z_arrow_mesh(length: float = 0.15) -> trimesh.Trimesh:
    """A red +Z arrow (cylinder shaft + cone head) at the world origin."""
    shaft_len = length * 0.8
    head_len = length * 0.2
    shaft_r = length * 0.02
    head_r = length * 0.05

    shaft = trimesh.creation.cylinder(radius=shaft_r, height=shaft_len, sections=16)
    shaft.apply_translation([0.0, 0.0, shaft_len / 2.0])

    head = trimesh.creation.cone(radius=head_r, height=head_len, sections=16)
    head.apply_translation([0.0, 0.0, shaft_len])

    arrow = trimesh.util.concatenate([shaft, head])
    arrow.visual.face_colors = np.tile([220, 30, 30, 255], (len(arrow.faces), 1)).astype(np.uint8)
    return arrow


def write_glb(pcd, surfaces: list, path, max_cloud_pts: int = 30000, arrow_len: float = 0.15) -> None:
    """Export `pcd` + `surfaces` + a Z arrow as a GLB via trimesh.Scene."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    scene.add_geometry(_cloud_points_geom(pcd), node_name="cloud_points")

    box_mesh = _downsampled_box_mesh(pcd, max_cloud_pts=max_cloud_pts)
    if box_mesh is not None:
        # tentatively include it; drop later if the exported file is too big
        scene.add_geometry(box_mesh, node_name="cloud_boxes")

    for i, surf in enumerate(surfaces):
        mesh = _surface_mesh(surf)
        if mesh is not None:
            scene.add_geometry(mesh, node_name=f"surface_{i}_{surf.role}")

    scene.add_geometry(_z_arrow_mesh(length=arrow_len), node_name="z_arrow")

    glb_bytes = scene.export(file_type="glb")
    if box_mesh is not None and len(glb_bytes) >= _MAX_GLB_BYTES:
        # drop the box representation and re-export with points only
        scene.delete_geometry("cloud_boxes")
        glb_bytes = scene.export(file_type="glb")

    with open(path, "wb") as f:
        f.write(glb_bytes)

    # sanity check: the file must be reloadable
    trimesh.load(str(path))


# --------------------------------------------------------------------------
# target.json
# --------------------------------------------------------------------------

def build_target(
    frame: dict,
    scale: dict,
    ply_rel: str,
    n_points: int,
    surfaces,
    obb,
    warnings: list,
    context: dict = None,
) -> Target:
    return Target(
        frame=frame,
        scale=scale,
        cloud={"ply": str(ply_rel), "n_points": int(n_points)},
        surfaces=list(surfaces),
        obb=obb,
        context=context if context is not None else {"floor_plane": None},
        warnings=list(warnings),
    )


def export_all(
    out_dir,
    pcd,
    surfaces,
    obb,
    frame: dict,
    scale: dict,
    warnings: list,
    context: dict = None,
) -> Target:
    """Write target.ply, target.glb, target.json into out_dir; return the Target."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_dir / "target.ply"
    glb_path = out_dir / "target.glb"
    json_path = out_dir / "target.json"

    n_points = write_ply(pcd, ply_path)
    write_glb(pcd, surfaces, glb_path)

    target = build_target(
        frame=frame,
        scale=scale,
        ply_rel=ply_path.name,
        n_points=n_points,
        surfaces=surfaces,
        obb=obb,
        warnings=warnings,
        context=context,
    )
    target.save(json_path)
    return target
