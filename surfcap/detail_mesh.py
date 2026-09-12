"""D1: object-only, coherent, detailed mesh from a raw TSDF fusion mesh.

The whole-scene TSDF mesh (see :mod:`surfcap.recon`, ``--recon-mode sfm_tsdf``)
covers the whole room and is riddled with thousands of tiny disconnected
fragments (floating debris from noisy depth/SfM fusion). This module crops it
down to the target object using its oriented bounding box (``target.json``'s
``obb``), removes debris, fills small holes, smooths lightly and decimates it
to a size a viewer can actually render.

CLI:
    python -m surfcap.detail_mesh <tsdf_mesh.ply> <target.json> <out_dir>

Frame: input/output meshes are all in the surfcap world frame (metres, Z-up,
credit-card centre at the origin; see ``surfcap/types.py``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------
# 1. crop to the object's OBB
# --------------------------------------------------------------------------

def crop_to_object(
    mesh: o3d.geometry.TriangleMesh,
    obb: dict,
    pad_m: float = 0.03,
) -> o3d.geometry.TriangleMesh:
    """Keep only triangles whose centroid lies inside ``obb`` expanded by ``pad_m``.

    ``obb`` is ``{"centre": [x,y,z], "axes": [[...],[...],[...]], "extents_m": [ex,ey,ez]}``
    with ``axes`` rows as the box's (orthonormal) local axes, matching the
    convention written by :mod:`surfcap.recon` / ``target.json``.
    """
    centre = np.asarray(obb["centre"], dtype=np.float64).reshape(3)
    axes = np.asarray(obb["axes"], dtype=np.float64).reshape(3, 3)
    extents = np.asarray(obb["extents_m"], dtype=np.float64).reshape(3)

    box = o3d.geometry.OrientedBoundingBox(
        center=centre,
        R=axes.T,  # o3d expects columns = axis directions
        extent=extents + 2.0 * float(pad_m),
    )

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles)
    if len(f) == 0 or len(v) == 0:
        return o3d.geometry.TriangleMesh()

    tri_centroids = v[f].mean(axis=1)
    idx = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(tri_centroids))
    keep = np.zeros(len(f), dtype=bool)
    keep[np.asarray(idx, dtype=np.int64)] = True

    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(~keep)
    out.remove_unreferenced_vertices()
    return out


# --------------------------------------------------------------------------
# 2. clean shell: dedup/degenerate, small components, holes, smooth
# --------------------------------------------------------------------------

def _drop_oversized_faces(tm, max_area_factor: float = 4.0):
    """Return a trimesh copy with faces > ``max_area_factor`` x median area removed.

    ``trimesh.repair.fill_holes`` can bridge a hole with one huge cap triangle
    when the boundary is irregular; this trims those degenerate caps back out.
    """
    import trimesh

    areas = tm.area_faces
    if len(areas) == 0:
        return tm
    median = np.median(areas)
    if median <= 0:
        return tm
    keep = areas <= max_area_factor * median
    if keep.all():
        return tm
    tm2 = tm.copy()
    tm2.update_faces(keep)
    tm2.remove_unreferenced_vertices()
    return tm2


def clean_shell(
    mesh: o3d.geometry.TriangleMesh,
    min_component_tris: int = 300,
    fill_hole_max_edge_m: float = 0.02,
    taubin_iters: int = 10,
) -> o3d.geometry.TriangleMesh:
    """Debris removal + hole filling + light smoothing -> a single coherent shell."""
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_duplicated_vertices()
    out.remove_unreferenced_vertices()

    if len(out.triangles) == 0:
        return out

    tri_ids, n_tris_per_comp, _ = out.cluster_connected_triangles()
    tri_ids = np.asarray(tri_ids)
    n_tris_per_comp = np.asarray(n_tris_per_comp)

    keep_components = np.where(n_tris_per_comp >= min_component_tris)[0]
    if len(keep_components) == 0:
        # nothing survives the threshold: keep the single largest component
        keep_components = np.array([int(np.argmax(n_tris_per_comp))])
    keep_mask = np.isin(tri_ids, keep_components)

    out.remove_triangles_by_mask(~keep_mask)
    out.remove_unreferenced_vertices()

    if len(out.triangles) == 0:
        return out

    # hole filling via trimesh (open3d has no direct hole-fill in this version)
    import trimesh

    v = np.asarray(out.vertices)
    f = np.asarray(out.triangles)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    try:
        trimesh.repair.fill_holes(tm)
    except Exception:
        pass
    # cap fill_hole_max_edge_m loosely: drop any newly-formed oversized cap
    # faces (a simple, robust stand-in for a true boundary-length filter).
    tm = _drop_oversized_faces(tm, max_area_factor=4.0)

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)),
    )
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_duplicated_vertices()
    out.remove_unreferenced_vertices()

    if len(out.triangles) and taubin_iters > 0:
        out = out.filter_smooth_taubin(number_of_iterations=int(taubin_iters))

    out.remove_non_manifold_edges()
    out.compute_vertex_normals()
    return out


# --------------------------------------------------------------------------
# 3. decimate
# --------------------------------------------------------------------------

def decimate(mesh: o3d.geometry.TriangleMesh, target_tris: int = 200_000) -> o3d.geometry.TriangleMesh:
    """Quadric-decimate to ``target_tris`` (no-op if already smaller)."""
    if len(mesh.triangles) <= target_tris:
        out = o3d.geometry.TriangleMesh(mesh)
    else:
        out = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_tris))
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


# --------------------------------------------------------------------------
# 2b. alternative shell: Poisson surface reconstruction on the cropped TSDF
# --------------------------------------------------------------------------

def poisson_after_tsdf(
    mesh_cropped: o3d.geometry.TriangleMesh,
    obb: dict,
    depth: int = 10,
    density_quantile: float = 0.02,
    pad_m: float = 0.03,
    target_tris: int = 200_000,
    stats_out: dict | None = None,
) -> o3d.geometry.TriangleMesh:
    """Poisson-reconstruct a watertight shell from the object-cropped TSDF mesh.

    Uses the cropped TSDF mesh's vertices + (computed) vertex normals as an
    oriented point cloud, voxel-downsampled to ~2 mm so Poisson isn't swamped
    by the TSDF's dense, uneven sampling. Low-density (low-support) vertices
    are dropped, the result is re-cropped to the padded OBB (Poisson
    reconstructs a closed shell that can overshoot the object's extent), the
    largest connected component is kept, and it is decimated to ``target_tris``.
    """
    m = o3d.geometry.TriangleMesh(mesh_cropped)
    if len(m.vertices) == 0:
        return o3d.geometry.TriangleMesh()
    m.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(m.vertex_normals))
    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    if len(pcd.points) < 10:
        return o3d.geometry.TriangleMesh()

    poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), linear_fit=True
    )
    densities = np.asarray(densities)
    if len(densities):
        thresh = float(np.quantile(densities, float(density_quantile)))
        poisson.remove_vertices_by_mask(densities < thresh)

    poisson = crop_to_object(poisson, obb, pad_m=pad_m)
    if len(poisson.triangles) == 0:
        if stats_out is not None:
            stats_out["n_components_before_largest"] = 0
        return poisson

    tri_ids, n_tris_per_comp, _ = poisson.cluster_connected_triangles()
    tri_ids = np.asarray(tri_ids)
    n_tris_per_comp = np.asarray(n_tris_per_comp)
    n_components = int(len(n_tris_per_comp))
    largest = int(np.argmax(n_tris_per_comp))
    poisson.remove_triangles_by_mask(tri_ids != largest)
    poisson.remove_unreferenced_vertices()
    poisson.compute_vertex_normals()

    if stats_out is not None:
        stats_out["n_components_before_largest"] = n_components

    return decimate(poisson, target_tris=target_tris)


# --------------------------------------------------------------------------
# surface-fit stat: rms distance to the planar idealisation
# --------------------------------------------------------------------------

def _rms_to_planar_solid(mesh: o3d.geometry.TriangleMesh, planar_mesh_path, n_samples: int = 20_000):
    planar_mesh_path = Path(planar_mesh_path)
    if not planar_mesh_path.exists() or len(mesh.triangles) == 0:
        return None
    planar = o3d.io.read_triangle_mesh(str(planar_mesh_path))
    if len(planar.triangles) == 0:
        return None
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_samples))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(planar))
    q = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)
    d = scene.compute_distance(q).numpy()
    return float(np.sqrt(np.mean(np.square(d))) * 1000.0)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _tsdf_mesh_to_world(
    mesh: o3d.geometry.TriangleMesh, tgt: dict, obb: dict, pad_m: float = 0.03
) -> o3d.geometry.TriangleMesh:
    """Map ``debug/tsdf_mesh*.ply`` vertices into the surfcap world frame, if needed.

    ``pipeline.py``'s ``sfm_tsdf`` recon mode fuses depth with the raw pycolmap
    poses (``recon.c2w``) and writes the whole-scene ``debug/tsdf_mesh.ply``
    straight out of that fusion, *before* the "frame" stage's
    ``sim3_world_from_recon`` similarity transform is applied (the transform
    that later produces ``table_world.ply`` / ``full_world.ply`` and the
    ``obb``/``surfaces`` in ``target.json``) -- so it needs that transform
    re-applied here, or the OBB crop is a no-op (recon-frame extents are
    ~10-20x too big and offset). ``debug/tsdf_mesh_crop.ply``, however, is
    already object-scale and world-aligned (a separate ad-hoc crop probe) and
    would be pushed *out* of the OBB by the same transform.

    Rather than hard-code which file needs which frame, try both (identity
    and the sim3) and keep whichever puts more vertices inside the padded
    OBB -- self-correcting regardless of which convention a given debug file
    happens to use.
    """
    sim3 = tgt.get("frame", {}).get("sim3_world_from_recon")
    if sim3 is None or len(mesh.vertices) == 0:
        return mesh

    centre = np.asarray(obb["centre"], dtype=np.float64).reshape(3)
    axes = np.asarray(obb["axes"], dtype=np.float64).reshape(3, 3)
    extents = np.asarray(obb["extents_m"], dtype=np.float64).reshape(3)
    box = o3d.geometry.OrientedBoundingBox(
        center=centre, R=axes.T, extent=extents + 2.0 * float(pad_m)
    )

    v = np.asarray(mesh.vertices, dtype=np.float64)
    T = np.asarray(sim3, dtype=np.float64)
    v_world = v @ T[:3, :3].T + T[:3, 3]

    n_raw = len(box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(v)))
    n_xform = len(box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(v_world)))

    out = o3d.geometry.TriangleMesh(mesh)
    if n_xform > n_raw:
        out.vertices = o3d.utility.Vector3dVector(v_world)
        out.vertex_normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))  # stale; recomputed later
    return out


def _find_repo_root(*candidate_paths) -> Path:
    """Locate the repo root (the dir holding ``out/``) from I/O paths first.

    ``detail_mesh.py`` can run out of a different checkout (e.g. a worktree)
    than the one holding the real ``out/`` tree it's asked to read/write, so
    ``Path(__file__).resolve().parent.parent`` is not reliable here. Walk up
    from each given path looking for an ancestor directory named ``out``, and
    fall back to this module's own repo root if none is found.
    """
    for p in candidate_paths:
        p = Path(p).resolve()
        for anc in (p, *p.parents):
            if anc.name == "out":
                return anc.parent
    return Path(__file__).resolve().parent.parent


def detail_mesh(
    tsdf_mesh_path,
    target_json_path,
    out_dir,
    target_tris: int = 200_000,
) -> dict:
    t0 = time.time()
    tsdf_mesh_path = Path(tsdf_mesh_path)
    target_json_path = Path(target_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(target_json_path) as fh:
        tgt = json.load(fh)
    obb = tgt["obb"]

    mesh_in = o3d.io.read_triangle_mesh(str(tsdf_mesh_path))
    tris_in = len(mesh_in.triangles)
    mesh_in = _tsdf_mesh_to_world(mesh_in, tgt, obb)

    cropped = crop_to_object(mesh_in, obb, pad_m=0.03)
    tris_after_crop = len(cropped.triangles)

    # colour source for the exported meshes: the cropped TSDF vertices' own RGB
    # when present, else the scene's cleaned coloured cloud (target.ply /
    # target_clean.ply) -- pipeline-produced debug/tsdf_mesh.ply frequently has
    # no vertex colour at all.
    from .hybrid_mesh import _scene_colour_source, colorize

    if cropped.has_vertex_colors():
        colour_source = (
            np.asarray(cropped.vertices, dtype=np.float64).copy(),
            np.asarray(cropped.vertex_colors, dtype=np.float64).copy(),
        )
    else:
        colour_source = _scene_colour_source(target_json_path)

    if tris_after_crop:
        tri_ids0, n0, _ = cropped.cluster_connected_triangles()
        components_before = int(len(np.asarray(n0)))
    else:
        components_before = 0

    cleaned = clean_shell(cropped, min_component_tris=300, fill_hole_max_edge_m=0.02, taubin_iters=10)

    if len(cleaned.triangles):
        _, n1, _ = cleaned.cluster_connected_triangles()
        components_after = int(len(np.asarray(n1)))
    else:
        components_after = 0

    final = decimate(cleaned, target_tris=target_tris)
    tris_out = len(final.triangles)
    watertight = bool(final.is_watertight()) if tris_out else False

    v = np.asarray(final.vertices)
    if len(v):
        bbox_extents = (v.max(axis=0) - v.min(axis=0)).tolist()
    else:
        bbox_extents = [0.0, 0.0, 0.0]

    # scene name -> planar solid to compare against, e.g. sfm_cabinet_c -> out/cabinet_c
    stem = tsdf_mesh_path.parent.parent.name  # .../sfm_cabinet_c/debug/tsdf_mesh.ply
    scene = stem.split("_", 1)[1] if "_" in stem else stem
    repo_root = _find_repo_root(out_dir, tsdf_mesh_path)
    planar_path = repo_root / "out" / scene / "target_mesh.ply"
    rms_vs_planar_mm = _rms_to_planar_solid(final, planar_path)

    if colour_source is not None:
        final = colorize(final, colour_source, max_dist_m=0.015)

    ply_path = out_dir / "target_mesh_detail.ply"
    glb_path = out_dir / "target_mesh_detail.glb"
    o3d.io.write_triangle_mesh(
        str(ply_path), final, write_vertex_normals=True, write_vertex_colors=True
    )

    _write_glb(final, glb_path)

    # --- variant 2: Poisson surface reconstruction on the cropped raw TSDF mesh ---
    psr_stats_extra: dict = {}
    psr = poisson_after_tsdf(
        cropped, obb, depth=10, density_quantile=0.02, pad_m=0.03,
        target_tris=target_tris, stats_out=psr_stats_extra,
    )
    tris_out_psr = len(psr.triangles)
    watertight_psr = bool(psr.is_watertight()) if tris_out_psr else False
    v_psr = np.asarray(psr.vertices)
    bbox_extents_psr = (v_psr.max(axis=0) - v_psr.min(axis=0)).tolist() if len(v_psr) else [0.0, 0.0, 0.0]
    rms_vs_planar_mm_psr = _rms_to_planar_solid(psr, planar_path)

    if colour_source is not None:
        psr = colorize(psr, colour_source, max_dist_m=0.015)

    ply_path_psr = out_dir / "target_mesh_detail_psr.ply"
    glb_path_psr = out_dir / "target_mesh_detail_psr.glb"
    o3d.io.write_triangle_mesh(
        str(ply_path_psr), psr, write_vertex_normals=True, write_vertex_colors=True
    )
    _write_glb(psr, glb_path_psr)

    stats = {
        "tris_in": int(tris_in),
        "tris_after_crop": int(tris_after_crop),
        "components_before": components_before,
        "components_after": components_after,
        "tris_out": int(tris_out),
        "watertight": watertight,
        "bbox_extents_m": bbox_extents,
        "rms_vs_planar_mm": rms_vs_planar_mm,
        "planar_ref": str(planar_path) if planar_path.exists() else None,
        "runtime_s": round(time.time() - t0, 2),
        "out_ply": str(ply_path),
        "out_glb": str(glb_path),
        "psr": {
            "components_before_largest": psr_stats_extra.get("n_components_before_largest", 0),
            "components_after": 1 if tris_out_psr else 0,
            "tris_out": int(tris_out_psr),
            "watertight": watertight_psr,
            "bbox_extents_m": bbox_extents_psr,
            "rms_vs_planar_mm": rms_vs_planar_mm_psr,
            "out_ply": str(ply_path_psr),
            "out_glb": str(glb_path_psr),
        },
    }
    with open(out_dir / "detail_mesh_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    try:
        from surfcap.postprocess import render_mesh_png

        render_mesh_png(final, out_dir / "detail_mesh_iso.png", max_tris=60_000)
    except Exception as e:  # pragma: no cover - rendering is best-effort
        stats["render_error"] = str(e)

    try:
        from surfcap.postprocess import render_mesh_png

        render_mesh_png(psr, out_dir / "detail_mesh_psr_iso.png", max_tris=60_000)
    except Exception as e:  # pragma: no cover - rendering is best-effort
        stats["render_error_psr"] = str(e)

    stats["runtime_s"] = round(time.time() - t0, 2)
    return stats


def _write_glb(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    import trimesh

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    if len(f) == 0:
        return
    vc = None
    if mesh.has_vertex_colors():
        vc = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    vn = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    tm = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vc, vertex_normals=vn, process=False)
    tm.export(str(path))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="D1: object-only detailed mesh from a raw TSDF mesh")
    ap.add_argument("tsdf_mesh_ply")
    ap.add_argument("target_json")
    ap.add_argument("out_dir")
    ap.add_argument("--target-tris", type=int, default=200_000)
    args = ap.parse_args(argv)

    stats = detail_mesh(args.tsdf_mesh_ply, args.target_json, args.out_dir, target_tris=args.target_tris)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
