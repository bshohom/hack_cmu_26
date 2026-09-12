"""R6 post-processing: denoise -> flatten-to-planes -> Poisson mesh.

Input clouds/surfaces are assumed to be in the surfcap world frame (metres,
Z-up, table top at z ~= 0, card centre at the origin; see surfcap/types.py).

CLI:
    python -m surfcap.postprocess out/table_a
reads ``target.ply`` + ``target.json``, runs :func:`postprocess`, prints the
stats and renders ``debug/mesh_iso.png``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from surfcap.types import Surface


class _StepTimer:
    """Print elapsed wall time for each named step (postprocess perf debugging)."""

    def __init__(self, label: str = "postprocess"):
        self.label = label
        self._t0 = time.time()

    def lap(self, step: str) -> None:
        t1 = time.time()
        print(f"[{self.label}] {step}: {t1 - self._t0:.2f}s", flush=True)
        self._t0 = t1

# Fallback slab thickness (m) when the OBB gives nothing plausible.
DEFAULT_THICKNESS_M = 0.018
_THICKNESS_RANGE_M = (0.01, 0.05)


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------

def _as_surface(s) -> Surface:
    if isinstance(s, Surface):
        return s
    return Surface(**s)


def _plane_of(surf: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal + a point on the plane (the surface centroid)."""
    n = np.asarray(surf.normal, dtype=np.float64).reshape(3)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / ln
    c = np.asarray(surf.centroid, dtype=np.float64).reshape(3)
    return n, c


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(tmp, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _points_in_polygon(pts2d: np.ndarray, poly2d: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Boolean mask of 2D points inside a 2D polygon (matplotlib.path, even-odd)."""
    if len(poly2d) < 3 or len(pts2d) == 0:
        return np.zeros(len(pts2d), dtype=bool)
    from matplotlib.path import Path as MplPath

    path = MplPath(np.asarray(poly2d, dtype=np.float64))
    return path.contains_points(np.asarray(pts2d, dtype=np.float64), radius=pad)


def _rms_mm(d: np.ndarray) -> float:
    if len(d) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(d))) * 1000.0)


# --------------------------------------------------------------------------
# 1. denoise
# --------------------------------------------------------------------------

def denoise(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.003,
    sor_nb: int = 30,
    sor_std: float = 1.5,
    ror_radius: float = 0.012,
    ror_min: int = 8,
) -> o3d.geometry.PointCloud:
    """Voxel-downsample, then statistical + radius outlier removal."""
    out = pcd
    if voxel and voxel > 0:
        out = out.voxel_down_sample(voxel_size=float(voxel))
    if sor_nb and len(out.points) > sor_nb:
        out, _ = out.remove_statistical_outlier(
            nb_neighbors=int(sor_nb), std_ratio=float(sor_std)
        )
    if ror_min and len(out.points) > ror_min:
        out, _ = out.remove_radius_outlier(
            nb_points=int(ror_min), radius=float(ror_radius)
        )
    return out


# --------------------------------------------------------------------------
# 2. flatten to planes
# --------------------------------------------------------------------------

def flatten_to_planes(
    pcd: o3d.geometry.PointCloud,
    surfaces: list,
    band_m: float = 0.012,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Project points near (and inside) each fitted surface exactly onto its plane.

    Surfaces are handled in the given order (``top`` first in the surfcap
    contract); a point already claimed by an earlier surface is not reassigned.
    Returns the flattened cloud (a copy) and
    ``{surface_id: {n_projected, rms_before_mm, rms_after_mm}}``.
    """
    out = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(out.points, dtype=np.float64).copy()
    stats: dict[str, dict] = {}
    if len(pts) == 0:
        return out, stats

    claimed = np.zeros(len(pts), dtype=bool)
    for s in surfaces:
        surf = _as_surface(s)
        n, c = _plane_of(surf)
        poly3d = np.asarray(surf.polygon_3d, dtype=np.float64)

        signed = (pts - c) @ n
        band = (np.abs(signed) <= float(band_m)) & (~claimed)

        if len(poly3d) >= 3:
            if abs(n[2]) > 0.9:
                # near-horizontal surface: plain XY containment test
                inside = _points_in_polygon(pts[:, :2], poly3d[:, :2])
            else:
                u, v = _plane_basis(n)
                pc = poly3d - c
                poly2d = np.stack([pc @ u, pc @ v], axis=1)
                rel = pts - c
                pts2d = np.stack([rel @ u, rel @ v], axis=1)
                inside = _points_in_polygon(pts2d, poly2d)
        else:
            inside = np.ones(len(pts), dtype=bool)

        sel = band & inside
        k = int(sel.sum())
        rms_before = _rms_mm(signed[sel])
        if k:
            pts[sel] = pts[sel] - np.outer(signed[sel], n)
            claimed |= sel
        rms_after = _rms_mm((pts[sel] - c) @ n) if k else 0.0
        stats[surf.id] = {
            "n_projected": k,
            "rms_before_mm": round(rms_before, 6),
            "rms_after_mm": round(rms_after, 9),
        }

    out.points = o3d.utility.Vector3dVector(pts)
    return out, stats


# --------------------------------------------------------------------------
# 3. Poisson mesh
# --------------------------------------------------------------------------

def _normal_radius(pts: np.ndarray, default: float = 0.02) -> float:
    """Normal-estimation radius that stays inside one face of a thin slab."""
    if len(pts) < 10:
        return default
    rel = pts - pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(rel, full_matrices=False)
    n = vt[-1]
    s = rel @ n
    thin = float(np.percentile(s, 99) - np.percentile(s, 1))
    if thin <= 1e-6:
        return default
    return float(np.clip(0.4 * thin, 0.005, default))


def _slab_reference_normals(pts: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Outward reference direction per point, assuming the cloud is a thin slab.

    PCA on a slab gives the slab normal as the smallest-variance axis. A point
    whose (unoriented) normal is roughly parallel to that axis is on one of the
    two faces, so its outward direction is +/- the axis according to which side
    of the mid-plane it sits on. Everything else is on the rim and points
    radially outward.
    """
    rel = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)

    s = rel @ n
    s = s - np.median(s)

    ref = np.zeros_like(pts)
    face = np.abs(nrm @ n) > 0.7
    sgn = np.where(s >= 0, 1.0, -1.0)
    ref[face] = sgn[face, None] * n[None, :]

    rim = ~face
    if rim.any():
        radial = rel[rim] - np.outer(rel[rim] @ n, n)
        ln = np.linalg.norm(radial, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        ref[rim] = radial / ln
    return ref


def _top_band_mask(pts: np.ndarray) -> np.ndarray:
    z_hi = np.percentile(pts[:, 2], 90.0)
    m = pts[:, 2] >= z_hi - 0.005
    if m.sum() < 10:
        m = pts[:, 2] >= np.percentile(pts[:, 2], 75.0)
    return m


def _orient_normals(pcd: o3d.geometry.PointCloud, mst_max_points: int = 20000) -> None:
    """Estimate normals and orient them outward, +Z on the top band.

    The geometric slab orientation (``_slab_reference_normals``) runs first: it
    is O(n) and, for the thin-slab geometry this pipeline always produces,
    reliable. ``orient_normals_consistent_tangent_plane`` is a nice-to-have
    refinement for clouds without an obvious slab axis, but its MST can stall
    for a very long time (minutes, not seconds) once the cloud has a genuine
    interior hole -- e.g. a card cut out of the middle of the top face plus its
    mirrored underside -- so it is only attempted on clouds small enough that a
    stall is cheap, and only when the geometric pass did not already agree with
    itself on the top band.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return
    # A 0.02 m neighbourhood straddles both faces of a thin slab and yields
    # meaningless normals, so shrink it to stay inside one face.
    radius = _normal_radius(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )

    nrm = np.asarray(pcd.normals)
    ref = _slab_reference_normals(pts, nrm)
    flip = np.einsum("ij,ij->i", nrm, ref) < 0
    nrm = nrm.copy()
    nrm[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nrm)

    top = _top_band_mask(pts)
    agree = max(
        float((nrm[top, 2] > 0).mean()), float((nrm[top, 2] < 0).mean())
    )
    if agree < 0.9 and len(pts) <= mst_max_points:
        # The slab assumption itself looks shaky here (small clouds only, so
        # a slow MST is bounded) -- try the general-purpose MST orientation.
        try:
            pcd.orient_normals_consistent_tangent_plane(30)
        except Exception:
            pass
        nrm = np.asarray(pcd.normals)
        agree = max(
            float((nrm[top, 2] > 0).mean()), float((nrm[top, 2] < 0).mean())
        )

    if agree < 0.9:
        # Still unreliable -> orient outward geometrically (this also covers
        # the case where the MST branch above was skipped as too large).
        ref = _slab_reference_normals(pts, nrm)
        flip = np.einsum("ij,ij->i", nrm, ref) < 0
        nrm = nrm.copy()
        nrm[flip] *= -1.0
        pcd.normals = o3d.utility.Vector3dVector(nrm)

    nrm = np.asarray(pcd.normals)
    if np.median(nrm[top, 2]) < 0:
        pcd.normals = o3d.utility.Vector3dVector(-nrm)


def _nn_query(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour distances + indices of `query` in `ref` (vectorised)."""
    try:
        from scipy.spatial import cKDTree

        return cKDTree(ref).query(query, k=1)
    except Exception:  # pragma: no cover - scipy is in the env
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(ref)
        tree = o3d.geometry.KDTreeFlann(pc)
        d = np.empty(len(query))
        idx = np.empty(len(query), dtype=int)
        for i, q in enumerate(query):
            _, j, dist2 = tree.search_knn_vector_3d(q, 1)
            d[i] = np.sqrt(dist2[0])
            idx[i] = j[0]
        return d, idx


def _transfer_colours(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud
) -> None:
    if not pcd.has_colors() or len(mesh.vertices) == 0:
        return
    src = np.asarray(pcd.colors)
    _, idx = _nn_query(np.asarray(mesh.vertices), np.asarray(pcd.points))
    mesh.vertex_colors = o3d.utility.Vector3dVector(src[idx])


def poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int = 8,
    density_quantile: float = 0.05,
    smooth_iters: int = 20,
    crop_to_pcd_bbox_pad: float = 0.01,
) -> o3d.geometry.TriangleMesh:
    """Screened-Poisson surface reconstruction with cleanup + colour transfer."""
    pt = _StepTimer("poisson_mesh")
    work = o3d.geometry.PointCloud(pcd)
    _orient_normals(work)
    pt.lap(f"orient_normals (n={len(work.points)})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        work, depth=int(depth), linear_fit=True
    )
    densities = np.asarray(densities)
    pt.lap("create_from_point_cloud_poisson")

    if density_quantile and len(densities):
        thr = np.quantile(densities, float(density_quantile))
        mesh.remove_vertices_by_mask(densities < thr)
    pt.lap("density_trim")

    pts = np.asarray(work.points)
    if len(pts):
        pad = float(crop_to_pcd_bbox_pad)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=pts.min(axis=0) - pad, max_bound=pts.max(axis=0) + pad
        )
        mesh = mesh.crop(bbox)
    pt.lap("crop")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    pt.lap("remove_degenerate/duplicated")

    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if len(counts):
            keep = int(np.argmax(counts))
            mesh.remove_triangles_by_mask(labels != keep)
            mesh.remove_unreferenced_vertices()
    pt.lap("cluster_connected_components")

    if smooth_iters and len(mesh.triangles):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
    pt.lap("taubin_smooth")

    mesh.compute_vertex_normals()
    _transfer_colours(mesh, pcd)
    pt.lap("compute_normals/colour_transfer")
    return mesh


# --------------------------------------------------------------------------
# mesh <-> cloud agreement
# --------------------------------------------------------------------------

def mesh_to_cloud_rms_mm(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, n_samples: int = 20000
) -> float:
    if len(mesh.triangles) == 0 or len(pcd.points) == 0:
        return float("nan")
    samp = mesh.sample_points_uniformly(number_of_points=int(n_samples))
    d, _ = _nn_query(np.asarray(samp.points), np.asarray(pcd.points))
    return float(np.sqrt(np.mean(np.square(d))) * 1000.0)


# --------------------------------------------------------------------------
# mirror-closing (Poisson needs a closed shell, a single sheet bulges)
# --------------------------------------------------------------------------

def slab_thickness(
    obb: dict | None, thickness_m: float | None = None
) -> tuple[float, str]:
    """Slab thickness: explicit override > OBB's third extent > default fallback.

    Returns ``(thickness_m, source)`` with ``source`` one of "arg", "obb",
    "default".
    """
    if thickness_m is not None:
        return float(thickness_m), "arg"
    try:
        t = float(np.asarray(obb["extents_m"], dtype=float)[2])
    except Exception:
        return DEFAULT_THICKNESS_M, "default"
    lo, hi = _THICKNESS_RANGE_M
    if lo <= t <= hi:
        return t, "obb"
    return DEFAULT_THICKNESS_M, "default"


def _rim_points(poly3d: np.ndarray, n: np.ndarray, thickness: float, step: float) -> np.ndarray:
    """Sample the slab's side walls: the top polygon's edges swept down by `thickness`."""
    if len(poly3d) < 3 or thickness <= 0:
        return np.zeros((0, 3))
    n_depth = max(2, int(np.ceil(thickness / step)) + 1)
    depths = np.linspace(0.0, thickness, n_depth)
    out = []
    for i in range(len(poly3d)):
        a = poly3d[i]
        b = poly3d[(i + 1) % len(poly3d)]
        seg = np.linalg.norm(b - a)
        k = max(2, int(np.ceil(seg / step)) + 1)
        ts = np.linspace(0.0, 1.0, k)[:, None]
        edge = a[None, :] * (1 - ts) + b[None, :] * ts       # (k,3)
        wall = edge[:, None, :] - depths[None, :, None] * n[None, None, :]
        out.append(wall.reshape(-1, 3))
    return np.concatenate(out, axis=0)


def mirror_close(
    pcd: o3d.geometry.PointCloud,
    surfaces: list,
    thickness: float,
    band_m: float = 0.002,
    rim_step: float = 0.004,
) -> o3d.geometry.PointCloud:
    """Close the slab so Poisson has a solid, not a one-sided sheet.

    A single sheet of points makes screened Poisson balloon around it. We give
    it (a) the flattened top points mirrored `thickness` below the plane and
    (b) points along the side walls swept from the surface polygon, so the
    reconstruction stays a thin box.
    """
    if not surfaces:
        return pcd
    surf = _as_surface(surfaces[0])
    n, c = _plane_of(surf)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        return pcd
    signed = (pts - c) @ n
    sel = np.abs(signed) <= band_m
    if sel.sum() < 10:
        return pcd

    extra = [pts[sel] - np.outer(np.full(int(sel.sum()), float(thickness)), n)]

    poly3d = np.asarray(surf.polygon_3d, dtype=np.float64)
    if len(poly3d) >= 3:
        # project the polygon exactly onto the plane before sweeping it
        poly3d = poly3d - np.outer((poly3d - c) @ n, n)
        rim = _rim_points(poly3d, n, float(thickness), float(rim_step))
        if len(rim):
            extra.append(rim)

    add = np.concatenate(extra, axis=0)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack([pts, add]))
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        mirror_cols = cols[sel]
        pad = np.tile(cols.mean(axis=0), (len(add) - len(mirror_cols), 1))
        out.colors = o3d.utility.Vector3dVector(
            np.vstack([cols, mirror_cols, pad])
        )
    return out


# --------------------------------------------------------------------------
# 4. driver
# --------------------------------------------------------------------------

def postprocess(
    pcd: o3d.geometry.PointCloud,
    surfaces: list,
    out_dir,
    obb: dict | None = None,
    band_m: float = 0.012,
    depth: int = 8,
    mirror: bool = True,
    thickness_m: float | None = None,
    mesh_mode: str = "planar",
    alpha_m: float = 0.03,
) -> dict:
    """denoise -> flatten -> mesh; writes clean PLY + mesh PLY/GLB.

    ``mesh_mode="planar"`` (default, F4) builds the mesh from the fitted planes
    (:mod:`surfcap.planar_mesh`): one concave-hull patch per surface, corners
    snapped to the planes' intersection lines, closed into a solid.  This is the
    right model for unions of planes (cabinet corner, window sill) and never
    leaves holes where the cloud is sparse.

    ``mesh_mode="poisson"`` keeps the original mirror-closed screened-Poisson
    path, which assumes a single thin slab.

    Returns a stats dict (also embedded into target.json by the pipeline).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = _StepTimer("postprocess")

    n_before = int(len(pcd.points))
    clean = denoise(pcd)
    n_after = int(len(clean.points))
    pt.lap(f"denoise ({n_before} -> {n_after})")

    flat, flat_stats = flatten_to_planes(clean, surfaces, band_m=band_m)
    pt.lap("flatten_to_planes")

    clean_path = out_dir / "target_clean.ply"
    mesh_ply = out_dir / "target_mesh.ply"
    mesh_glb = out_dir / "target_mesh.glb"

    stats = {
        "n_before": n_before,
        "n_after_denoise": n_after,
        "n_removed": n_before - n_after,
        "removed_frac": round((n_before - n_after) / max(1, n_before), 4),
        "flatten": flat_stats,
        "band_m": band_m,
        "mesh_mode": mesh_mode,
    }

    if mesh_mode == "planar":
        from surfcap import planar_mesh as pm

        tm, pstats = pm.planar_mesh(
            flat, surfaces, thickness_m=thickness_m, alpha_m=alpha_m, obb=obb
        )
        patches = pstats.pop("_patches", [])
        pt.lap(f"planar_mesh (case {pstats.get('case')})")
        mesh = pm.to_open3d(tm)
        patches_glb = out_dir / "target_mesh_planar_patches.glb"
        try:
            pm.write_patches_glb(patches, patches_glb)
            stats["patches_glb"] = patches_glb.name
        except Exception as exc:  # pragma: no cover - viewer aid only
            print(f"[postprocess] patches glb failed: {exc}")
        if len(tm.faces):
            tm.export(str(mesh_glb))
        stats.update(pstats)
        stats["watertight"] = bool(pstats.get("closed", False))
        stats["mesh_to_observed_rms_mm"] = pstats.get("mesh_to_cloud_rms_mm")
        for w in pstats.pop("warnings", []) or []:
            print(f"[postprocess] warning: {w}")
    else:
        thickness, thickness_source = slab_thickness(obb, thickness_m=thickness_m)
        mesh_input = mirror_close(flat, surfaces, thickness) if mirror else flat
        n_mirrored = int(len(mesh_input.points) - len(flat.points))
        pt.lap(f"mirror_close (+{n_mirrored} pts)")
        mesh = poisson_mesh(mesh_input, depth=depth)
        pt.lap("poisson_mesh total")
        stats.update({
            "mirror_closed": bool(mirror),
            "n_mirrored": n_mirrored,
            "thickness_m": round(float(thickness), 6),
            "thickness_source": thickness_source,
            "mesh_n_vertices": int(len(mesh.vertices)),
            "mesh_n_triangles": int(len(mesh.triangles)),
            "watertight": bool(mesh.is_watertight()),
            # agreement with the cloud the mesh was reconstructed from (includes
            # the mirrored underside when mirror-closing is on)...
            "mesh_to_cloud_rms_mm": round(mesh_to_cloud_rms_mm(mesh, mesh_input), 4),
            # ...and with the observed (single-sided) cloud only
            "mesh_to_observed_rms_mm": round(mesh_to_cloud_rms_mm(mesh, flat), 4),
        })
        _write_mesh_glb(mesh, mesh_glb)

    o3d.io.write_point_cloud(str(clean_path), flat, write_ascii=False)
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
    pt.lap("write_ply/glb")

    stats.update({
        "clean_ply": clean_path.name,
        "mesh_ply": mesh_ply.name,
        "mesh_glb": mesh_glb.name,
    })
    stats["_mesh"] = mesh
    return stats


def _write_mesh_glb(mesh: o3d.geometry.TriangleMesh, path) -> None:
    """GLB via trimesh so per-vertex colours survive."""
    import trimesh

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        return
    kw = {}
    if mesh.has_vertex_colors():
        cols = np.clip(np.asarray(mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        cols = np.concatenate([cols, np.full((len(cols), 1), 255, np.uint8)], axis=1)
        kw["vertex_colors"] = cols
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False, **kw)
    tm.export(str(path))


# --------------------------------------------------------------------------
# debug render
# --------------------------------------------------------------------------

def render_mesh_png(mesh: o3d.geometry.TriangleMesh, path, max_tris: int = 4000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = mesh
    if len(m.triangles) > max_tris:
        try:
            m = m.simplify_quadric_decimation(target_number_of_triangles=max_tris)
        except Exception:
            pass
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)

    fig = plt.figure(figsize=(13, 5))
    if len(f):
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax.plot_trisurf(
            v[:, 0], v[:, 1], f, v[:, 2], cmap="viridis", linewidth=0, antialiased=False
        )
        ax.set_title(f"iso ({len(np.asarray(mesh.triangles))} tris)")
        ax.view_init(elev=32, azim=-55)
        try:
            ax.set_box_aspect(np.ptp(v, axis=0) + 1e-6)
        except Exception:
            pass

    for i, (a, b, lbl) in enumerate(((0, 2, "front X-Z"), (1, 2, "side Y-Z"))):
        ax = fig.add_subplot(1, 3, i + 2)
        ax.scatter(v[:, a], v[:, b], s=1, c=v[:, 2], cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(lbl)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="surfcap R6 post-processing")
    ap.add_argument("out_dir", help="a run directory containing target.ply/target.json")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--band", type=float, default=0.012)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--thickness", type=float, default=None,
                    help="override slab thickness (m); default derives from OBB")
    ap.add_argument("--mesh-mode", choices=("planar", "poisson"), default="planar")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    with open(out_dir / "target.json") as fh:
        tgt = json.load(fh)
    surfaces = [Surface(**s) for s in tgt.get("surfaces", [])]
    pcd = o3d.io.read_point_cloud(str(out_dir / tgt.get("cloud", {}).get("ply", "target.ply")))

    stats = postprocess(
        pcd,
        surfaces,
        out_dir,
        obb=tgt.get("obb"),
        band_m=args.band,
        depth=args.depth,
        mirror=not args.no_mirror,
        thickness_m=args.thickness,
        mesh_mode=args.mesh_mode,
    )
    stats.pop("_mesh", None)
    print(json.dumps(stats, indent=2))

    mesh = o3d.io.read_triangle_mesh(str(out_dir / "target_mesh.ply"))
    png = out_dir / "debug" / "mesh_iso.png"
    render_mesh_png(mesh, png)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
