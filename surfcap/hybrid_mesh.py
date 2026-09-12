"""F9: piecewise-planar hybrid mesh -- flat where flat, detail where not, closed where unobserved.

Motivation (user, verbatim intent): *"Planar solids are better outputs but lack
fine features like a lip or an edge. TSDF-PSR captures a fine mesh, but surfaces
are not planar, there is noise, and for table A it did not capture the bottom
surface. We need a middle ground where we recover an accurate surface mesh."*

The two existing mesh outputs sit at opposite extremes:

* ``target_mesh.ply`` (:mod:`surfcap.planar_mesh`) -- a closed union of the
  fitted planes. Exactly flat, watertight, but every lip/rim/seam is gone.
* ``target_mesh_detail_psr.ply`` (:mod:`surfcap.detail_mesh`) -- Poisson over the
  raw TSDF shell. Keeps detail, but the planes carry ~2-3 mm of fusion noise and
  unobserved faces (e.g. a table's underside) are missing entirely.

This module merges them at the *point* level before meshing:

1. crop the whole-scene TSDF mesh to the object OBB, turn it into an oriented
   cloud;
2. :func:`planar_patches` -- detect planar patches (Open3D 0.19
   ``detect_planar_patches``, RANSAC+DBSCAN fallback);
3. :func:`snap_to_patches` -- patch inliers are projected *exactly* onto their
   plane and given the plane normal; everything else stays put and is
   ``detail`` (this is where the lip survives);
4. :func:`hull_closure_points` -- sample the planar solid's surface and keep
   only the samples nowhere near an observed point: the unobserved hull (table
   underside, hidden sides);
5. Poisson over ``snapped u detail u hull`` -> trim -> crop -> largest component
   -> post-snap mesh vertices back onto their patch planes -> decimate.

CLI::

    python -m surfcap.hybrid_mesh out/e1b/table_a

Frame: everything is in the surfcap world frame (metres, Z-up, card centre at
the origin). The whole-scene ``debug/tsdf_mesh.ply`` may be in the *recon*
frame; :func:`surfcap.detail_mesh._tsdf_mesh_to_world` auto-detects and applies
``frame.sim3_world_from_recon`` and is reused here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from surfcap.detail_mesh import (
    _tsdf_mesh_to_world,
    _write_glb,
    crop_to_object,
    decimate,
)


# --------------------------------------------------------------------------
# patch container
# --------------------------------------------------------------------------

@dataclass
class Patch:
    """One detected planar patch.

    ``normal``/``d`` define the plane ``n . x + d = 0`` (``normal`` unit length).
    ``obb`` is the detector's oriented bounding box (used to bound the patch
    laterally); ``inlier_idx`` indexes the cloud the patch was detected in.
    """

    normal: np.ndarray
    d: float
    obb: o3d.geometry.OrientedBoundingBox
    inlier_idx: np.ndarray
    area_m2: float
    label: str = "patch"
    extent_m: tuple = field(default=(0.0, 0.0))
    basis: np.ndarray | None = None      # (2, 3) in-plane u, v
    hull2d: np.ndarray | None = None     # (k, 2) convex hull of the inliers in (u, v)
    source: str = "observed"             # "observed" | "hull"

    def signed_dist(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d

    def project(self, pts: np.ndarray) -> np.ndarray:
        return pts - np.outer(self.signed_dist(pts), self.normal)

    def edge_dist(self, pts: np.ndarray) -> np.ndarray:
        """In-plane distance from each point to the patch's boundary polygon.

        Large => deep inside the flat region; ~0 => on the rim, where a real
        rounded edge (fillet) would live. Returns +inf when no hull is known.
        """
        if self.hull2d is None or self.basis is None or len(self.hull2d) < 3:
            return np.full(len(pts), np.inf)
        q = (pts - (-self.d) * self.normal) @ self.basis.T
        poly = self.hull2d
        a = poly
        b = np.roll(poly, -1, axis=0)
        ab = b - a                                        # (k, 2)
        L2 = np.maximum((ab * ab).sum(axis=1), 1e-18)
        ap = q[:, None, :] - a[None, :, :]                # (n, k, 2)
        t = np.clip((ap * ab[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        closest = a[None] + t[:, :, None] * ab[None]
        return np.linalg.norm(q[:, None, :] - closest, axis=2).min(axis=1)


def _plane_uv(n: np.ndarray) -> np.ndarray:
    """Two orthonormal in-plane axes for a unit normal, as a (2, 3) array."""
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return np.stack([u, v])


def _hull2d(pts: np.ndarray, n: np.ndarray, d: float, basis: np.ndarray):
    """Convex hull of ``pts`` in the plane's (u, v) basis, or ``None``."""
    if len(pts) < 3:
        return None
    q = (pts - (-d) * n) @ basis.T
    try:
        from scipy.spatial import ConvexHull

        return q[ConvexHull(q).vertices]
    except Exception:
        return None


def _normal_label(n: np.ndarray) -> str:
    """Crude role-ish label from the patch normal (Z-up world frame)."""
    nz = float(n[2])
    if nz > 0.85:
        return "top"
    if nz < -0.85:
        return "bottom"
    if abs(nz) < 0.35:
        return "vertical"
    return "slanted"


def _lateral_obb(obb: o3d.geometry.OrientedBoundingBox, thickness_m: float) -> o3d.geometry.OrientedBoundingBox:
    """Copy of ``obb`` with its plane-normal (local z) extent set to ``thickness_m``.

    ``detect_planar_patches`` returns boxes whose z extent is whatever the
    contributing points spanned; for "is this point laterally inside the patch"
    tests we want to control that band ourselves.
    """
    ext = np.asarray(obb.extent, dtype=np.float64).copy()
    ext[2] = float(thickness_m)
    return o3d.geometry.OrientedBoundingBox(center=obb.center, R=np.asarray(obb.R), extent=ext)


# --------------------------------------------------------------------------
# 1. planar patch detection
# --------------------------------------------------------------------------

def _fallback_patches(
    pcd: o3d.geometry.PointCloud,
    dist_thresh: float,
    min_points: int,
    min_edge_m: float,
    max_planes: int = 8,
) -> list[o3d.geometry.OrientedBoundingBox]:
    """Iterative RANSAC + DBSCAN, as in :mod:`surfcap.primitives`, returning OBBs.

    Only used when ``detect_planar_patches`` is unavailable (older Open3D).
    """
    work = o3d.geometry.PointCloud(pcd)
    remaining = np.arange(len(work.points))
    boxes: list[o3d.geometry.OrientedBoundingBox] = []
    for _ in range(max_planes):
        if len(work.points) < min_points:
            break
        try:
            _, inl = work.segment_plane(
                distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000
            )
        except Exception:
            break
        inl = np.asarray(inl, dtype=np.int64)
        if len(inl) < min_points:
            break
        sub = work.select_by_index(inl.tolist())
        labels = np.asarray(sub.cluster_dbscan(eps=max(3.0 * dist_thresh, 0.02), min_points=10))
        for lab in np.unique(labels[labels >= 0]):
            sel = np.where(labels == lab)[0]
            if len(sel) < min_points:
                continue
            cl = sub.select_by_index(sel.tolist())
            try:
                box = cl.get_oriented_bounding_box()
            except Exception:
                continue
            ext = np.sort(np.asarray(box.extent))[::-1]
            if ext[0] < min_edge_m:
                continue
            # re-order so local z is the thin (normal) axis, matching o3d's
            # detect_planar_patches convention
            order = np.argsort(np.asarray(box.extent))[::-1]  # long, mid, thin
            R = np.asarray(box.R)[:, order]
            if np.linalg.det(R) < 0:
                R[:, 0] = -R[:, 0]
            boxes.append(
                o3d.geometry.OrientedBoundingBox(
                    center=box.center, R=R, extent=np.asarray(box.extent)[order]
                )
            )
        keep = np.setdiff1d(np.arange(len(work.points)), inl)
        remaining = remaining[keep]
        work = work.select_by_index(keep.tolist())
    return boxes


def planar_patches(
    pcd: o3d.geometry.PointCloud,
    dist_thresh: float = 0.003,
    normal_deg: float = 12.0,
    min_points: int = 300,
    min_edge_m: float = 0.02,
    coplanarity_deg: float = 75.0,
    detector_min_num_points: int = 0,
) -> list[Patch]:
    """Detect planar patches and collect their inliers.

    A point is an inlier of a patch if it lies within ``dist_thresh`` of the
    patch plane *and* laterally inside the patch's own oriented bounding box.
    Points that qualify for several patches are assigned to the nearest plane.
    Plane parameters are refit by PCA over the inliers (the detector's box axes
    are quantised by its normal-clustering step).

    ``min_points`` is our own post-filter on a patch's inlier count. Note that
    Open3D's ``min_num_points`` is *not* that: it is the minimum size of an
    octree node the detector will still try to split/fit, so raising it to 300
    coarsens the partition until nothing is found at all (measured: 2 patches
    at 0, 0 patches at 300 on the synthetic slab). It is therefore exposed
    separately as ``detector_min_num_points`` and left at Open3D's auto value.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        return []

    detector = getattr(pcd, "detect_planar_patches", None)
    if detector is not None:
        boxes = detector(
            normal_variance_threshold_deg=float(normal_deg),
            coplanarity_deg=float(coplanarity_deg),
            outlier_ratio=0.5,
            min_plane_edge_length=float(min_edge_m),
            min_num_points=int(detector_min_num_points),
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
        )
    else:  # pragma: no cover - only on Open3D < 0.15
        boxes = _fallback_patches(pcd, dist_thresh, min_points, min_edge_m)

    # --- candidate (patch, point) distances, then a nearest-plane assignment ---
    band = 2.0 * dist_thresh
    cand: list[tuple[np.ndarray, np.ndarray, float, o3d.geometry.OrientedBoundingBox, float, tuple]] = []
    for box in boxes:
        R = np.asarray(box.R, dtype=np.float64)
        n = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-12)
        c = np.asarray(box.center, dtype=np.float64)
        d = -float(n @ c)
        lat = _lateral_obb(box, thickness_m=max(float(np.asarray(box.extent)[2]), band))
        idx = np.asarray(
            lat.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pts)),
            dtype=np.int64,
        )
        if len(idx) < min_points:
            continue
        sel = idx[np.abs(pts[idx] @ n + d) <= dist_thresh]
        if len(sel) < min_points:
            continue
        # PCA refit on the inliers
        q = pts[sel]
        cq = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - cq, full_matrices=False)
        n2 = vt[2] / (np.linalg.norm(vt[2]) + 1e-12)
        if n2 @ n < 0:
            n2 = -n2
        d2 = -float(n2 @ cq)
        sel = idx[np.abs(pts[idx] @ n2 + d2) <= dist_thresh]
        if len(sel) < min_points:
            continue
        ext = np.asarray(box.extent, dtype=np.float64)
        e = tuple(float(x) for x in np.sort(ext)[::-1][:2])
        cand.append((n2, sel, d2, box, float(e[0] * e[1]), e))

    if not cand:
        return []

    # nearest-plane assignment for points claimed by more than one patch
    best_pid = np.full(len(pts), -1, dtype=np.int64)
    best_dist = np.full(len(pts), np.inf)
    for pid, (n, sel, d, _box, _a, _e) in enumerate(cand):
        dist = np.abs(pts[sel] @ n + d)
        better = dist < best_dist[sel]
        best_pid[sel[better]] = pid
        best_dist[sel[better]] = dist[better]

    out: list[Patch] = []
    for pid, (n, _sel, d, box, area, e) in enumerate(cand):
        own = np.where(best_pid == pid)[0]
        if len(own) < min_points:
            continue
        basis = _plane_uv(n)
        out.append(
            Patch(
                normal=n,
                d=d,
                obb=box,
                inlier_idx=own,
                area_m2=area,
                label=_normal_label(n),
                extent_m=e,
                basis=basis,
                hull2d=_hull2d(pts[own], n, d, basis),
            )
        )
    out.sort(key=lambda p: -len(p.inlier_idx))
    return out


# --------------------------------------------------------------------------
# 2. snap
# --------------------------------------------------------------------------

def snap_to_patches(
    pcd: o3d.geometry.PointCloud, patches: list[Patch], edge_band_m: float = 0.012
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Project patch inliers exactly onto their plane; keep everything else as-is.

    Returns ``(pcd_snapped, detail_mask)`` where ``detail_mask[i]`` is True for
    points that belong to no patch (the detail: lips, rims, seams, clutter).
    Snapped points get the plane normal (signed to agree with the cloud normal
    where one exists); detail points keep the cloud's normals.
    """
    pts = np.asarray(pcd.points, dtype=np.float64).copy()
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals, dtype=np.float64).copy()
    else:
        nrm = np.zeros_like(pts)

    detail_mask = np.ones(len(pts), dtype=bool)
    for p in patches:
        idx = p.inlier_idx
        if len(idx) == 0:
            continue
        if edge_band_m > 0:
            # leave a band along the patch rim unsnapped so a rounded edge
            # (fillet) can survive as detail instead of being pulled flat
            idx = idx[p.edge_dist(pts[idx]) >= float(edge_band_m)]
            if len(idx) == 0:
                continue
        pts[idx] = p.project(pts[idx])
        sgn = np.sign(nrm[idx] @ p.normal)
        sgn[sgn == 0] = 1.0
        nrm[idx] = p.normal[None, :] * sgn[:, None]
        detail_mask[idx] = False

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.normals = o3d.utility.Vector3dVector(nrm)
    if pcd.has_colors():
        out.colors = pcd.colors
    return out, detail_mask


# --------------------------------------------------------------------------
# 3. hull closure for unobserved faces
# --------------------------------------------------------------------------

def hull_closure_points(
    planar_solid_mesh,
    observed_pts: np.ndarray,
    spacing_m: float = 0.004,
    exclude_within_m: float = 0.006,
) -> o3d.geometry.PointCloud:
    """Sample the planar solid's faces, drop everything near an observed point.

    What is left is the part of the closed planar hull that the capture never
    saw -- a table's underside, the hidden sides of a cabinet -- which is what
    makes the Poisson result closed instead of a floating sheet. Observed
    geometry always wins: any sample within ``exclude_within_m`` of an observed
    (snapped or detail) point is dropped.

    ``planar_solid_mesh`` is a ``trimesh.Trimesh`` or a path to one.
    """
    import trimesh

    if not isinstance(planar_solid_mesh, trimesh.Trimesh):
        loaded = trimesh.load(str(planar_solid_mesh), process=False, force="mesh")
        planar_solid_mesh = loaded
    tm = planar_solid_mesh
    if tm is None or len(tm.faces) == 0:
        return o3d.geometry.PointCloud()

    tm = tm.copy()
    try:
        if tm.is_watertight:
            tm.fix_normals()
    except Exception:
        pass

    area = float(tm.area)
    spacing_m = float(spacing_m)
    count = int(min(400_000, max(1000, 2.0 * area / (spacing_m ** 2))))
    try:
        samples, face_idx = trimesh.sample.sample_surface_even(
            tm, count, radius=spacing_m * 0.9, seed=0
        )
    except TypeError:  # older trimesh without seed=
        samples, face_idx = trimesh.sample.sample_surface_even(tm, count, radius=spacing_m * 0.9)
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) == 0:
        return o3d.geometry.PointCloud()
    normals = np.asarray(tm.face_normals, dtype=np.float64)[np.asarray(face_idx, dtype=np.int64)]

    observed_pts = np.asarray(observed_pts, dtype=np.float64)
    if len(observed_pts):
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(observed_pts).query(samples, k=1)
        keep = dist > float(exclude_within_m)
        samples = samples[keep]
        normals = normals[keep]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(samples)
    out.normals = o3d.utility.Vector3dVector(normals)
    return out


def hull_patches(
    hull_pcd: o3d.geometry.PointCloud,
    min_points: int = 150,
    angle_tol_deg: float = 5.0,
    offset_tol_m: float = 0.010,
    min_area_frac: float = 0.02,
    dropped_out: list | None = None,
) -> list[Patch]:
    """Group the closure samples by plane -> one :class:`Patch` per *unobserved*
    face of the planar solid (table underside, hidden sides).

    Poisson reconstructs those faces from the closure points alone, so they come
    back visibly wavy (user feedback). Treating them as patches lets the
    post-snap flatten them exactly like the observed ones.
    """
    pts = np.asarray(hull_pcd.points, dtype=np.float64)
    nrm = np.asarray(hull_pcd.normals, dtype=np.float64)
    if len(pts) < min_points:
        return []

    cos_tol = np.cos(np.deg2rad(float(angle_tol_deg)))
    groups: list[tuple[np.ndarray, float, list[int]]] = []
    for i in range(len(pts)):
        n = nrm[i]
        d = -float(n @ pts[i])
        for gi, (gn, gd, members) in enumerate(groups):
            if gn @ n >= cos_tol and abs(float(n @ pts[i] + gd)) <= offset_tol_m:
                members.append(i)
                break
        else:
            groups.append((n, d, [i]))

    # (N3) The greedy first-match pass above is order dependent: a planar solid
    # whose hull has many nearly-coplanar facets (cabinet_c: 13) comes back as
    # that many groups, and Poisson then closes the unobserved sides as a fan of
    # spiky facets. Union-find over the *group* planes with the same tolerance
    # merges them transitively -- 13 -> <= 6 on the cabinet.
    parent = list(range(len(groups)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(groups)):
        na, _da, ma = groups[a]
        ca = pts[np.asarray(ma, dtype=np.int64)].mean(axis=0)
        for b in range(a + 1, len(groups)):
            nb, _db, mb = groups[b]
            cb = pts[np.asarray(mb, dtype=np.int64)].mean(axis=0)
            if abs(float(na @ nb)) < cos_tol:
                continue
            off = max(abs(float(na @ (cb - ca))), abs(float(nb @ (ca - cb))))
            if off <= offset_tol_m:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[ra] = rb
    merged: dict[int, list[int]] = {}
    for gi, (_gn, _gd, members) in enumerate(groups):
        merged.setdefault(_find(gi), []).extend(members)
    groups = []
    for root, members in merged.items():
        m = np.asarray(members, dtype=np.int64)
        nsum = nrm[m].copy()
        ref = nsum[0]
        nsum[nsum @ ref < 0] *= -1.0
        gn = nsum.mean(axis=0)
        gn = gn / (np.linalg.norm(gn) + 1e-12)
        groups.append((gn, -float(gn @ pts[m].mean(axis=0)), members))

    out: list[Patch] = []
    for gn, _gd, members in groups:
        if len(members) < min_points:
            continue
        m = np.asarray(members, dtype=np.int64)
        q = pts[m]
        n = gn / (np.linalg.norm(gn) + 1e-12)
        d = -float(n @ q.mean(axis=0))
        basis = _plane_uv(n)
        uv = q @ basis.T
        eu = float(uv[:, 0].max() - uv[:, 0].min())
        ev = float(uv[:, 1].max() - uv[:, 1].min())
        centre = (-d) * n + np.array(
            [(uv[:, 0].max() + uv[:, 0].min()) / 2.0, (uv[:, 1].max() + uv[:, 1].min()) / 2.0]
        ) @ basis
        R = np.stack([basis[0], basis[1], n], axis=1)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        box = o3d.geometry.OrientedBoundingBox(
            center=centre, R=R, extent=np.array([max(eu, 1e-3), max(ev, 1e-3), 0.004])
        )
        out.append(
            Patch(
                normal=n, d=d, obb=box, inlier_idx=m, area_m2=eu * ev,
                label=_normal_label(n), extent_m=(eu, ev), basis=basis,
                hull2d=_hull2d(q, n, d, basis), source="hull",
            )
        )
    # (N3) drop slivers: a hull face under `min_area_frac` of the largest is a
    # corner chamfer of the planar solid, not a real unobserved face, and it is
    # exactly what shows up as a spike after Poisson.
    if out and min_area_frac > 0:
        a_max = max(p.area_m2 for p in out)
        keep = [p for p in out if p.area_m2 >= float(min_area_frac) * a_max]
        if dropped_out is not None:
            kept_ids = {id(p) for p in keep}
            for p_ in out:
                if id(p_) not in kept_ids:
                    dropped_out.extend(int(i) for i in p_.inlier_idx)
        out = keep
    out.sort(key=lambda p: -len(p.inlier_idx))
    return out


# --------------------------------------------------------------------------
# 4. post-snap of mesh vertices
# --------------------------------------------------------------------------

def post_snap_mesh(
    mesh: o3d.geometry.TriangleMesh,
    patches: list[Patch],
    dist_thresh: float = 0.003,
    edge_band_m: float = 0.012,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Project mesh vertices that are within ``2*dist_thresh`` of a patch plane
    (and laterally inside that patch's box) exactly onto the plane.

    Poisson smooths across the snapped cloud, so the flats come back ~0.3-0.8 mm
    wavy; this puts them exactly on the plane again while leaving vertices in
    the detail band untouched.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()
    if len(v) == 0 or not patches:
        return mesh, {}

    tol = 2.0 * float(dist_thresh)
    best_pid = np.full(len(v), -1, dtype=np.int64)
    best_dist = np.full(len(v), np.inf)
    for pid, p in enumerate(patches):
        lat = _lateral_obb(p.obb, thickness_m=max(float(np.asarray(p.obb.extent)[2]), 4.0 * tol))
        idx = np.asarray(
            lat.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(v)),
            dtype=np.int64,
        )
        if len(idx) == 0:
            continue
        dist = np.abs(v[idx] @ p.normal + p.d)
        ok = dist <= tol
        idx, dist = idx[ok], dist[ok]
        if len(idx) and edge_band_m > 0:
            keep_e = p.edge_dist(v[idx]) >= float(edge_band_m)
            idx, dist = idx[keep_e], dist[keep_e]
        better = dist < best_dist[idx]
        best_pid[idx[better]] = pid
        best_dist[idx[better]] = dist[better]

    per_patch = {}
    for pid, p in enumerate(patches):
        sel = np.where(best_pid == pid)[0]
        if len(sel) == 0:
            continue
        v[sel] = p.project(v[sel])
        resid = np.abs(v[sel] @ p.normal + p.d)
        per_patch[f"patch_{pid}"] = {
            "label": p.label,
            "source": p.source,
            "n_vertices_snapped": int(len(sel)),
            "post_snap_rms_mm": float(np.sqrt(np.mean(resid ** 2)) * 1000.0),
        }

    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(v)
    out.compute_vertex_normals()
    return out, per_patch


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Vertex loops around the mesh's boundary (edges used by exactly one face)."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = uniq[cnt == 1]
    if len(bnd) == 0:
        return []
    adj: dict[int, list[int]] = {}
    for a, b in bnd:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    used: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for a, b in bnd:
        a, b = int(a), int(b)
        if (a, b) in used:
            continue
        loop = [a, b]
        used.add((a, b))
        used.add((b, a))
        cur, prev = b, a
        while True:
            nxt = None
            for c in adj.get(cur, ()):
                if c != prev and (cur, c) not in used:
                    nxt = c
                    break
            if nxt is None:
                break
            used.add((cur, nxt))
            used.add((nxt, cur))
            if nxt == loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _fill_boundary_loops(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Fan-triangulate every boundary loop from its centroid.

    ``trimesh.repair.fill_holes`` bails out on loops that touch a non-manifold
    vertex (measured on table_a: 445 boundary edges, unchanged after five
    fill_holes passes, because 32 non-manifold edges sat on them). A centroid
    fan closes any loop unconditionally, which is what "closed where
    unobserved" needs.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    if len(f) == 0:
        return mesh, 0
    loops = _boundary_loops(f)
    if not loops:
        return mesh, 0
    new_v = [v]
    new_f = [f]
    n = len(v)
    for loop in loops:
        c = v[loop].mean(axis=0)
        ci = n
        n += 1
        new_v.append(c[None, :])
        tris = [[loop[i], loop[(i + 1) % len(loop)], ci] for i in range(len(loop))]
        new_f.append(np.asarray(tris, dtype=np.int64))
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack(new_v)),
        o3d.utility.Vector3iVector(np.vstack(new_f)),
    )
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_unreferenced_vertices()
    return out, len(loops)


def _close_holes(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove non-manifold junk and bridge the small boundary loops left behind
    by the density/distance trim and by degenerate-triangle removal.

    Poisson gives a closed shell, but trimming it by density and by distance to
    the fused cloud punches a few hundred boundary edges in it; without this the
    result is never ``is_watertight``.
    """
    import trimesh

    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_duplicated_vertices()
    out.remove_non_manifold_edges()
    out.remove_unreferenced_vertices()
    if len(out.triangles) == 0:
        return out

    tm = trimesh.Trimesh(
        vertices=np.asarray(out.vertices), faces=np.asarray(out.triangles), process=False
    )
    try:
        trimesh.repair.fill_holes(tm)
    except Exception:
        pass
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)),
    )
    out.remove_unreferenced_vertices()

    # anything fill_holes could not bridge (loops through non-manifold vertices)
    for _ in range(4):
        out.remove_non_manifold_edges()
        out.remove_unreferenced_vertices()
        out, n_loops = _fill_boundary_loops(out)
        if n_loops == 0:
            break
    out.remove_duplicated_triangles()
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


# --------------------------------------------------------------------------
# stats helpers
# --------------------------------------------------------------------------

def _dist_to_patch_planes(pts: np.ndarray, patches: list[Patch], lateral_pad_m: float = 0.05):
    """Min |distance| to any patch plane, over patches whose lateral footprint
    (box extended infinitely along its normal) contains the point.

    Returns ``(dist, has_patch)``; ``dist`` is ``inf`` where no patch applies.
    """
    dist = np.full(len(pts), np.inf)
    if len(pts) == 0 or not patches:
        return dist, np.zeros(len(pts), dtype=bool)
    for p in patches:
        lat = _lateral_obb(p.obb, thickness_m=10.0)  # effectively an infinite prism
        ext = np.asarray(lat.extent).copy()
        ext[0] += 2.0 * lateral_pad_m
        ext[1] += 2.0 * lateral_pad_m
        lat = o3d.geometry.OrientedBoundingBox(center=lat.center, R=np.asarray(lat.R), extent=ext)
        idx = np.asarray(
            lat.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pts)),
            dtype=np.int64,
        )
        if len(idx) == 0:
            continue
        dd = np.abs(pts[idx] @ p.normal + p.d)
        dist[idx] = np.minimum(dist[idx], dd)
    return dist, np.isfinite(dist)


def _detail_band_counts(pts: np.ndarray, patches: list[Patch], lo: float = 0.004, hi: float = 0.025) -> dict:
    d, has = _dist_to_patch_planes(pts, patches)
    in_band = has & (d >= lo) & (d <= hi)
    return {
        "n_points": int(len(pts)),
        "n_over_patch_footprint": int(has.sum()),
        "n_in_band_4_25mm": int(in_band.sum()),
        "frac_in_band": float(in_band.sum() / max(1, int(has.sum()))),
    }


def _rms_to_mesh_mm(query_pts: np.ndarray, mesh: o3d.geometry.TriangleMesh):
    if len(query_pts) == 0 or len(mesh.triangles) == 0:
        return None
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    q = o3d.core.Tensor(np.asarray(query_pts, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    d = scene.compute_distance(q).numpy()
    return float(np.sqrt(np.mean(np.square(d))) * 1000.0)


def _edge_length_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    if len(f) == 0:
        return {}
    e = np.vstack([v[f[:, 0]] - v[f[:, 1]], v[f[:, 1]] - v[f[:, 2]], v[f[:, 2]] - v[f[:, 0]]])
    L = np.linalg.norm(e, axis=1) * 1000.0
    q1, q2, q3 = np.percentile(L, [25, 50, 75])
    return {
        "median_mm": round(float(q2), 3),
        "iqr_mm": round(float(q3 - q1), 3),
        "p25_mm": round(float(q1), 3),
        "p75_mm": round(float(q3), 3),
        "n_tris": int(len(f)),
    }


def uniform_remesh(
    mesh: o3d.geometry.TriangleMesh, target_edge_m: float = 0.003, taubin_iters: int = 3
) -> tuple[o3d.geometry.TriangleMesh, str]:
    """Re-tile the mesh with near-uniform triangles.

    Quadric decimation minimises geometric error, which on a Poisson shell means
    huge slivers over the flats and dense clutter on the detail -- visibly
    non-uniform (user feedback). ``pymeshlab``'s isotropic explicit remeshing is
    the better tool but is not installed in this env (checked: ModuleNotFoundError,
    and installing is out of scope), so this uses Open3D's vertex clustering,
    which lays vertices on a regular ``target_edge_m`` lattice, plus a short
    Taubin pass to take the staircase off. The flats are put back exactly on
    their planes by the post-snap that follows.
    """
    method = "vertex_clustering"
    try:
        import pymeshlab  # noqa: F401

        method = "pymeshlab_isotropic"
    except Exception:
        pass

    if method == "pymeshlab_isotropic":  # pragma: no cover - not installed here
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        )
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PureValue(float(target_edge_m)), iterations=5
        )
        m = ms.current_mesh()
        out = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(m.vertex_matrix()),
            o3d.utility.Vector3iVector(m.face_matrix()),
        )
    else:
        out = mesh.simplify_vertex_clustering(
            voxel_size=float(target_edge_m),
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        out.remove_degenerate_triangles()
        out.remove_duplicated_triangles()
        out.remove_duplicated_vertices()
        out.remove_unreferenced_vertices()
        if taubin_iters > 0 and len(out.triangles):
            out = out.filter_smooth_taubin(number_of_iterations=int(taubin_iters))

    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out, method


# --------------------------------------------------------------------------
# F11a: colour transfer
# --------------------------------------------------------------------------

def colorize(
    mesh: o3d.geometry.TriangleMesh,
    cloud,
    max_dist_m: float = 0.05,
    default=None,
) -> o3d.geometry.TriangleMesh:
    """Nearest-neighbour vertex-colour transfer from a coloured cloud onto ``mesh``.

    ``cloud`` may be an Open3D ``PointCloud`` (points + colors), a
    ``TriangleMesh`` (vertices + vertex_colors) or a ``(points, colours)``
    tuple; colours are floats in [0, 1]. Every mesh vertex takes the colour of
    the nearest coloured point; vertices further than ``max_dist_m`` from any of
    them (the unobserved hull faces, typically) fall back to ``default`` if
    given, else the *median* colour of the source cloud (a plain grey 0.66 only
    when the source has no colours at all).

    The GLB writer (:func:`surfcap.detail_mesh._write_glb`) and Open3D's PLY
    writer both pick ``vertex_colors`` up automatically, so calling this before
    the export is all that is needed to get colour into both files.
    """
    from scipy.spatial import cKDTree

    if isinstance(cloud, tuple):
        cpts, ccol = cloud
    elif isinstance(cloud, o3d.geometry.TriangleMesh):
        cpts = np.asarray(cloud.vertices, dtype=np.float64)
        ccol = np.asarray(cloud.vertex_colors, dtype=np.float64)
    else:
        cpts = np.asarray(cloud.points, dtype=np.float64)
        ccol = np.asarray(cloud.colors, dtype=np.float64)
    cpts = np.asarray(cpts, dtype=np.float64).reshape(-1, 3)
    ccol = np.asarray(ccol, dtype=np.float64).reshape(-1, 3)

    v = np.asarray(mesh.vertices, dtype=np.float64)
    out = o3d.geometry.TriangleMesh(mesh)
    if len(v) == 0:
        return out
    if default is not None:
        fill = np.asarray(default, dtype=np.float64)
    elif len(ccol):
        fill = np.median(ccol, axis=0)
    else:
        fill = np.array([0.66, 0.66, 0.66])
    col = np.tile(fill, (len(v), 1))
    if len(cpts) and len(ccol) == len(cpts):
        dist, idx = cKDTree(cpts).query(v, k=1)
        ok = dist <= float(max_dist_m)
        col[ok] = ccol[idx[ok]]
    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(col, 0.0, 1.0))
    return out


def _scene_colour_source(target_json_path, max_points: int = 400_000):
    """Fallback coloured point source for a scene: ``target.ply`` else ``target_clean.ply``.

    The whole-scene TSDF mesh is *supposed* to carry vertex colour (D1/F9's
    original design fell back to that), but on pipeline-produced runs the
    per-scene ``debug/tsdf_mesh.ply`` frequently does not, or is unavailable to
    the caller (e.g. when colourizing an already-written mesh standalone). Both
    ``target.ply`` (the cleaned, colour-carrying object cloud, world frame) and
    ``target_clean.ply`` are scene-root siblings of ``target.json``. Returns
    ``(points, colours)`` or ``None`` if neither file exists / carries colour.
    """
    scene_dir = Path(target_json_path).resolve().parent
    for name in ("target.ply", "target_clean.ply"):
        p = scene_dir / name
        if not p.exists():
            continue
        cloud = o3d.io.read_point_cloud(str(p))
        if len(cloud.points) and cloud.has_colors():
            pts = np.asarray(cloud.points, dtype=np.float64)
            cols = np.asarray(cloud.colors, dtype=np.float64)
            if len(pts) > max_points:
                sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
                pts, cols = pts[sel], cols[sel]
            return pts, cols
    return None


# --------------------------------------------------------------------------
# F11b: plane-first mesh -- exact planes, exact corners, measured displacement
# --------------------------------------------------------------------------

def _inside_poly(q: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test (ring open, any orientation)."""
    x, y = q[:, 0], q[:, 1]
    inside = np.zeros(len(q), dtype=bool)
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        cond = (yi > y) != (yj > y)
        denom = (yj - yi) if abs(yj - yi) > 1e-18 else 1e-18
        xint = xi + (y - yi) * (xj - xi) / denom
        inside ^= cond & (x < xint)
        j = i
    return inside


def _poly_dist2d(q: np.ndarray, ring: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Unsigned distance from each 2D point to the polygon's boundary."""
    a = ring
    b = np.roll(ring, -1, axis=0)
    ab = b - a
    L2 = np.maximum((ab * ab).sum(axis=1), 1e-18)
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        qq = q[s:s + chunk]
        ap = qq[:, None, :] - a[None, :, :]
        t = np.clip((ap * ab[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        closest = a[None] + t[:, :, None] * ab[None]
        out[s:s + chunk] = np.linalg.norm(qq[:, None, :] - closest, axis=2).min(axis=1)
    return out


def _ring_from_patch(p: Patch, pts: np.ndarray, concave_alpha_m: float = 0.03) -> np.ndarray | None:
    """The patch's boundary polygon in its own (u, v) basis.

    Concave hull (shapely, via :func:`surfcap.planar_mesh.concave_ring`) for
    observed patches -- an L-shaped or notched face is common -- with the
    detector's convex ``hull2d`` as the fallback.
    """
    if p.basis is None:
        return None
    # NB: a hull patch's inlier_idx indexes the *closure* cloud, not ``pts`` --
    # it only ever gets its stored convex hull2d.
    q = None
    if p.source == "observed" and len(p.inlier_idx):
        q = (pts[p.inlier_idx] - (-p.d) * p.normal) @ p.basis.T
    ring = None
    if q is not None and len(q) >= 3:
        try:
            from surfcap.planar_mesh import concave_ring

            r = concave_ring(q, alpha_m=float(concave_alpha_m))
            if len(r) >= 3 and abs(_poly_area(r)) > 1e-6:
                ring = np.asarray(r, dtype=np.float64)
        except Exception:
            ring = None
    if ring is None and p.hull2d is not None and len(p.hull2d) >= 3:
        ring = np.asarray(p.hull2d, dtype=np.float64)
    if ring is None or len(ring) < 3:
        return None
    if _poly_area(ring) < 0:
        ring = ring[::-1]
    return ring


def _poly_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _to2d(p: Patch, pts3: np.ndarray) -> np.ndarray:
    return (pts3 - (-p.d) * p.normal) @ p.basis.T


def _to3d(p: Patch, q2: np.ndarray) -> np.ndarray:
    return (-p.d) * p.normal + q2 @ p.basis


def _clip_ring(ring: np.ndarray, a2: np.ndarray, d2: np.ndarray, keep_sign: float) -> np.ndarray | None:
    """Clip a (possibly concave) 2D ring to one side of a line, via shapely."""
    try:
        from shapely.geometry import Polygon

        m2 = np.array([-d2[1], d2[0]]) * float(keep_sign)
        big = 50.0
        c = a2 + m2 * (big * 0.5)
        quad = np.array(
            [a2 - d2 * big, a2 + d2 * big, a2 + d2 * big + m2 * big, a2 - d2 * big + m2 * big]
        )
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        res = poly.intersection(Polygon(quad))
        if res.is_empty:
            return None
        if res.geom_type == "MultiPolygon":
            res = max(res.geoms, key=lambda g: g.area)
        if res.geom_type != "Polygon" or res.area < 1e-7:
            return None
        out = np.asarray(res.exterior.coords, dtype=np.float64)[:-1]
        return out if len(out) >= 3 else None
    except Exception:
        return None


def merge_coplanar_patches(
    patches: list[Patch],
    pts: np.ndarray,
    angle_deg: float = 6.0,
    offset_m: float = 0.004,
) -> tuple[list[Patch], int]:
    """Merge observed patches that are the same plane seen in pieces.

    ``detect_planar_patches`` happily returns a face as two or three patches when
    something (a groove, a seam, a sparse strip) breaks it up -- on the synthetic
    grooved box the top comes back as two patches 0.35 deg apart. Left alone they
    become two faces with a gap between them; merged, they are one exact plane and
    the groove goes back to being *detail*, which is what the displacement pass is
    for. Hull (unobserved) patches are never merged: their ``inlier_idx`` indexes
    a different cloud.
    """
    obs = [i for i, p in enumerate(patches) if p.source == "observed"]
    cos_tol = np.cos(np.deg2rad(float(angle_deg)))
    parent = {i: i for i in obs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_merged = 0
    for a_i, i in enumerate(obs):
        for j in obs[a_i + 1:]:
            pi, pj = patches[i], patches[j]
            sgn = 1.0 if pi.normal @ pj.normal >= 0 else -1.0
            if sgn * float(pi.normal @ pj.normal) < cos_tol:
                continue
            ci = (-pi.d) * pi.normal
            cj = (-pj.d) * pj.normal
            if abs(float(pj.normal @ ci + pj.d)) > offset_m:
                continue
            if abs(float(pi.normal @ cj + pi.d)) > offset_m:
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
                n_merged += 1
    if n_merged == 0:
        return patches, 0

    groups: dict[int, list[int]] = {}
    for i in obs:
        groups.setdefault(find(i), []).append(i)
    out = [p for p in patches if p.source != "observed"]
    for root, members in groups.items():
        if len(members) == 1:
            out.append(patches[members[0]])
            continue
        idx = np.unique(np.concatenate([patches[m].inlier_idx for m in members]))
        q = pts[idx]
        c = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        n = vt[2] / (np.linalg.norm(vt[2]) + 1e-12)
        if n @ patches[root].normal < 0:
            n = -n
        d = -float(n @ c)
        basis = _plane_uv(n)
        uv = (q - (-d) * n) @ basis.T
        eu = float(uv[:, 0].max() - uv[:, 0].min())
        ev = float(uv[:, 1].max() - uv[:, 1].min())
        centre = (-d) * n + np.array(
            [(uv[:, 0].max() + uv[:, 0].min()) / 2.0, (uv[:, 1].max() + uv[:, 1].min()) / 2.0]
        ) @ basis
        R = np.stack([basis[0], basis[1], n], axis=1)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        box = o3d.geometry.OrientedBoundingBox(
            center=centre, R=R, extent=np.array([max(eu, 1e-3), max(ev, 1e-3), 0.006])
        )
        out.append(
            Patch(normal=n, d=d, obb=box, inlier_idx=idx, area_m2=eu * ev,
                  label=_normal_label(n), extent_m=(eu, ev), basis=basis,
                  hull2d=_hull2d(q, n, d, basis), source="observed")
        )
    out.sort(key=lambda p: -p.area_m2)
    return out, n_merged


def face_polygons(
    patches: list[Patch],
    pts: np.ndarray,
    max_gap_m: float = 0.04,
    angle_lo_deg: float = 60.0,
    concave_alpha_m: float = 0.03,
) -> tuple[list[dict], dict, int]:
    """One boundary polygon per patch, cut/extended to the neighbouring planes.

    For every pair of patches meeting at ``angle_lo_deg``..180-``angle_lo_deg``
    degrees whose polygons come within ``max_gap_m`` of the planes' intersection
    line, both polygons are (a) *extended*: boundary vertices within
    ``max_gap_m`` of the line are projected onto it -- the
    :func:`surfcap.planar_mesh.snap_adjacent` move -- and (b) *clipped*: the
    polygon is intersected with the half-plane on its own side of the line, so
    any overhang past the corner is cut off. The resulting corner vertices are
    then *exactly* the intersection of the two clip lines, i.e. the analytic
    plane-plane-plane triple point, which is what makes the corners crisp.

    Returns ``(faces, lines, n_pairs)``; ``faces[i]`` carries ``ring`` (2D),
    ``patch`` and ``line_ids`` (registry keys the ring was snapped to), and
    ``lines[key] = (p0, dir)`` is the canonical 3D line shared by both faces.
    """
    rings: list[np.ndarray | None] = [_ring_from_patch(p, pts, concave_alpha_m) for p in patches]
    lines: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    used: list[set] = [set() for _ in patches]
    cos_lo = np.cos(np.deg2rad(float(angle_lo_deg)))
    n_pairs = 0

    from surfcap.planar_mesh import _plane_intersection_line

    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            if rings[i] is None or rings[j] is None:
                continue
            pi, pj = patches[i], patches[j]
            if abs(float(pi.normal @ pj.normal)) > cos_lo:
                continue
            line = _plane_intersection_line(
                pi.normal, (-pi.d) * pi.normal, pj.normal, (-pj.d) * pj.normal
            )
            if line is None:
                continue
            p0, dirv = line
            hit = True
            for k, pk in ((i, pi), (j, pj)):
                a2, d2 = _line2d(pk, p0, dirv)
                m2 = np.array([-d2[1], d2[0]])
                sd = (rings[k] - a2) @ m2
                if np.abs(sd).min() > max_gap_m:
                    hit = False
                    break
            if not hit:
                continue
            key = (i, j)
            lines[key] = (p0, dirv)
            n_pairs += 1
            for k in (i, j):
                pk = patches[k]
                a2, d2 = _line2d(pk, p0, dirv)
                m2 = np.array([-d2[1], d2[0]])
                r = rings[k].copy()
                sd = (r - a2) @ m2
                # side to keep = where the polygon's mass is
                keep = 1.0 if float(np.median(sd)) >= 0 else -1.0
                # (a) extend: pull near-line boundary vertices onto the line
                near = np.abs(sd) <= max_gap_m
                if near.any():
                    r[near] = r[near] - np.outer(sd[near], m2)
                # (b) clip: cut the overhang past the line
                r2 = _clip_ring(r, a2, d2, keep)
                if r2 is not None and len(r2) >= 3:
                    r = r2
                rings[k] = r
                used[k].add(key)

    faces = []
    for i, (p, r) in enumerate(zip(patches, rings)):
        if r is None or len(r) < 3 or abs(_poly_area(r)) < 1e-7:
            continue
        if _poly_area(r) < 0:
            r = r[::-1]
        faces.append({"patch": p, "ring": r, "line_ids": sorted(used[i]), "pid": i})
    return faces, lines, n_pairs


def _line2d(p: Patch, p0: np.ndarray, dirv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A 3D line expressed in a patch's (u, v) basis: (point, unit direction)."""
    a2 = _to2d(p, p0.reshape(1, 3))[0]
    d2 = dirv @ p.basis.T
    nrm = np.linalg.norm(d2)
    if nrm < 1e-12:
        return a2, np.array([1.0, 0.0])
    return a2, d2 / nrm


def _grid_triangulate(
    ring: np.ndarray,
    h: float,
    edge_samples: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary-conforming ~uniform triangulation of a 2D polygon.

    Interior vertices sit on a regular ``h`` lattice, at least ``0.6 h`` inside
    the boundary; the boundary itself is resampled at ``h`` (or, for edges that
    lie on a shared plane-intersection line, at the *global* lattice positions
    supplied in ``edge_samples``, so the neighbouring face produces bit-identical
    vertices there and the two weld). Delaunay over the union, then triangles
    whose centroid falls outside the polygon (or that span a concave notch) are
    dropped. Returns ``(pts2d, tris, is_boundary)``.
    """
    from scipy.spatial import Delaunay

    bnd = []
    n = len(ring)
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        if edge_samples is not None and k in edge_samples:
            seg = edge_samples[k]
        else:
            L = float(np.linalg.norm(b - a))
            m = max(1, int(np.ceil(L / h)))
            t = np.linspace(0.0, 1.0, m + 1)[:-1]
            seg = a[None, :] + t[:, None] * (b - a)[None, :]
        if len(seg):
            bnd.append(np.asarray(seg, dtype=np.float64).reshape(-1, 2))
    bnd = np.vstack(bnd) if bnd else ring.copy()
    # de-duplicate boundary samples
    if len(bnd) > 1:
        from scipy.spatial import cKDTree

        keep = np.ones(len(bnd), dtype=bool)
        for a_, b_ in cKDTree(bnd).query_pairs(h * 0.25, output_type="ndarray"):
            if keep[a_]:
                keep[b_] = False
        bnd = bnd[keep]

    lo = ring.min(axis=0)
    hi = ring.max(axis=0)
    gx = np.arange(np.floor(lo[0] / h) * h, hi[0] + h, h)
    gy = np.arange(np.floor(lo[1] / h) * h, hi[1] + h, h)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    if len(G):
        ok = _inside_poly(G, ring) & (_poly_dist2d(G, ring) > 0.6 * h)
        G = G[ok]

    pts = np.vstack([bnd, G]) if len(G) else bnd
    is_bnd = np.zeros(len(pts), dtype=bool)
    is_bnd[: len(bnd)] = True
    if len(pts) < 3:
        return pts, np.zeros((0, 3), dtype=np.int64), is_bnd
    try:
        tri = Delaunay(pts).simplices
    except Exception:
        return pts, np.zeros((0, 3), dtype=np.int64), is_bnd
    cen = pts[tri].mean(axis=1)
    good = _inside_poly(cen, ring)
    e0 = np.linalg.norm(pts[tri[:, 0]] - pts[tri[:, 1]], axis=1)
    e1 = np.linalg.norm(pts[tri[:, 1]] - pts[tri[:, 2]], axis=1)
    e2 = np.linalg.norm(pts[tri[:, 2]] - pts[tri[:, 0]], axis=1)
    good &= np.maximum.reduce([e0, e1, e2]) < 4.0 * h
    tri = tri[good]
    # CCW
    v0 = pts[tri[:, 1]] - pts[tri[:, 0]]
    v1 = pts[tri[:, 2]] - pts[tri[:, 0]]
    flip = (v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]) < 0
    tri[flip] = tri[flip][:, ::-1]
    return pts, tri.astype(np.int64), is_bnd


def _ring_median(vals: np.ndarray, tris: np.ndarray, n_v: int) -> np.ndarray:
    """One-ring median filter (self included) over a triangle mesh's vertex graph.

    A median filter is the edge-preserving smoother we actually want here: an
    isolated spike (one vertex that caught a stray cloud point) is replaced by
    its neighbourhood's value, while a *line* feature -- a drawer seam, a rim --
    has half its 1-ring on the feature and survives.
    """
    if len(tris) == 0 or n_v == 0:
        return vals
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    order = np.argsort(e[:, 0], kind="stable")
    e = e[order]
    starts = np.searchsorted(e[:, 0], np.arange(n_v))
    ends = np.searchsorted(e[:, 0], np.arange(n_v), side="right")
    out = vals.copy()
    for i in range(n_v):
        s, t = starts[i], ends[i]
        if t <= s:
            continue
        nb = e[s:t, 1]
        out[i] = np.median(np.concatenate([vals[nb], vals[i:i + 1]]))
    return out


def face_displacement(
    face_pts3: np.ndarray,
    normal: np.ndarray,
    d: float,
    basis: np.ndarray,
    cloud_pts: np.ndarray,
    cloud_nrm: np.ndarray | None,
    radius_m: float = 0.006,
    k: int = 8,
    band_m: float = 0.030,
    noise_floor_m: float = 0.004,
    clamp_m: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of the fused cloud from the face plane, per face vertex.

    For every face vertex: take the fused-cloud points within ``radius_m``
    *in-plane* (k nearest of them), project them onto the face normal and take
    the median. Candidates are restricted to a +-``band_m`` slab around the plane
    and, when normals are available, to points facing roughly the same way, so
    the *other* side of an 18 mm slab cannot pull a face through itself.

    Returns ``(d_raw, has_support)``; the caller applies the noise floor.
    """
    from scipy.spatial import cKDTree

    n_v = len(face_pts3)
    if n_v == 0 or len(cloud_pts) == 0:
        return np.zeros(n_v), np.zeros(n_v, dtype=bool)
    s = cloud_pts @ normal + d
    sel = np.abs(s) <= float(band_m)
    if cloud_nrm is not None and len(cloud_nrm) == len(cloud_pts):
        sel &= (cloud_nrm @ normal) > 0.2
    if sel.sum() < 3:
        return np.zeros(n_v), np.zeros(n_v, dtype=bool)
    cp = cloud_pts[sel]
    cs = s[sel]
    c2 = (cp - (-d) * normal) @ basis.T
    q2 = (face_pts3 - (-d) * normal) @ basis.T
    tree = cKDTree(c2)
    kk = int(min(k, len(c2)))
    dist, idx = tree.query(q2, k=kk, distance_upper_bound=float(radius_m))
    dist = np.atleast_2d(dist.T).T.reshape(n_v, kk)
    idx = np.atleast_2d(idx.T).T.reshape(n_v, kk)
    valid = np.isfinite(dist)
    idx = np.clip(idx, 0, len(c2) - 1)
    sv = np.where(valid, cs[idx], np.nan)
    cnt = valid.sum(axis=1)
    out = np.zeros(n_v)
    has = cnt > 0
    if has.any():
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(sv[has], axis=1)
        out[has] = np.clip(np.nan_to_num(med), -float(clamp_m), float(clamp_m))
    return out, has


def plane_first_mesh(
    pcd: o3d.geometry.PointCloud,
    patches: list[Patch],
    grid_m: float = 0.003,
    noise_floor_m: float = 0.004,
    clamp_m: float = 0.025,
    radius_m: float = 0.006,
    k_neighbours: int = 8,
    edge_band_m: float = 0.012,
    max_gap_m: float = 0.04,
    min_face_extent_m: float = 0.02,
    weld_tol_m: float = 0.0005,
    smooth_detail_iters: int = 5,
    colour_source=None,
) -> tuple[o3d.geometry.TriangleMesh, dict, o3d.geometry.TriangleMesh]:
    """A planar solid at the current level of detail.

    Faces are *exactly* their fitted planes; corners come from plane-plane
    intersections (:func:`face_polygons`); each face is tiled on a uniform
    ``grid_m`` lattice; and only then is the fused cloud allowed to push
    individual vertices off the plane, and only where it actually says so (a
    median over the k nearest cloud points within ``radius_m`` in-plane, bigger
    than ``noise_floor_m``, one-ring-median filtered). Seams, lips and rims
    survive as displacement; fusion noise does not.

    The reverse of :func:`hybrid_mesh`'s Poisson path, which meshes first and
    flattens afterwards and therefore can only ever *approximate* a flat face.
    """
    t0 = time.time()
    pts = np.asarray(pcd.points, dtype=np.float64)
    nrm = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None

    patches, n_merged_patches = merge_coplanar_patches(patches, pts)
    # a face has to be a face: a strip narrower than ``min_face_extent_m`` (a
    # groove floor, a 10 mm rim band) is *detail*, and is left to the
    # displacement pass rather than becoming its own plane.
    faces_in = [p for p in patches if min(p.extent_m) >= float(min_face_extent_m)]
    n_thin = len(patches) - len(faces_in)
    faces, lines, n_pairs = face_polygons(faces_in, pts, max_gap_m=max_gap_m)
    t_poly = time.time()

    # --- boundary edges that live on a shared intersection line -------------
    # both faces sample such an edge at the *same* global lattice positions, so
    # their vertices coincide exactly and the weld closes the seam.
    for fc in faces:
        p = fc["patch"]
        ring = fc["ring"]
        shared: dict[int, np.ndarray] = {}
        on_line = np.zeros(len(ring), dtype=bool)
        for key in fc["line_ids"]:
            p0, dirv = lines[key]
            a2, d2 = _line2d(p, p0, dirv)
            m2 = np.array([-d2[1], d2[0]])
            sd = np.abs((ring - a2) @ m2)
            near = sd <= 1e-6
            on_line |= near
            for kk in range(len(ring)):
                k2 = (kk + 1) % len(ring)
                if not (near[kk] and near[k2]):
                    continue
                t_a = float((ring[kk] - a2) @ d2)
                t_b = float((ring[k2] - a2) @ d2)
                lo, hi = (t_a, t_b) if t_a < t_b else (t_b, t_a)
                ts = np.arange(np.ceil(lo / grid_m) * grid_m, hi, grid_m)
                ts = ts[(ts > lo + 0.25 * grid_m) & (ts < hi - 0.25 * grid_m)]
                if t_a > t_b:
                    ts = ts[::-1]
                seg = np.vstack([ring[kk][None, :], a2[None, :] + ts[:, None] * d2[None, :]])
                shared[kk] = seg
        fc["shared_edges"] = shared
        fc["on_line"] = on_line

    # --- tile + displace each face -----------------------------------------
    V: list[np.ndarray] = []
    V_raw: list[np.ndarray] = []
    F: list[np.ndarray] = []
    per_face = []
    off = 0
    n_disp_total = 0
    max_disp = 0.0
    for fi, fc in enumerate(faces):
        p = fc["patch"]
        ring = fc["ring"]
        q2, tris, is_bnd = _grid_triangulate(ring, grid_m, fc["shared_edges"])
        if len(tris) == 0:
            continue
        v3 = _to3d(p, q2)

        d_raw, has = face_displacement(
            v3, p.normal, p.d, p.basis, pts, nrm,
            radius_m=radius_m, k=k_neighbours, noise_floor_m=noise_floor_m,
            clamp_m=clamp_m,
        )
        d_use = np.where(np.abs(d_raw) > float(noise_floor_m), d_raw, 0.0)
        d_use = _ring_median(d_use, tris, len(q2))

        # edge band: only displace the rim when the cloud consistently says so
        edist = _poly_dist2d(q2, ring)
        band = edist < float(edge_band_m)
        band_med = float(np.median(np.abs(d_raw[band]))) if band.any() else 0.0
        band_kept = bool(band_med >= float(noise_floor_m))
        if not band_kept:
            d_use[band] = 0.0
        # vertices sitting exactly on a shared intersection line are the crisp
        # corner itself and are never moved (also what keeps the seam welded)
        for key in fc["line_ids"]:
            p0, dirv = lines[key]
            a2, d2 = _line2d(p, p0, dirv)
            m2 = np.array([-d2[1], d2[0]])
            d_use[np.abs((q2 - a2) @ m2) <= 1e-6] = 0.0

        # --- selective smoothing: Taubin on the *detail* (displaced) vertices
        # only, along the face normal, with every on-plane vertex (and every
        # crease/intersection-line vertex) locked. Isolated fusion spikes that
        # survived the median pass are flattened; the seam, whose whole 1-ring
        # is displaced, keeps its depth.
        d_smooth = d_use
        if smooth_detail_iters > 0:
            mask_detail = np.abs(d_use) > 1e-9
            if mask_detail.any():
                d_smooth = _taubin_scalar(
                    d_use, tris, mask_detail, iters=int(smooth_detail_iters)
                )
                if not band_kept:
                    d_smooth[band] = 0.0
                for key in fc["line_ids"]:
                    p0, dirv = lines[key]
                    a2, d2 = _line2d(p, p0, dirv)
                    m2 = np.array([-d2[1], d2[0]])
                    d_smooth[np.abs((q2 - a2) @ m2) <= 1e-6] = 0.0

        V_raw.append(v3 + np.outer(d_use, p.normal))
        d_use = d_smooth
        v3 = v3 + np.outer(d_use, p.normal)
        nz = int((np.abs(d_use) > 1e-9).sum())
        n_disp_total += nz
        max_disp = max(max_disp, float(np.abs(d_use).max()) if len(d_use) else 0.0)
        per_face.append(
            {
                "face": fi,
                "patch_id": int(fc["pid"]),
                "label": p.label,
                "source": p.source,
                "area_m2": round(float(abs(_poly_area(ring))), 5),
                "n_vertices": int(len(q2)),
                "n_tris": int(len(tris)),
                "n_displaced": nz,
                "displaced_frac": round(float(nz / max(1, len(q2))), 4),
                "max_disp_mm": round(float(np.abs(d_use).max() * 1000.0) if len(d_use) else 0.0, 3),
                "median_abs_disp_mm": round(float(np.median(np.abs(d_use)) * 1000.0) if len(d_use) else 0.0, 3),
                "n_ring_vertices": int(len(ring)),
                "n_shared_edges": int(len(fc["shared_edges"])),
                "edge_band_displaced": band_kept,
                "edge_band_median_abs_mm": round(band_med * 1000.0, 3),
                "support_frac": round(float(has.mean()) if len(has) else 0.0, 4),
            }
        )
        V.append(v3)
        F.append(tris + off)
        off += len(v3)
    t_tile = time.time()

    if not V:
        return o3d.geometry.TriangleMesh(), {"n_faces": 0}, o3d.geometry.TriangleMesh()

    v_all = np.vstack(V)
    f_all = np.vstack(F)

    # --- sew: weld coincident vertices along the shared intersection edges ---
    v_all, f_all, n_welded, inv = _weld_vertices(v_all, f_all, tol=weld_tol_m)
    # same topology, pre-smoothing positions -- the A/B reference
    v_raw = np.zeros_like(v_all)
    cnt = np.bincount(inv, minlength=len(v_all)).astype(float)
    np.add.at(v_raw, inv, np.vstack(V_raw))
    v_raw /= np.maximum(cnt, 1.0)[:, None]
    mesh_raw = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v_raw), o3d.utility.Vector3iVector(f_all)
    )

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v_all), o3d.utility.Vector3iVector(f_all)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    n_bnd_before = _n_boundary_edges(np.asarray(mesh.triangles))
    for _ in range(3):
        mesh, n_loops = _fill_boundary_loops(mesh)
        if n_loops == 0:
            break
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if colour_source is not None:
        mesh = colorize(mesh, colour_source, max_dist_m=0.015)

    f_out = np.asarray(mesh.triangles)
    n_bnd = _n_boundary_edges(f_out)
    stats = {
        "mode": "plane_first",
        "grid_m": float(grid_m),
        "noise_floor_m": float(noise_floor_m),
        "clamp_m": float(clamp_m),
        "n_faces": len(per_face),
        "n_coplanar_merges": int(n_merged_patches),
        "n_thin_patches_as_detail": int(n_thin),
        "n_intersection_pairs": int(n_pairs),
        "n_displaced_vertices": int(n_disp_total),
        "displaced_frac_overall": round(float(n_disp_total / max(1, off)), 4),
        "max_displacement_mm": round(max_disp * 1000.0, 3),
        "n_welded_vertices": int(n_welded),
        "n_boundary_edges_before_fill": int(n_bnd_before),
        "n_boundary_edges": int(n_bnd),
        "closed_no_boundary": bool(n_bnd == 0 and len(f_out) > 0),
        "tris": int(len(f_out)),
        "verts": int(len(np.asarray(mesh.vertices))),
        "watertight": bool(mesh.is_watertight()) if len(f_out) else False,
        "per_face": per_face,
        "timings_s": {
            "polygons": round(t_poly - t0, 2),
            "tile_displace": round(t_tile - t_poly, 2),
            "sew": round(time.time() - t_tile, 2),
        },
        "smooth_detail_iters": int(smooth_detail_iters),
    }
    return mesh, stats, mesh_raw


def _taubin_scalar(
    vals: np.ndarray,
    tris: np.ndarray,
    mask: np.ndarray,
    lam: float = 0.5,
    mu: float = -0.53,
    iters: int = 5,
) -> np.ndarray:
    """Taubin lambda/mu smoothing of a scalar field on a triangle mesh's vertex graph,
    applied to ``mask`` vertices only (the rest are fixed boundary conditions).

    In plane-first mode the detail *is* the displacement field, so smoothing it
    along the face normal is exactly a normal-constrained Taubin pass on the
    detail vertices with every on-plane vertex locked -- no lateral drift, so the
    uniform in-plane lattice survives.
    """
    if len(tris) == 0 or not np.any(mask) or iters <= 0:
        return vals
    n_v = len(vals)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n_v).astype(float)
    deg[deg == 0] = 1.0
    out = vals.astype(np.float64).copy()
    for it in range(int(iters) * 2):
        w = float(lam) if it % 2 == 0 else float(mu)
        acc = np.bincount(e[:, 0], weights=out[e[:, 1]], minlength=n_v)
        lap = acc / deg - out
        out[mask] += w * lap[mask]
    return out


def _n_boundary_edges(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


def _weld_vertices(
    v: np.ndarray, f: np.ndarray, tol: float = 0.0005
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Merge vertices closer than ``tol`` (union-find over the close pairs)."""
    from scipy.spatial import cKDTree

    if len(v) == 0:
        return v, f, 0
    pairs = cKDTree(v).query_pairs(float(tol), output_type="ndarray")
    parent = np.arange(len(v))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_merged = 0
    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            n_merged += 1
    roots = np.array([find(i) for i in range(len(v))])
    uniq, inv = np.unique(roots, return_inverse=True)
    v_new = np.zeros((len(uniq), 3))
    np.add.at(v_new, inv, v)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    v_new /= cnt[:, None]
    f_new = inv[f]
    keep = (f_new[:, 0] != f_new[:, 1]) & (f_new[:, 1] != f_new[:, 2]) & (f_new[:, 0] != f_new[:, 2])
    return v_new, f_new[keep], n_merged, inv


def seam_check(
    mesh: o3d.geometry.TriangleMesh,
    patches: list[Patch],
    z0: float = -0.30,
    half_window_m: float = 0.06,
    bin_m: float = 0.002,
) -> dict:
    """Displacement profile across a horizontal seam line on the largest vertical face.

    Measured as the mesh's own signed offset from that face's plane, binned in
    ``bin_m`` bands of z around ``z0``. A drawer seam shows up as a dip: the
    report gives its depth and the width over which the profile stays below half
    that depth. ``depth >= 3 mm`` over ``<= 15 mm`` == seam captured.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    vert_patches = [p for p in patches if abs(float(p.normal[2])) < 0.35 and p.source == "observed"]
    if not vert_patches or len(verts) == 0:
        return {"available": False}
    p = max(vert_patches, key=lambda q: len(q.inlier_idx))
    s = verts @ p.normal + p.d
    sel = (np.abs(s) < 0.05) & (np.abs(verts[:, 2] - z0) <= half_window_m)
    if sel.sum() < 50:
        return {"available": False, "n_vertices": int(sel.sum())}
    z = verts[sel, 2]
    sv = s[sel]
    edges = np.arange(z0 - half_window_m, z0 + half_window_m + bin_m, bin_m)
    idx = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    prof, centres = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 3:
            continue
        prof.append(float(np.median(sv[m])))
        centres.append(float(0.5 * (edges[b] + edges[b + 1])))
    if len(prof) < 5:
        return {"available": False}
    prof = np.asarray(prof)
    centres = np.asarray(centres)
    base = float(np.median(prof))
    rel = prof - base
    imin = int(np.argmin(rel))
    depth = float(-rel[imin])
    width = float((rel < -0.5 * depth).sum() * bin_m) if depth > 0 else 0.0
    return {
        "available": True,
        "face_label": p.label,
        "z0": float(z0),
        "baseline_mm": round(base * 1000.0, 3),
        "dip_depth_mm": round(depth * 1000.0, 3),
        "dip_width_mm": round(width * 1000.0, 3),
        "dip_z": round(float(centres[imin]), 4),
        "seam_captured": bool(depth >= 0.003 and 0 < width <= 0.015),
        "profile_z": [round(float(c), 4) for c in centres],
        "profile_mm": [round(float(x) * 1000.0, 2) for x in rel],
    }


# --------------------------------------------------------------------------
# F11c: selective detail smoothing (creases locked)
# --------------------------------------------------------------------------

def masked_taubin(
    v: np.ndarray,
    f: np.ndarray,
    mask: np.ndarray,
    lam: float = 0.5,
    mu: float = -0.53,
    iters: int = 5,
    constrain_dir: np.ndarray | None = None,
) -> np.ndarray:
    """Taubin (lambda/mu) smoothing applied to ``mask`` vertices only.

    Unmasked vertices are *fixed* -- they are the patch/plane vertices, so the
    creases and the exactly-flat faces cannot move, and the umbrella update for a
    masked vertex still reads its fixed neighbours, which is what stops the
    detail from being dragged away from the plane it sits on. ``constrain_dir``
    (per-vertex unit vector) restricts the update to that direction, used in
    plane-first mode so the uniform in-plane lattice is preserved and only the
    displacement profile is smoothed.
    """
    if len(f) == 0 or not np.any(mask) or iters <= 0:
        return v
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    n_v = len(v)
    deg = np.bincount(e[:, 0], minlength=n_v).astype(float)
    deg[deg == 0] = 1.0
    out = v.copy()
    for it in range(int(iters) * 2):
        w = float(lam) if it % 2 == 0 else float(mu)
        acc = np.zeros_like(out)
        np.add.at(acc, e[:, 0], out[e[:, 1]])
        lap = acc / deg[:, None] - out
        if constrain_dir is not None:
            lap = (np.einsum("ij,ij->i", lap, constrain_dir))[:, None] * constrain_dir
        out[mask] += w * lap[mask]
    return out


def patch_vertex_ids(
    mesh: o3d.geometry.TriangleMesh,
    patches: list[Patch],
    dist_thresh: float = 0.003,
    edge_band_m: float = 0.012,
) -> np.ndarray:
    """Per-vertex owning patch id (``-1`` = detail), same test as :func:`post_snap_mesh`."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    best_pid = np.full(len(v), -1, dtype=np.int64)
    if len(v) == 0 or not patches:
        return best_pid
    best_dist = np.full(len(v), np.inf)
    tol = 2.0 * float(dist_thresh)
    for pid, p in enumerate(patches):
        lat = _lateral_obb(p.obb, thickness_m=max(float(np.asarray(p.obb.extent)[2]), 4.0 * tol))
        idx = np.asarray(
            lat.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(v)),
            dtype=np.int64,
        )
        if len(idx) == 0:
            continue
        dist = np.abs(v[idx] @ p.normal + p.d)
        ok = dist <= tol
        idx, dist = idx[ok], dist[ok]
        if len(idx) and edge_band_m > 0:
            keep_e = p.edge_dist(v[idx]) >= float(edge_band_m)
            idx, dist = idx[keep_e], dist[keep_e]
        better = dist < best_dist[idx]
        best_pid[idx[better]] = pid
        best_dist[idx[better]] = dist[better]
    return best_pid


def smooth_detail(
    mesh: o3d.geometry.TriangleMesh,
    patches: list[Patch],
    iters: int = 5,
    lam: float = 0.5,
    mu: float = -0.53,
    dist_thresh: float = 0.003,
    edge_band_m: float = 0.012,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Taubin-smooth only the *detail* vertices of a hybrid mesh, then re-snap.

    Detail = every vertex that :func:`patch_vertex_ids` does not assign to a
    patch/hull plane. Patch vertices stay put (creases locked); the post-snap is
    re-asserted afterwards so the flats are still exactly on their planes.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    pid = patch_vertex_ids(mesh, patches, dist_thresh=dist_thresh, edge_band_m=edge_band_m)
    mask = pid < 0
    info = {"n_vertices": int(len(v)), "n_detail_smoothed": int(mask.sum()), "iters": int(iters)}
    if iters <= 0 or not mask.any():
        return mesh, info
    v2 = masked_taubin(v, f, mask, lam=lam, mu=mu, iters=iters)
    info["mean_move_mm"] = round(float(np.linalg.norm(v2[mask] - v[mask], axis=1).mean() * 1000.0), 4)
    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(v2)
    out, _ = post_snap_mesh(out, patches, dist_thresh=dist_thresh, edge_band_m=edge_band_m)
    return out, info


def detail_roughness(
    mesh: o3d.geometry.TriangleMesh,
    patches: list[Patch],
    lo: float = 0.004,
    hi: float = 0.025,
    radius_m: float = 0.005,
    max_q: int = 4000,
    seed: int = 0,
) -> dict:
    """Local roughness of the detail band: rms residual of a 5 mm-radius plane fit.

    Vertices 4-25 mm off the nearest patch plane are the "detail band" (lips,
    rims, seams *and* fusion noise). For each one, fit a plane to the mesh
    vertices within ``radius_m`` and record the rms distance to it. Noise pushes
    this up; a genuine lip or seam is locally planar and does not.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(v) == 0:
        return {"available": False}
    d, has = _dist_to_patch_planes(v, patches)
    band = has & (d >= lo) & (d <= hi)
    qi = np.where(band)[0]
    if len(qi) < 20:
        return {"available": False, "n_band": int(len(qi))}
    rng = np.random.default_rng(seed)
    if len(qi) > max_q:
        qi = rng.choice(qi, size=max_q, replace=False)
    tree = cKDTree(v)
    nbrs = tree.query_ball_point(v[qi], r=float(radius_m))
    res = []
    for i, nb in zip(qi, nbrs):
        if len(nb) < 6:
            continue
        q = v[nb]
        q = q - q.mean(axis=0)
        s = np.linalg.svd(q, compute_uv=False)
        res.append(np.sqrt((s[2] ** 2) / len(q)))
    if not res:
        return {"available": False}
    res = np.asarray(res)
    return {
        "available": True,
        "n_band_vertices": int(band.sum()),
        "n_sampled": int(len(res)),
        "local_plane_rms_mm": round(float(np.sqrt(np.mean(res ** 2)) * 1000.0), 4),
        "median_mm": round(float(np.median(res) * 1000.0), 4),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def hybrid_mesh(
    tsdf_mesh_path,
    target_json,
    planar_solid_path,
    out_dir,
    poisson_depth: int = 10,
    target_tris: int = 200_000,
    dist_thresh: float = 0.003,
    voxel_m: float = 0.002,
    pad_m: float = 0.03,
    spacing_m: float = 0.004,
    exclude_within_m: float = 0.006,
    trim_dist_m: float = 0.012,
    min_detail_cluster: int = 50,
    edge_band_m: float = 0.012,
    target_edge_m: float = 0.003,
    mode: str = "poisson",
    grid_m: float = 0.003,
    noise_floor_m: float = 0.004,
    smooth_detail_iters: int = 5,
    tol_m: float | None = None,
    noise_mm: float | None = None,
) -> dict:
    t0 = time.time()
    tsdf_mesh_path = Path(tsdf_mesh_path)
    target_json = Path(target_json)
    planar_solid_path = Path(planar_solid_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(target_json) as fh:
        tgt = json.load(fh)
    obb = tgt["obb"]

    # --- 0. (N3) noise-adaptive tolerance ----------------------------------
    # Everything here used to run at a fixed 3 mm. On a white glossy desk the
    # fused planes carry ~4 mm of noise, so *no* patch could be found (desk_d:
    # n_patches=1, 122k "detail" points) and the whole object went to Poisson
    # as noise. `tol` = clip(2 x noise, 3, 8) mm makes the patch band, the snap
    # band, the detail threshold and the hull exclusion follow the capture; on a
    # low-noise scene tol == 3 mm and nothing changes.
    from .primitives import tol_from_noise_mm

    if noise_mm is None:
        noise_mm = (tgt.get("cloud") or {}).get("noise_mm")
    if tol_m is None:
        tol_m = tol_from_noise_mm(noise_mm) if noise_mm is not None else float(dist_thresh)
    tol_m = float(tol_m)
    dist_thresh = tol_m
    # 12 deg at 3 mm -> 20 deg at 8 mm: noisy normals scatter, so the detector's
    # normal-variance gate has to open with the noise or it never groups a face.
    patch_normal_deg = float(np.clip(12.0 + (tol_m * 1000.0 - 3.0) * 1.6, 12.0, 20.0))
    exclude_within_m = max(float(exclude_within_m), 1.5 * tol_m)
    spacing_m = max(float(spacing_m), 0.005)

    # --- 1. crop to the object, to an oriented cloud -----------------------
    mesh_in = o3d.io.read_triangle_mesh(str(tsdf_mesh_path))
    mesh_in = _tsdf_mesh_to_world(mesh_in, tgt, obb)
    cropped = crop_to_object(mesh_in, obb, pad_m=pad_m)
    del mesh_in
    if len(cropped.triangles) == 0:
        raise RuntimeError(f"nothing left after cropping {tsdf_mesh_path} to the OBB")
    cropped.compute_vertex_normals()

    # colour source for the final mesh: the *full-resolution* cropped TSDF
    # vertices, which carry the fused RGB, when present. Kept separately from
    # the working cloud so the voxel down-sample cannot blur it. Pipeline-
    # produced ``debug/tsdf_mesh.ply`` frequently has no vertex colour at all,
    # so fall back to the scene's cleaned cloud (``target.ply`` / ``target_clean.ply``).
    colour_source = None
    if cropped.has_vertex_colors():
        colour_source = (
            np.asarray(cropped.vertices, dtype=np.float64).copy(),
            np.asarray(cropped.vertex_colors, dtype=np.float64).copy(),
        )
    else:
        colour_source = _scene_colour_source(target_json)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(cropped.vertices))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(cropped.vertex_normals))
    if cropped.has_vertex_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cropped.vertex_colors))
    n_raw = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_m))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    n_cloud = len(pcd.points)
    t_crop = time.time()

    # --- 2. planar patches --------------------------------------------------
    patches = planar_patches(
        pcd, dist_thresh=dist_thresh, normal_deg=patch_normal_deg,
        min_points=300, min_edge_m=0.02,
    )
    t_patch = time.time()

    # --- 3. snap ------------------------------------------------------------
    snapped, detail_mask = snap_to_patches(pcd, patches, edge_band_m=edge_band_m)

    # drop floating detail specks (TSDF debris) but keep connected features
    n_detail_dropped = 0
    if detail_mask.any() and min_detail_cluster > 0:
        di = np.where(detail_mask)[0]
        sub = snapped.select_by_index(di.tolist())
        labels = np.asarray(sub.cluster_dbscan(eps=3.0 * voxel_m, min_points=5))
        bad = np.ones(len(di), dtype=bool)
        for lab in np.unique(labels[labels >= 0]):
            sel = labels == lab
            if int(sel.sum()) >= int(min_detail_cluster):
                bad[sel] = False
        drop = di[bad]
        n_detail_dropped = int(len(drop))
        if n_detail_dropped:
            keep = np.ones(len(detail_mask), dtype=bool)
            keep[drop] = False
            snapped = snapped.select_by_index(np.where(keep)[0].tolist())
            detail_mask = detail_mask[keep]
            pcd = pcd.select_by_index(np.where(keep)[0].tolist())
            patches = planar_patches(
                pcd, dist_thresh=dist_thresh, normal_deg=patch_normal_deg,
                min_points=300, min_edge_m=0.02,
            )
            snapped, detail_mask = snap_to_patches(pcd, patches, edge_band_m=edge_band_m)

    obs_pts = np.asarray(snapped.points)
    band_before = _detail_band_counts(np.asarray(pcd.points), patches)

    # --- 4. unobserved hull closure ----------------------------------------
    hull = hull_closure_points(
        planar_solid_path, obs_pts, spacing_m=spacing_m, exclude_within_m=exclude_within_m
    )
    # the unobserved faces are planes too -- snap the mesh onto them as well,
    # otherwise Poisson leaves the table underside visibly wavy
    dropped_hull: list = []
    hpatches = hull_patches(hull, min_points=150, dropped_out=dropped_hull)
    if dropped_hull:
        # the sliver facets' closure points go too, otherwise Poisson still
        # reconstructs the spike they describe
        keep_h = np.ones(len(hull.points), dtype=bool)
        keep_h[np.asarray(sorted(set(dropped_hull)), dtype=np.int64)] = False
        hull = hull.select_by_index(np.where(keep_h)[0].tolist())
        hpatches = hull_patches(hull, min_points=150)
    all_patches = patches + hpatches
    t_hull = time.time()

    if str(mode) == "plane_first":
        final, pf, mesh_raw = plane_first_mesh(
            pcd, all_patches, grid_m=grid_m, noise_floor_m=noise_floor_m,
            edge_band_m=edge_band_m, smooth_detail_iters=smooth_detail_iters,
            colour_source=colour_source,
        )
        stats = {
            "scene_out_dir": str(out_dir),
            "mode": "plane_first",
            "n_tsdf_vertices_cropped": int(n_raw),
            "n_cloud": int(n_cloud),
            "n_patches": len(patches),
            "n_hull_patches": len(hpatches),
            "patches": [
                {"id": i, "label": p.label, "source": p.source,
                 "area_m2": round(p.area_m2, 5),
                 "extent_m": [round(float(x), 4) for x in p.extent_m],
                 "normal": [round(float(x), 4) for x in p.normal],
                 "n_inliers": int(len(p.inlier_idx))}
                for i, p in enumerate(all_patches)
            ],
            "n_snapped": int((~detail_mask).sum()),
            "n_detail": int(detail_mask.sum()),
            "n_hull_pts": int(len(hull.points)),
            "edge_band_m": float(edge_band_m),
            "edge_len_after": _edge_length_stats(final),
            "detail_roughness_before_smooth": detail_roughness(mesh_raw, all_patches),
            "detail_roughness_after_smooth": detail_roughness(final, all_patches),
            "seam_check_before_smooth": seam_check(mesh_raw, all_patches),
            "seam_check": seam_check(final, all_patches),
            "coloured": bool(final.has_vertex_colors()),
        }
        stats.update(pf)

        # rms vs the planar solid, over the *observed* part of the surface only
        from scipy.spatial import cKDTree as _KD

        obs_pts_pf = np.asarray(pcd.points)
        if len(final.triangles):
            n_s = int(min(len(obs_pts_pf), 300_000))
            samp = np.asarray(final.sample_points_uniformly(number_of_points=max(n_s, 1000)).points)
            d_obs, _ = _KD(obs_pts_pf).query(samp, k=1)
            obs_sel = d_obs <= float(exclude_within_m)
            stats["observed_frac_of_surface"] = float(obs_sel.mean())
            if planar_solid_path.exists() and obs_sel.any():
                planar = o3d.io.read_triangle_mesh(str(planar_solid_path))
                stats["rms_vs_planar_solid_observed_mm"] = _rms_to_mesh_mm(samp[obs_sel], planar)
            stats["cloud_to_mesh_rms_mm"] = _rms_to_mesh_mm(obs_pts_pf, final)
            v_pf = np.asarray(final.vertices)
            stats["bbox_extents_m"] = (v_pf.max(axis=0) - v_pf.min(axis=0)).tolist()

        ply_path = out_dir / "target_mesh_hybrid2.ply"
        glb_path = out_dir / "target_mesh_hybrid2.glb"
        o3d.io.write_triangle_mesh(str(ply_path), final, write_vertex_normals=True,
                                   write_vertex_colors=True)
        _write_glb(final, glb_path)
        stats["out_ply"], stats["out_glb"] = str(ply_path), str(glb_path)
        try:
            from surfcap.postprocess import render_mesh_png

            render_mesh_png(final, out_dir / "target_mesh_hybrid2_iso.png", max_tris=60_000)
            stats["out_png"] = str(out_dir / "target_mesh_hybrid2_iso.png")
        except Exception as e:  # pragma: no cover - rendering is best-effort
            stats["render_error"] = str(e)
        stats["timings_s"] = dict(stats.get("timings_s", {}))
        stats["timings_s"].update({
            "crop": round(t_crop - t0, 2),
            "patches": round(t_patch - t_crop, 2),
            "hull": round(t_hull - t_patch, 2),
            "plane_first": round(time.time() - t_hull, 2),
        })
        stats["runtime_s"] = round(time.time() - t0, 2)
        with open(out_dir / "hybrid2_mesh_stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)
        return stats

    # --- 5. Poisson over snapped u detail u hull ---------------------------
    fused = o3d.geometry.PointCloud()
    all_pts = np.vstack([obs_pts, np.asarray(hull.points)]) if len(hull.points) else obs_pts
    all_nrm = (
        np.vstack([np.asarray(snapped.normals), np.asarray(hull.normals)])
        if len(hull.points)
        else np.asarray(snapped.normals)
    )
    fused.points = o3d.utility.Vector3dVector(all_pts)
    fused.normals = o3d.utility.Vector3dVector(all_nrm)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        fused, depth=int(poisson_depth), linear_fit=True
    )
    densities = np.asarray(densities)
    if len(densities):
        mesh.remove_vertices_by_mask(densities < float(np.quantile(densities, 0.02)))
    # Poisson closes its shell out to the octree bounds, so it bulges well past
    # the object. Cropping that bulge with the padded OBB *cuts the shell open*
    # (measured: bbox == the padded OBB exactly, watertight False). Trimming by
    # distance to the fused cloud instead keeps the closed surface, because the
    # fused cloud (snapped u detail u hull) already covers the whole intended
    # surface -- that is what the hull-closure step is for.
    if len(mesh.vertices):
        from scipy.spatial import cKDTree

        dv, _ = cKDTree(all_pts).query(np.asarray(mesh.vertices), k=1)
        mesh.remove_vertices_by_mask(dv > float(trim_dist_m))
    mesh = crop_to_object(mesh, obb, pad_m=pad_m)  # safety net; normally a no-op
    n_components = 0
    if len(mesh.triangles):
        tri_ids, n_per, _ = mesh.cluster_connected_triangles()
        tri_ids, n_per = np.asarray(tri_ids), np.asarray(n_per)
        n_components = int(len(n_per))
        mesh.remove_triangles_by_mask(tri_ids != int(np.argmax(n_per)))
        mesh.remove_unreferenced_vertices()
    t_poisson = time.time()

    # --- 6. post-snap + decimate -------------------------------------------
    mesh = _close_holes(mesh)
    edge_stats_before = _edge_length_stats(mesh)
    final, remesh_method = uniform_remesh(mesh, target_edge_m=target_edge_m, taubin_iters=3)
    if len(final.triangles) > target_tris:
        final = decimate(final, target_tris=target_tris)
    # remeshing/smoothing moves vertices off the planes -> snap them back,
    # against the observed patches *and* the planar solid's unobserved faces
    final, per_patch = post_snap_mesh(
        final, all_patches, dist_thresh=dist_thresh, edge_band_m=edge_band_m
    )
    final = _close_holes(final)
    # closing holes adds vertices off the planes -> snap once more (cheap)
    final, per_patch = post_snap_mesh(
        final, all_patches, dist_thresh=dist_thresh, edge_band_m=edge_band_m
    )
    rough_before = detail_roughness(final, all_patches)
    seam_before = seam_check(final, all_patches)
    final, smooth_info = smooth_detail(
        final, all_patches, iters=int(smooth_detail_iters),
        dist_thresh=dist_thresh, edge_band_m=edge_band_m,
    )
    edge_stats_after = _edge_length_stats(final)

    # --- stats --------------------------------------------------------------
    from scipy.spatial import cKDTree

    n_after = min(len(pcd.points), 300_000)
    sample_after = np.asarray(final.sample_points_uniformly(number_of_points=int(n_after)).points)
    obs_tree = cKDTree(obs_pts)
    d_obs, _ = obs_tree.query(sample_after, k=1)
    observed_sel = d_obs <= float(exclude_within_m)
    # the fair before/after comparison is over the *observed* surface only:
    # the whole-mesh count would be dominated by the closure's underside, which
    # sits 18-25 mm off the top plane and has nothing to do with detail.
    band_after = _detail_band_counts(sample_after[observed_sel], patches)
    band_after["n_sampled_total"] = int(len(sample_after))
    band_after["n_observed_region"] = int(observed_sel.sum())

    # rms vs the planar solid, restricted to the *observed* part of the surface
    rms_vs_planar_observed = None
    stats_observed_frac = None
    if planar_solid_path.exists():
        planar = o3d.io.read_triangle_mesh(str(planar_solid_path))
        stats_observed_frac = float(observed_sel.mean()) if len(observed_sel) else 0.0
        rms_vs_planar_observed = _rms_to_mesh_mm(sample_after[observed_sel], planar)

    v = np.asarray(final.vertices)
    f_final = np.asarray(final.triangles)
    n_boundary_edges = 0
    euler = None
    if len(f_final):
        _e = np.sort(
            np.vstack([f_final[:, [0, 1]], f_final[:, [1, 2]], f_final[:, [2, 0]]]), axis=1
        )
        _u, _c = np.unique(_e, axis=0, return_counts=True)
        n_boundary_edges = int((_c == 1).sum())
        euler = int(len(v) - len(_u) + len(f_final))
    # reference: what plain quadric decimation would have given (uniformity A/B)
    edge_stats_quadric = _edge_length_stats(decimate(mesh, target_tris=len(f_final) or 1))
    stats = {
        "scene_out_dir": str(out_dir),
        "mode": "poisson",
        "smooth_detail": smooth_info,
        "detail_roughness_before_smooth": rough_before,
        "detail_roughness_after_smooth": detail_roughness(final, all_patches),
        "seam_check_before_smooth": seam_before,
        "seam_check": seam_check(final, all_patches),
        "n_tsdf_vertices_cropped": int(n_raw),
        "n_cloud": int(n_cloud),
        "noise_mm": (round(float(noise_mm), 3) if noise_mm is not None else None),
        "tol_mm": round(tol_m * 1000.0, 3),
        "patch_normal_deg": round(patch_normal_deg, 2),
        "n_patches": len(patches),
        "patches": [
            {
                "id": i,
                "label": p.label,
                "area_m2": round(p.area_m2, 5),
                "extent_m": [round(float(x), 4) for x in p.extent_m],
                "normal": [round(float(x), 4) for x in p.normal],
                "n_inliers": int(len(p.inlier_idx)),
            }
            for i, p in enumerate(patches)
        ],
        "n_snapped": int((~detail_mask).sum()),
        "n_detail": int(detail_mask.sum()),
        "n_detail_specks_dropped": int(n_detail_dropped),
        "n_hull_pts": int(len(hull.points)),
        "n_hull_patches": len(hpatches),
        "hull_patches": [
            {"label": p.label, "extent_m": [round(float(x), 4) for x in p.extent_m],
             "n_samples": int(len(p.inlier_idx))}
            for p in hpatches
        ],
        "edge_band_m": float(edge_band_m),
        "remesh_method": remesh_method,
        "edge_len_before": edge_stats_before,
        "edge_len_after": edge_stats_after,
        "edge_len_quadric_ref": edge_stats_quadric,
        "poisson_depth": int(poisson_depth),
        "components_before_largest": n_components,
        "components": 1 if len(final.triangles) else 0,
        "tris_out": int(len(final.triangles)),
        "verts_out": int(len(v)),
        "watertight": bool(final.is_watertight()) if len(final.triangles) else False,
        "n_boundary_edges": n_boundary_edges,
        "closed_no_boundary": bool(n_boundary_edges == 0 and len(f_final) > 0),
        "euler_number": euler,
        "bbox_extents_m": (v.max(axis=0) - v.min(axis=0)).tolist() if len(v) else [0, 0, 0],
        "per_patch_post_snap": per_patch,
        "detail_band_before": band_before,
        "detail_band_after": band_after,
        "rms_vs_planar_solid_observed_mm": rms_vs_planar_observed,
        "observed_frac_of_surface": stats_observed_frac,
        "timings_s": {
            "crop": round(t_crop - t0, 2),
            "patches": round(t_patch - t_crop, 2),
            "hull": round(t_hull - t_patch, 2),
            "poisson": round(t_poisson - t_hull, 2),
            "finish": round(time.time() - t_poisson, 2),
        },
    }

    if colour_source is not None:
        final = colorize(final, colour_source, max_dist_m=0.015)
    stats["coloured"] = bool(final.has_vertex_colors())

    ply_path = out_dir / "target_mesh_hybrid.ply"
    glb_path = out_dir / "target_mesh_hybrid.glb"
    o3d.io.write_triangle_mesh(str(ply_path), final, write_vertex_normals=True,
                               write_vertex_colors=True)
    _write_glb(final, glb_path)
    stats["out_ply"] = str(ply_path)
    stats["out_glb"] = str(glb_path)

    try:
        from surfcap.postprocess import render_mesh_png

        render_mesh_png(final, out_dir / "target_mesh_hybrid_iso.png", max_tris=60_000)
        stats["out_png"] = str(out_dir / "target_mesh_hybrid_iso.png")
    except Exception as e:  # pragma: no cover - rendering is best-effort
        stats["render_error"] = str(e)

    stats["runtime_s"] = round(time.time() - t0, 2)
    with open(out_dir / "hybrid_mesh_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="F9: piecewise-planar hybrid mesh (flat where flat, detail where not, closed where unobserved)"
    )
    ap.add_argument("scene_out_dir", help="e.g. out/e1b/table_a (inputs are inferred)")
    ap.add_argument("--tsdf-mesh", default=None)
    ap.add_argument("--target-json", default=None)
    ap.add_argument("--planar-solid", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--target-tris", type=int, default=200_000)
    ap.add_argument("--edge-band", type=float, default=0.012,
                    help="leave this band along each patch rim unsnapped (fillet preservation)")
    ap.add_argument("--target-edge", type=float, default=0.003,
                    help="target triangle edge length for the uniform remesh")
    ap.add_argument("--mode", choices=["poisson", "plane_first"], default="poisson",
                    help="poisson: mesh then flatten (F9). plane_first: exact planes, "
                         "exact corners, measured displacement (F11)")
    ap.add_argument("--grid", type=float, default=0.003,
                    help="plane_first: uniform in-face triangulation grid")
    ap.add_argument("--noise-floor", type=float, default=0.004,
                    help="plane_first: displacements smaller than this are not applied")
    ap.add_argument("--smooth-detail-iters", type=int, default=5,
                    help="Taubin iterations on the detail vertices only (creases locked)")
    args = ap.parse_args(argv)

    scene = Path(args.scene_out_dir)
    tsdf = Path(args.tsdf_mesh) if args.tsdf_mesh else scene / "debug" / "tsdf_mesh.ply"
    tj = Path(args.target_json) if args.target_json else scene / "target.json"
    solid = Path(args.planar_solid) if args.planar_solid else scene / "target_mesh.ply"
    out_dir = Path(args.out_dir) if args.out_dir else scene

    stats = hybrid_mesh(
        tsdf, tj, solid, out_dir,
        poisson_depth=args.poisson_depth, target_tris=args.target_tris,
        edge_band_m=args.edge_band, target_edge_m=args.target_edge,
        mode=args.mode, grid_m=args.grid, noise_floor_m=args.noise_floor,
        smooth_detail_iters=args.smooth_detail_iters,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
