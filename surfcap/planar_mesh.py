"""F4 planar mesh mode: model the target as a union of observed planes, not a slab.

``postprocess.poisson_mesh`` + ``mirror_close`` assume the object is a single thin
slab: mirror the top face down by the thickness and let screened Poisson wrap a
shell around it.  That is right for a table top and wrong for everything else --
a cabinet corner (top + front + side) grows a phantom mirrored sheet under the
top plus ragged fins, a window sill grows a rounded bulge that wraps the 25 cm
front face, and any hole in the observed top (e.g. the reference card cut out)
becomes a hole in the mesh.

Planar mode instead builds the mesh directly out of the fitted planes:

1. :func:`surface_patch` -- concave hull of each surface's points, in that
   surface's plane basis, triangulated into a flat polygon patch.
2. :func:`snap_adjacent` -- for near-perpendicular surface pairs whose patches
   nearly touch, pull both boundaries onto the planes' intersection line so the
   corner is sharp and closed.
3. :func:`close_solid` -- turn the patches into one closed solid: a slab when
   only a top was observed, an extruded prism down to the lowest observed
   vertical-face point when vertical faces exist.  Prism walls that coincide
   with an observed vertical plane are made exactly coplanar with it.

Everything is metres in the surfcap world frame (Z-up, top near z = 0).
"""
from __future__ import annotations

import numpy as np
import trimesh

from surfcap.postprocess import _as_surface, _plane_basis, _plane_of
from surfcap.types import Surface

# per-role patch colours for target_mesh_planar_patches.glb
ROLE_COLOURS = {
    "top": (0.20, 0.78, 0.33),
    "front": (0.20, 0.42, 0.90),
    "side": (0.95, 0.60, 0.15),
    "bottom": (0.60, 0.30, 0.80),
    "other": (0.62, 0.62, 0.62),
}

DEFAULT_THICKNESS_M = 0.018
_THICKNESS_RANGE_M = (0.01, 0.05)


# --------------------------------------------------------------------------
# 2D helpers
# --------------------------------------------------------------------------

def _largest_cluster(p2: np.ndarray, eps: float = 0.03, min_samples: int = 4) -> np.ndarray:
    """Keep only the biggest DBSCAN cluster of a 2D point set (drops strays)."""
    if len(p2) < 3 * min_samples:
        return np.arange(len(p2))
    try:
        from sklearn.cluster import DBSCAN

        lab = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(p2)
    except Exception:
        return np.arange(len(p2))
    good = lab >= 0
    if not good.any():
        return np.arange(len(p2))
    vals, counts = np.unique(lab[good], return_counts=True)
    keep = vals[int(np.argmax(counts))]
    idx = np.flatnonzero(lab == keep)
    return idx if len(idx) >= 3 else np.arange(len(p2))


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _dedupe_ring(ring: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Drop consecutive duplicates (incl. a repeated closing vertex) and collinear spikes."""
    out = [ring[0]]
    for p in ring[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.asarray(out, dtype=np.float64)


def concave_ring(p2: np.ndarray, alpha_m: float = 0.03) -> np.ndarray:
    """Concave hull ring (CCW, open) of a 2D point set; convex hull as fallback.

    ``alphashape`` is not importable in this env (it needs rtree), so we use
    shapely's ``concave_hull`` with the length ratio that corresponds to
    ``alpha_m``: the ratio is relative to the longest/shortest edge of the
    Delaunay triangulation, so we clamp it into a sane band instead of trying to
    invert it exactly.
    """
    p2 = np.asarray(p2, dtype=np.float64)
    if len(p2) < 3:
        return p2.copy()
    ring = None
    try:
        import shapely
        from shapely.geometry import MultiPoint

        span = float(max(np.ptp(p2[:, 0]), np.ptp(p2[:, 1]), 1e-6))
        ratio = float(np.clip(alpha_m / span * 3.0, 0.25, 0.7))
        hull = shapely.concave_hull(MultiPoint(p2), ratio=ratio, allow_holes=False)
        if hull.geom_type == "Polygon" and hull.area > 1e-8:
            ring = np.asarray(hull.exterior.coords, dtype=np.float64)
    except Exception:
        ring = None
    if ring is None or len(ring) < 4:
        try:
            from scipy.spatial import ConvexHull

            ring = p2[ConvexHull(p2).vertices]
        except Exception:
            return p2.copy()
    ring = _dedupe_ring(ring)
    if len(ring) >= 3 and _signed_area(ring) < 0:
        ring = ring[::-1]
    return ring


def _ear_clip(ring: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation of a simple CCW polygon -> (T,3) index array.

    No triangulation engine (``triangle``/``mapbox_earcut``) is installed in this
    env, so trimesh's ``triangulate_polygon`` is unavailable; ear clipping is
    O(n^2) but the rings here have < 200 vertices.
    """
    n = len(ring)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    idx = list(range(n))
    faces: list[tuple[int, int, int]] = []

    def area2(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))

    def inside(p, a, b, c):
        d1 = area2(a, b, p)
        d2 = area2(b, c, p)
        d3 = area2(c, a, p)
        return (d1 > 0) and (d2 > 0) and (d3 > 0)

    guard = 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0 = idx[(k - 1) % len(idx)]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = ring[i0], ring[i1], ring[i2]
            if area2(a, b, c) <= 1e-14:
                continue  # reflex or degenerate
            if any(
                inside(ring[j], a, b, c)
                for j in idx
                if j not in (i0, i1, i2)
            ):
                continue
            faces.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        faces.append(tuple(idx))
    elif len(idx) > 3:
        # fan fallback (only reached for self-intersecting rings)
        for k in range(1, len(idx) - 1):
            faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(faces, dtype=np.int64)


# --------------------------------------------------------------------------
# 1. per-surface planar patch
# --------------------------------------------------------------------------

def surface_patch(
    surface: Surface, pts_on_plane: np.ndarray, alpha_m: float = 0.03
) -> trimesh.Trimesh:
    """Concave-hull planar patch of one surface, oriented along ``surface.normal``.

    ``pts_on_plane`` are world-frame points belonging to the surface (they do not
    need to be exactly on the plane; they are projected).  The patch carries the
    data later stages need in ``mesh.metadata``: the 2D ring, the plane basis,
    the surface role/id, the patch colour and the points' z range.
    """
    surf = _as_surface(surface)
    n, c = _plane_of(surf)
    u, v = _plane_basis(n)

    pts = np.asarray(pts_on_plane, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        pts = np.asarray(surf.polygon_3d, dtype=np.float64).reshape(-1, 3)
    rel = pts - c
    p2 = np.stack([rel @ u, rel @ v], axis=1)
    keep = _largest_cluster(p2)
    p2 = p2[keep]
    pts = pts[keep] if len(pts) == len(rel) else pts

    ring = concave_ring(p2, alpha_m=alpha_m)
    mesh = _mesh_from_ring(ring, n, c, u, v)
    colour = np.asarray(ROLE_COLOURS.get(surf.role, ROLE_COLOURS["other"]))
    mesh.metadata.update(
        {
            "surface_id": surf.id,
            "role": surf.role,
            "ring2d": ring,
            "origin": c,
            "u": u,
            "v": v,
            "normal": n,
            "colour": colour,
            "z_min": float(pts[:, 2].min()) if len(pts) else float(c[2]),
            "z_max": float(pts[:, 2].max()) if len(pts) else float(c[2]),
            "n_points": int(len(pts)),
        }
    )
    return mesh


def _mesh_from_ring(
    ring: np.ndarray, n: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray
) -> trimesh.Trimesh:
    """Triangulate a 2D ring in the (u,v) basis and lift it to 3D with normal ``n``."""
    if len(ring) < 3:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), int), process=False)
    faces = _ear_clip(ring)
    verts = c[None, :] + ring[:, 0:1] * u[None, :] + ring[:, 1:2] * v[None, :]
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(faces):
        fn = m.face_normals
        if float(np.mean(fn @ n)) < 0:
            m.faces = np.ascontiguousarray(m.faces[:, ::-1])
            m._cache.clear()
    return m


# --------------------------------------------------------------------------
# 2. snap adjacent (near-perpendicular) patches to their intersection line
# --------------------------------------------------------------------------

def _plane_intersection_line(
    n1: np.ndarray, c1: np.ndarray, n2: np.ndarray, c2: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    d = np.cross(n1, n2)
    ln = np.linalg.norm(d)
    if ln < 1e-9:
        return None
    d = d / ln
    A = np.stack([n1, n2, d])
    b = np.array([n1 @ c1, n2 @ c2, 0.0])
    try:
        p0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p0, d


def _line_in_basis(p0, d, c, u, v) -> tuple[np.ndarray, np.ndarray]:
    """(point, unit direction) of a 3D line expressed in a plane's 2D basis."""
    rel = p0 - c
    a2 = np.array([rel @ u, rel @ v])
    d2 = np.array([d @ u, d @ v])
    ln = np.linalg.norm(d2)
    if ln < 1e-12:
        return a2, np.array([1.0, 0.0])
    return a2, d2 / ln


def _project_to_line2d(p2: np.ndarray, a2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    t = (p2 - a2) @ d2
    return a2[None, :] + t[:, None] * d2[None, :]


def snap_adjacent(
    surfaces, patches: list[trimesh.Trimesh], max_gap_m: float = 0.04
) -> list[trimesh.Trimesh]:
    """Sharpen corners: pull near-touching boundaries onto the planes' intersection line.

    A pair qualifies when the planes meet at 60-120 degrees and both patches have
    boundary vertices within ``max_gap_m`` of the intersection line.  Those
    vertices are replaced by their projection onto the line (in each patch's own
    2D basis) and the patch is re-triangulated, so the two patches meet exactly.
    The number of snapped pairs is recorded in ``patches[i].metadata['snapped_with']``.
    """
    surfs = [_as_surface(s) for s in surfaces]
    rings = [np.asarray(p.metadata.get("ring2d", np.zeros((0, 2)))) for p in patches]
    snapped_pairs = 0
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            mi, mj = patches[i].metadata, patches[j].metadata
            ni, ci = mi["normal"], mi["origin"]
            nj, cj = mj["normal"], mj["origin"]
            # abs() folds the 60-120 deg window onto "at least 60 deg apart"
            if abs(float(ni @ nj)) > np.cos(np.radians(60.0)):
                continue
            line = _plane_intersection_line(ni, ci, nj, cj)
            if line is None:
                continue
            p0, d = line
            ai, di = _line_in_basis(p0, d, ci, mi["u"], mi["v"])
            aj, dj = _line_in_basis(p0, d, cj, mj["u"], mj["v"])
            if len(rings[i]) < 3 or len(rings[j]) < 3:
                continue
            proj_i = _project_to_line2d(rings[i], ai, di)
            proj_j = _project_to_line2d(rings[j], aj, dj)
            gi = np.linalg.norm(rings[i] - proj_i, axis=1)
            gj = np.linalg.norm(rings[j] - proj_j, axis=1)
            sel_i = gi <= max_gap_m
            sel_j = gj <= max_gap_m
            if sel_i.sum() < 2 or sel_j.sum() < 2:
                continue
            rings[i] = rings[i].copy()
            rings[j] = rings[j].copy()
            rings[i][sel_i] = proj_i[sel_i]
            rings[j][sel_j] = proj_j[sel_j]
            snapped_pairs += 1
            mi.setdefault("snapped_with", []).append(surfs[j].id)
            mj.setdefault("snapped_with", []).append(surfs[i].id)

    out = []
    for p, ring in zip(patches, rings):
        m = p.metadata
        ring = _dedupe_ring(ring) if len(ring) >= 3 else ring
        if len(ring) >= 3 and _signed_area(ring) < 0:
            ring = ring[::-1]
        nm = _mesh_from_ring(ring, m["normal"], m["origin"], m["u"], m["v"])
        nm.metadata.update(dict(m))
        nm.metadata["ring2d"] = ring
        out.append(nm)
    out_pairs = snapped_pairs
    if out:
        out[0].metadata["n_snapped_pairs"] = out_pairs
    return out


# --------------------------------------------------------------------------
# 3. close the solid
# --------------------------------------------------------------------------

def _thickness(obb, thickness_m):
    if thickness_m is not None:
        return float(thickness_m), "arg"
    try:
        t = float(np.asarray(obb["extents_m"], dtype=float)[2])
        if _THICKNESS_RANGE_M[0] <= t <= _THICKNESS_RANGE_M[1]:
            return t, "obb"
    except Exception:
        pass
    return DEFAULT_THICKNESS_M, "default"


def _prism(top_ring3d: np.ndarray, bottom_ring3d: np.ndarray, ring2d: np.ndarray) -> trimesh.Trimesh:
    """Closed prism between two matching rings (top CCW seen from +Z)."""
    n = len(ring2d)
    faces_top = _ear_clip(ring2d)
    verts = np.vstack([top_ring3d, bottom_ring3d])
    faces = [faces_top]
    faces.append(faces_top[:, ::-1] + n)          # bottom cap, reversed
    walls = []
    for k in range(n):
        a, b = k, (k + 1) % n
        walls.append((a, b, b + n))
        walls.append((a, b + n, a + n))
    faces.append(np.asarray(walls, dtype=np.int64))
    f = np.vstack([x for x in faces if len(x)])
    m = trimesh.Trimesh(vertices=verts, faces=f, process=False)
    return m


def _drop_to_planes(p_xy: np.ndarray, z: float, planes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Minimum-norm XY shift of (p_xy, z) so it satisfies every plane in ``planes``."""
    if not planes:
        return np.array([p_xy[0], p_xy[1], z])
    A = np.array([[n[0], n[1]] for n, _ in planes], dtype=np.float64)
    r = np.array([np.dot(n, np.array([p_xy[0], p_xy[1], z]) - c) for n, c in planes])
    try:
        d, *_ = np.linalg.lstsq(A, -r, rcond=None)
    except np.linalg.LinAlgError:
        d = np.zeros(2)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) > 0.05:
        d = np.zeros(2)
    return np.array([p_xy[0] + d[0], p_xy[1] + d[1], z])


def close_solid(
    surfaces,
    patches: list[trimesh.Trimesh],
    pcd,
    thickness_default: float = DEFAULT_THICKNESS_M,
    thickness_m: float | None = None,
    obb: dict | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Build a closed solid out of the snapped patches.

    * case A -- only a top patch: slab = the top polygon extruded down by the
      thickness (``thickness_m`` > OBB third extent > ``thickness_default``).
    * case B -- top + >= 1 vertical face: the top polygon is extruded down to
      ``z_min``, the lowest observed point on the vertical faces.  Boundary
      vertices that were snapped onto a vertical plane are also pinned to that
      plane at ``z_min``, so the corresponding prism wall is exactly the observed
      plane; the remaining walls are flat.  The bottom is capped flat at z_min.
    * case C -- no top: the patches are returned as-is (open), warning
      ``planar_no_top``.
    """
    info: dict = {"warnings": []}
    tops = [p for p in patches if p.metadata.get("role") == "top"]
    if not tops:
        tops = [
            p for p in patches
            if abs(float(p.metadata["normal"][2])) > 0.85
        ]
    verticals = [
        p for p in patches
        if abs(float(p.metadata["normal"][2])) < 0.5 and len(p.faces) > 0
    ]

    if not tops or len(tops[0].faces) == 0:
        info["case"] = "C"
        info["closed"] = False
        info["warnings"].append("planar_no_top")
        info["z_min"] = float(min((p.metadata["z_min"] for p in patches), default=0.0))
        parts = [p for p in patches if len(p.faces)]
        mesh = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()
        return mesh, info

    top = max(tops, key=lambda p: float(p.area))
    ring2d = np.asarray(top.metadata["ring2d"], dtype=np.float64)
    c, u, v, n = top.metadata["origin"], top.metadata["u"], top.metadata["v"], top.metadata["normal"]
    top3d = c[None, :] + ring2d[:, 0:1] * u[None, :] + ring2d[:, 1:2] * v[None, :]

    # CCW seen from +Z so the top cap's normals point up
    xy = top3d[:, :2]
    if _signed_area(xy) < 0:
        ring2d = ring2d[::-1]
        top3d = top3d[::-1]

    # Snapping can push neighbouring ring vertices onto the same point; merging
    # them afterwards would delete faces and open the solid, so drop them now.
    keep = [0]
    for k in range(1, len(top3d)):
        if np.linalg.norm(top3d[k] - top3d[keep[-1]]) > 1e-4:
            keep.append(k)
    if len(keep) > 3 and np.linalg.norm(top3d[keep[0]] - top3d[keep[-1]]) <= 1e-4:
        keep.pop()
    ring2d = ring2d[keep]
    top3d = top3d[keep]

    if verticals:
        info["case"] = "B"
        z_min = float(min(p.metadata["z_min"] for p in verticals))
        info["thickness_source"] = "vertical_faces"
    else:
        info["case"] = "A"
        th, src = _thickness(obb, thickness_m if thickness_m is not None else None)
        if thickness_m is None and src == "default":
            th = float(thickness_default)
        z_min = float(top3d[:, 2].min() - th)
        info["thickness_m"] = round(float(th), 6)
        info["thickness_source"] = src

    # vertices snapped onto a vertical plane get pinned to it at z_min
    vplanes = [(p.metadata["normal"], p.metadata["origin"]) for p in verticals]
    bottom3d = np.empty_like(top3d)
    n_pinned = 0
    for k, p in enumerate(top3d):
        on = [(nv, cv) for nv, cv in vplanes if abs(np.dot(nv, p - cv)) < 1e-6]
        if len(on) > 2:
            on = on[:2]
        if on:
            n_pinned += 1
        bottom3d[k] = _drop_to_planes(p[:2], z_min, on)

    mesh = _prism(top3d, bottom3d, np.asarray(ring2d, dtype=np.float64))
    # NOTE: no nondegenerate/merge cleanup here -- the prism is watertight by
    # construction and any face removal would punch a hole in it.
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    info["closed"] = bool(mesh.is_watertight)
    info["z_min"] = round(float(z_min), 6)
    info["n_vertical_faces"] = len(verticals)
    info["n_pinned_vertices"] = int(n_pinned)
    if mesh.is_watertight:
        info["volume_m3"] = round(abs(float(mesh.volume)), 9)
    else:
        info["warnings"].append("planar_not_watertight")
    return mesh, info


# --------------------------------------------------------------------------
# colours + rms
# --------------------------------------------------------------------------

def _kdtree(pts):
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(pts, dtype=np.float64))


def _rms_mm(d) -> float:
    d = np.asarray(d, dtype=np.float64)
    if len(d) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(d))) * 1000.0)


def _surface_points(pcd_pts: np.ndarray, surf: Surface, band_m: float = 0.012) -> np.ndarray:
    """World points within ``band_m`` of a surface's plane and inside its polygon."""
    from surfcap.postprocess import _points_in_polygon

    n, c = _plane_of(surf)
    signed = (pcd_pts - c) @ n
    band = np.abs(signed) <= band_m
    poly3d = np.asarray(surf.polygon_3d, dtype=np.float64)
    if len(poly3d) >= 3:
        u, v = _plane_basis(n)
        pc = poly3d - c
        poly2d = np.stack([pc @ u, pc @ v], axis=1)
        rel = pcd_pts - c
        p2 = np.stack([rel @ u, rel @ v], axis=1)
        inside = _points_in_polygon(p2, poly2d, pad=0.01)
        band = band & inside
    return pcd_pts[band]


def _mean_colour(pcd, idx) -> np.ndarray | None:
    try:
        cols = np.asarray(pcd.colors)
        if len(cols) == 0 or len(idx) == 0:
            return None
        return np.clip(cols[idx].mean(axis=0), 0, 1)
    except Exception:
        return None


# --------------------------------------------------------------------------
# 4. driver
# --------------------------------------------------------------------------

def planar_mesh(
    pcd,
    surfaces,
    thickness_m: float | None = None,
    alpha_m: float = 0.03,
    obb: dict | None = None,
    max_gap_m: float = 0.04,
) -> tuple[trimesh.Trimesh, dict]:
    """patches -> snap -> close.  Returns the solid and a stats dict.

    ``pcd`` is the denoised + flattened Open3D cloud, ``surfaces`` the fitted
    surfaces (``top`` first per the surfcap contract).
    """
    import open3d as o3d  # noqa: F401  (pcd type comes from the caller)

    pts = np.asarray(pcd.points, dtype=np.float64)
    surfs = [_as_surface(s) for s in surfaces]

    patches: list[trimesh.Trimesh] = []
    for surf in surfs:
        sel = _surface_points(pts, surf)
        patch = surface_patch(surf, sel, alpha_m=alpha_m)
        if len(patch.faces) == 0:
            continue
        # per-patch mean cloud colour
        if len(sel):
            try:
                _, idx = _kdtree(pts).query(sel, k=1)
                mc = _mean_colour(pcd, idx)
                if mc is not None:
                    patch.metadata["cloud_colour"] = mc
            except Exception:
                pass
        patches.append(patch)

    stats: dict = {"n_patches": len(patches), "alpha_m": alpha_m}
    if not patches:
        stats.update({"case": "C", "closed": False, "n_snapped_pairs": 0,
                      "warnings": ["planar_no_patches"]})
        return trimesh.Trimesh(), stats

    snapped = snap_adjacent(surfs, patches, max_gap_m=max_gap_m)
    n_pairs = int(snapped[0].metadata.get("n_snapped_pairs", 0)) if snapped else 0

    mesh, info = close_solid(
        surfs, snapped, pcd, thickness_m=thickness_m, obb=obb
    )
    stats.update(info)
    stats["n_snapped_pairs"] = n_pairs

    # vertex colours: nearest patch's mean cloud colour (role colour as fallback)
    if len(mesh.vertices):
        _colour_solid(mesh, snapped)

    # agreement with the observed cloud
    if len(mesh.faces) and len(pts):
        samp, _ = trimesh.sample.sample_surface(mesh, 20000)
        d_m2c, _ = _kdtree(pts).query(samp, k=1)
        stats["mesh_to_cloud_rms_mm"] = round(_rms_mm(d_m2c), 4)
        # exact point-to-surface distance (sampled mesh points would floor this
        # at the sample spacing, ~2.5 mm for a 1 m^2 solid at 20k samples)
        sub = pts if len(pts) <= 20000 else pts[
            np.random.default_rng(0).choice(len(pts), 20000, replace=False)
        ]
        try:
            # naive (no rtree in this env) is fine: the solid has ~10^2 faces
            _, d_c2m, _ = trimesh.proximity.closest_point_naive(mesh, sub)
        except Exception:
            d_c2m, _ = _kdtree(samp).query(sub, k=1)
        stats["cloud_to_mesh_rms_mm"] = round(_rms_mm(d_c2m), 4)
        # the same, restricted to points that belong to a fitted surface (the
        # full cloud also carries floor/background points the solid never models)
        on_surf = np.zeros(len(sub), dtype=bool)
        for surf in surfs:
            n, c = _plane_of(surf)
            on_surf |= np.abs((sub - c) @ n) <= 0.012
        if on_surf.any():
            stats["surface_cloud_to_mesh_rms_mm"] = round(_rms_mm(d_c2m[on_surf]), 4)
            stats["n_surface_points"] = int(on_surf.sum())
    else:
        stats["mesh_to_cloud_rms_mm"] = float("nan")
        stats["cloud_to_mesh_rms_mm"] = float("nan")

    stats["mesh_n_vertices"] = int(len(mesh.vertices))
    stats["mesh_n_triangles"] = int(len(mesh.faces))
    stats["patches"] = [
        {
            "surface_id": p.metadata.get("surface_id"),
            "role": p.metadata.get("role"),
            "n_ring": int(len(p.metadata.get("ring2d", []))),
            "area_m2": round(float(p.area), 6),
            "snapped_with": list(p.metadata.get("snapped_with", [])),
        }
        for p in snapped
    ]
    stats["_patches"] = snapped  # for the patches GLB (stripped before JSON)
    return mesh, stats


def _colour_solid(mesh: trimesh.Trimesh, patches: list[trimesh.Trimesh]) -> None:
    """Per-vertex colour: the mean cloud colour of the nearest patch's plane."""
    verts = np.asarray(mesh.vertices)
    cols = np.full((len(verts), 3), 0.6)
    if patches:
        dists = []
        cand = []
        for p in patches:
            n, c = p.metadata["normal"], p.metadata["origin"]
            dists.append(np.abs((verts - c) @ n))
            col = p.metadata.get("cloud_colour")
            cand.append(np.asarray(col if col is not None else p.metadata["colour"]))
        D = np.stack(dists, axis=1)
        best = np.argmin(D, axis=1)
        cols = np.stack([cand[b] for b in best])
    rgba = np.concatenate(
        [np.clip(cols * 255, 0, 255).astype(np.uint8),
         np.full((len(cols), 1), 255, np.uint8)], axis=1
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)


def write_patches_glb(patches: list[trimesh.Trimesh], path) -> None:
    """Each patch as a separate role-coloured geometry (viewer aid)."""
    scene = trimesh.Scene()
    for i, p in enumerate(patches):
        if len(p.faces) == 0:
            continue
        q = p.copy()
        col = np.asarray(p.metadata.get("colour", ROLE_COLOURS["other"]))
        rgba = np.concatenate([np.clip(col * 255, 0, 255).astype(np.uint8), [255]])
        q.visual = trimesh.visual.ColorVisuals(
            mesh=q, vertex_colors=np.tile(rgba, (len(q.vertices), 1))
        )
        scene.add_geometry(q, node_name=f"{p.metadata.get('role','other')}_{i}")
    if len(scene.geometry):
        scene.export(str(path))


def to_open3d(mesh: trimesh.Trimesh):
    """trimesh -> open3d TriangleMesh (so the existing writers/renderers work)."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    if len(mesh.vertices) == 0:
        return m
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    try:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
        if len(vc) == len(mesh.vertices):
            m.vertex_colors = o3d.utility.Vector3dVector(vc)
    except Exception:
        pass
    m.compute_vertex_normals()
    return m
