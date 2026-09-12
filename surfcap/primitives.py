"""Plane segmentation, role labelling, polygon extraction, and gravity OBB.

Input point clouds are assumed already in the world frame: metres, Z-up,
table-top at z ~= 0 (per surfcap conventions, see surfcap/types.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from surfcap.types import Surface

_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    inlier_idx: np.ndarray
    centroid: np.ndarray
    rms_m: float
    seeded_role: str | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal basis (u, v, n) with u x v = n."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(tmp, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def _pca_extent(pts2d: np.ndarray, pct: tuple[float, float] | None = None) -> tuple[float, float]:
    """(long, short) extent of a 2D point set along its principal axes.

    By default this is the raw min/max range. Pass `pct=(lo, hi)` to instead use the
    `lo`..`hi` percentile range along each axis, which is robust to a few residual
    outlier points that survive clustering (see (F3) below).
    """
    if len(pts2d) < 2:
        return 0.0, 0.0
    mean = pts2d.mean(axis=0)
    centered = pts2d - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return 0.0, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    proj = centered @ eigvecs
    if pct is None:
        ext = proj.max(axis=0) - proj.min(axis=0)
    else:
        lo = np.percentile(proj, pct[0], axis=0)
        hi = np.percentile(proj, pct[1], axis=0)
        ext = hi - lo
    return float(ext[0]), float(ext[1])


def _largest_planar_cluster(
    xyz: np.ndarray,
    normal: np.ndarray,
    centroid: np.ndarray,
    eps: float = 0.02,
    min_points: int = 20,
) -> tuple[np.ndarray, int]:
    """(F3) Keep only the largest connected cluster of a plane's inliers.

    RANSAC plane fitting accepts every point within `dist` of the fitted plane,
    which can include thin stray streaks (e.g. a shadow or a sliver of an
    adjacent coplanar surface) that touch the real face only at a sparse corner
    bridge. Project inliers onto the plane basis (removing perpendicular noise)
    and DBSCAN-cluster them; keep only the largest cluster's local indices.

    Returns (bool mask into `xyz` for kept points, n_dropped).
    """
    n = len(xyz)
    if n < min_points:
        return np.ones(n, dtype=bool), 0
    u, v, nrm = _make_basis(normal)
    signed_dist = (xyz - centroid) @ nrm
    proj = xyz - signed_dist[:, None] * nrm
    proj_pcd = o3d.geometry.PointCloud()
    proj_pcd.points = o3d.utility.Vector3dVector(proj)
    labels = np.array(proj_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid = labels >= 0
    if not np.any(valid):
        return np.ones(n, dtype=bool), 0
    counts = np.bincount(labels[valid])
    best_label = int(np.argmax(counts))
    keep = labels == best_label
    n_dropped = int(n - keep.sum())
    return keep, n_dropped


def _plane_extent(xyz: np.ndarray, normal: np.ndarray, centroid: np.ndarray) -> tuple[float, float]:
    u, v, _ = _make_basis(normal)
    pts2d = np.stack([(xyz - centroid) @ u, (xyz - centroid) @ v], axis=1)
    return _pca_extent(pts2d)


# --------------------------------------------------------------------------
# (d) primitives.py
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# (N3) noise-adaptive tolerance
# --------------------------------------------------------------------------

def noise_mm_from_mount(
    pts: np.ndarray,
    card_len_m: float = 0.0856,
    band_m: float = 0.015,
    radius_mult: float = 3.0,
) -> float | None:
    """(N3) Robust plane rms, in mm, of the mount surface around the card.

    The cloud is in the world frame (card plane z = 0, card centre at the
    origin), so the points with |z| < ``band_m`` inside ``radius_mult`` card
    lengths of the origin are the mount surface and nothing else. Their
    residual to their own least-squares plane, measured with the MAD (x1.4826)
    instead of the rms, is the fusion noise of *this* capture: ~1.5 mm on a
    textured table, ~4 mm on a white glossy desk. Every tolerance downstream is
    derived from it, so one bad outlier must not be able to inflate it -- hence
    MAD rather than rms.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 200:
        return None
    r = float(radius_mult) * float(card_len_m)
    sel = pts[(np.abs(pts[:, 2]) < float(band_m))
              & (np.linalg.norm(pts[:, :2], axis=1) < r)]
    if len(sel) < 200:
        # the card may sit off to one side of a big surface: fall back to the
        # whole |z| < band_m sheet rather than giving up
        sel = pts[np.abs(pts[:, 2]) < float(band_m)]
        if len(sel) < 200:
            return None
    c = sel.mean(axis=0)
    _u, _s, vt = np.linalg.svd(sel - c, full_matrices=False)
    n = vt[-1]
    res = (sel - c) @ n
    mad = float(np.median(np.abs(res - np.median(res))))
    return float(1.4826 * mad * 1000.0)


TOL_MIN_MM, TOL_MAX_MM = 3.0, 8.0


def tol_from_noise_mm(noise_mm: float | None) -> float:
    """(N3) Plane tolerance in **metres** for a capture with this much noise.

    ``clip(2 x noise, 3, 8) mm``: 2 sigma keeps ~95 % of a plane's own points
    inside the band, the 3 mm floor is the historical fixed tolerance (so
    low-noise scenes behave exactly as before) and the 8 mm cap stops a failed
    capture from swallowing real 1 cm relief.
    """
    if noise_mm is None or not np.isfinite(noise_mm):
        return TOL_MIN_MM / 1000.0
    return float(np.clip(2.0 * float(noise_mm), TOL_MIN_MM, TOL_MAX_MM)) / 1000.0


def _seed_top_surface(
    pcd: o3d.geometry.PointCloud,
    pts: np.ndarray,
    cam_centres: np.ndarray | None,
    band_m: float | None = None,
    eps_m: float | None = None,
    min_points: int = 20,
    min_seed_points: int = 200,
    tol_m: float = 0.003,
) -> tuple["PlaneFit | None", np.ndarray]:
    """(F7) Seed the mount/top surface directly, before iterative RANSAC.

    The cloud is always in the world frame here: the card's plane is z = 0 with
    the card centre at the origin (see frame.py). Cluster the |z| < band_m
    band, keep the cluster around the origin (fallback: largest), fit its
    plane by SVD, and hand it back pre-labelled `top`. Without this, iterative
    RANSAC on a fabric partition can burn its whole `max_planes` budget on
    vertical panel fragments and never find a top at all (partition_e).

    Returns (seed_plane | None, remaining_indices_into_pts) -- the remaining
    indices are everything else in `pts` when a seed was found, or all of
    `pts` unchanged when it wasn't.
    """
    # (N3) the band and the DBSCAN radius follow the measured noise: at 4 mm
    # noise a 6 mm band holds barely 1.5 sigma of the top's own points, so the
    # top breaks into islands and the seed keeps only one of them (desk_d: a
    # 0.36 x 0.30 m "top" on a 0.6 x 0.9 m desk). Both floors are the old fixed
    # values, so a low-noise capture is unchanged.
    if band_m is None:
        band_m = max(0.006, 1.5 * float(tol_m))
    if eps_m is None:
        eps_m = max(0.015, 5.0 * float(tol_m))
    n = len(pts)
    all_idx = np.arange(n)
    band_idx = np.flatnonzero(np.abs(pts[:, 2]) < band_m)
    if len(band_idx) < min_seed_points:
        return None, all_idx

    sub = pcd.select_by_index(band_idx.tolist())
    labels = np.array(sub.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False))
    valid = labels >= 0
    if not np.any(valid):
        return None, all_idx

    xy = pts[band_idx][:, :2]
    origin_near = np.linalg.norm(xy, axis=1) < eps_m * 2.0
    if np.any(valid & origin_near):
        vals, counts = np.unique(labels[valid & origin_near], return_counts=True)
    else:
        vals, counts = np.unique(labels[valid], return_counts=True)
    best_label = int(vals[np.argmax(counts)])

    seed_idx = band_idx[labels == best_label]
    if len(seed_idx) < min_seed_points:
        return None, all_idx

    seed_pts = pts[seed_idx]
    centroid = seed_pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(seed_pts - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    if cam_centres is not None and len(cam_centres) > 0:
        cam_mean = np.asarray(cam_centres, dtype=np.float64).mean(axis=0)
        if float(np.dot(normal, cam_mean - centroid)) < 0:
            normal = -normal
    elif normal[2] < 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    rms = float(np.sqrt(np.mean((seed_pts @ normal + d) ** 2)))

    plane = PlaneFit(
        normal=normal, d=d, inlier_idx=seed_idx, centroid=centroid, rms_m=rms, seeded_role="top"
    )
    remaining = all_idx[~np.isin(all_idx, seed_idx)]
    return plane, remaining


def fit_planes(
    pcd: o3d.geometry.PointCloud,
    max_planes: int = 8,
    dist: float = 0.002,
    normal_deg: float = 45.0,
    min_inliers: int = 200,
    min_frac: float = 0.005,
    seed: int = 0,
    cam_centres: np.ndarray | None = None,
) -> list[PlaneFit]:
    """Iterative RANSAC plane segmentation with normal-consistency filtering
    and DBSCAN largest-cluster keep, per plan (d)."""
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(seed)

    pts = np.asarray(pcd.points)
    n_total = len(pts)
    if n_total == 0:
        return []

    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        if cam_centres is not None and len(cam_centres) > 0:
            cam_loc = np.asarray(cam_centres, dtype=np.float64).mean(axis=0)
        else:
            cam_loc = pts.mean(axis=0) + np.array([0.0, 0.0, 1.0])
        pcd.orient_normals_towards_camera_location(camera_location=cam_loc)

    normals_arr = np.asarray(pcd.normals)
    cloud_centroid = pts.mean(axis=0)

    remaining_mask = np.ones(n_total, dtype=bool)
    planes: list[PlaneFit] = []

    def _run_pass(dist_: float, cos_thresh_: float, min_inliers_: int, min_frac_: float) -> None:
        """One iterative RANSAC-fit / normal-filter / DBSCAN-largest-cluster /
        remove / repeat sweep (plan (d)) over whatever remains of the cloud."""
        consecutive_stalls = 0
        stall_budget = max_planes + 2
        while remaining_mask.sum() >= min_inliers_ and len(planes) < max_planes:
            idx_remaining = np.where(remaining_mask)[0]
            if len(idx_remaining) < 3:
                break
            sub_pcd = pcd.select_by_index(idx_remaining)
            try:
                plane_model, inliers_local = sub_pcd.segment_plane(
                    distance_threshold=dist_, ransac_n=3, num_iterations=3000
                )
            except RuntimeError:
                break
            if len(inliers_local) == 0:
                break

            normal = np.asarray(plane_model[:3], dtype=np.float64)
            norm_len = np.linalg.norm(normal)
            if norm_len == 0:
                break
            normal = normal / norm_len
            d_coef = float(plane_model[3]) / norm_len

            # raw distance-based RANSAC inliers for this round; only the
            # normal-consistent subset is retired from the working set below
            # (per plan (d): fit -> normal-consistency filter -> remove,
            # repeat). Retiring the *raw* (pre-filter) set instead
            # permanently destroys points belonging to other, as-yet-
            # undiscovered small planes (e.g. thin side/front walls)
            # whenever a mixed/contaminated RANSAC hypothesis spans several
            # nearby true planes.
            inliers_global = idx_remaining[np.asarray(inliers_local)]

            pt_normals = normals_arr[inliers_global]
            keep = np.abs(pt_normals @ normal) > cos_thresh_
            inliers_global = inliers_global[keep]
            pt_normals = pt_normals[keep]

            # Note: nothing is retired from remaining_mask here yet. A RANSAC
            # hypothesis fit against a sparse, multi-structure remainder can
            # graze tangentially across parts of *several* unrelated small
            # planes at once (e.g. two different side walls plus stray leg
            # points) without ever forming one coherent surface; permanently
            # discarding such a candidate's points -- even after the normal-
            # consistency filter -- would slowly erode genuinely-planar
            # points belonging to still-undiscovered surfaces every time this
            # happens, eventually starving those surfaces below min_inliers.
            # Points are retired only once a candidate clears every check
            # below (min_inliers/min_frac and a dominant DBSCAN cluster),
            # i.e. once it is confirmed to be one real, coherent surface.

            if len(inliers_global) < min_inliers_ or len(inliers_global) < min_frac_ * n_total:
                consecutive_stalls += 1
                if consecutive_stalls > stall_budget:
                    break
                continue

            sub2 = pcd.select_by_index(inliers_global)
            labels = np.array(sub2.cluster_dbscan(eps=0.03, min_points=30, print_progress=False))
            valid = labels >= 0
            if not np.any(valid):
                consecutive_stalls += 1
                if consecutive_stalls > stall_budget:
                    break
                continue
            counts = np.bincount(labels[valid])
            best_label = int(np.argmax(counts))
            cluster_mask = labels == best_label
            final_idx = inliers_global[cluster_mask]

            if len(final_idx) < min_inliers_:
                consecutive_stalls += 1
                if consecutive_stalls > stall_budget:
                    break
                continue

            consecutive_stalls = 0
            remaining_mask[final_idx] = False
            centroid = pts[final_idx].mean(axis=0)

            # orient the stored plane normal to point outward from the overall
            # cloud body (robust convention: correctly signs even occluded/
            # underside planes such as a table's bottom face, unlike a purely
            # camera-based per-point orientation).
            outward_ref = centroid - cloud_centroid
            proj = float(np.dot(normal, outward_ref))
            # For a slab-like cloud (e.g. a table-mask cloud that is essentially
            # just the top face) every plane centroid coincides with the cloud
            # centroid along its own normal, so `proj` is pure noise and the sign
            # is a coin flip.  Fall back to the plan's camera-facing convention.
            if abs(proj) < 0.01 and cam_centres is not None and len(cam_centres) > 0:
                cam_mean = np.asarray(cam_centres, dtype=np.float64).mean(axis=0)
                proj = float(np.dot(normal, cam_mean - centroid))
            if proj < 0:
                normal = -normal
                d_coef = -d_coef

            dists = pts[final_idx] @ normal + d_coef
            rms = float(np.sqrt(np.mean(dists ** 2)))

            planes.append(
                PlaneFit(normal=normal, d=d_coef, inlier_idx=final_idx, centroid=centroid, rms_m=rms)
            )

    # Tier 1: strict pass close to plan (d)'s literal numeric defaults
    # (dist=4mm, ~23 deg, >=400 inliers, >=1.5% of cloud; loosened from 20 to
    # ~23 deg to recover genuine top/bottom edge points whose per-point
    # normals are mildly noisy near the table's perimeter -- see extent
    # accuracy note below). This reliably finds large, cleanly-separated
    # planes (table top/bottom) with accurate normals/extents, because
    # per-point normal estimates away from sharp box corners are
    # trustworthy at this tight tolerance.
    _run_pass(dist_=0.004, cos_thresh_=np.cos(np.radians(23.0)), min_inliers_=400, min_frac_=0.015)

    # Tier 2: relaxed pass using this call's own (dist, normal_deg,
    # min_inliers, min_frac) on whatever remains. Thin structures (side/
    # front walls only a few cm tall) sit close enough to a perpendicular
    # neighbour (table top/bottom) that estimate_normals' KD-tree radius
    # straddles the corner, so most of their per-point normals are >20 deg
    # off even though the points themselves are genuinely planar; a looser
    # normal-consistency tolerance is required to recover these as surfaces
    # at all. Because tier 1 already claimed the large, contamination-prone
    # planes, this pass's own looser filter mostly only ever sees genuinely
    # small/thin plane candidates.
    _run_pass(
        dist_=dist,
        cos_thresh_=np.cos(np.radians(normal_deg)),
        min_inliers_=min_inliers,
        min_frac_=min_frac,
    )

    return planes


def _refit_plane(pts: np.ndarray, ref_normal: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Total-least-squares plane through `pts` (SVD).  Returns (n, d, centroid, rms_m)."""
    centroid = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    if float(np.dot(n, ref_normal)) < 0:   # keep the incoming (already outward) orientation
        n = -n
    d = -float(np.dot(n, centroid))
    rms = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
    return n, d, centroid, rms


def merge_coplanar(
    planes: list[PlaneFit],
    pts: np.ndarray,
    angle_deg: float = 5.0,
    offset_mm: float = 6.0,
    horiz_angle_deg: float = 8.0,
    horiz_offset_mm: float = 10.0,
) -> tuple[list[PlaneFit], int]:
    """Iteratively merge plane pairs that are the same physical surface.

    Two planes merge when their normals agree to within `angle_deg` (sign-insensitive) AND
    each centroid lies within `offset_mm` of the other's plane.  Inliers are concatenated and
    the merged plane is refit by SVD.  Repeats until no pair merges.

    RANSAC on a slightly-bowed real table top routinely splits the one physical face into
    several near-identical planes; without this every downstream consumer sees 3+ `top_*`.

    (F2-4) 5 deg / 6 mm still left the desk with three `top_*` fragments: a 1 m top is
    bowed by more than 6 mm end to end and the RANSAC normals of its pieces differ by
    more than 5 deg.  Pairs where *both* planes are near-horizontal (|n_z| > cos 25 deg,
    i.e. a top / bottom face) therefore get the looser `horiz_angle_deg` /
    `horiz_offset_mm`; walls and fronts keep the tight thresholds.
    """
    planes = list(planes)
    cos_thresh = np.cos(np.radians(angle_deg))
    off = offset_mm / 1000.0
    cos_h = np.cos(np.radians(horiz_angle_deg))
    off_h = horiz_offset_mm / 1000.0
    cos_horiz = np.cos(np.radians(25.0))
    n_merges = 0
    changed = True
    while changed and len(planes) > 1:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                both_h = (abs(float(a.normal[2])) > cos_horiz
                          and abs(float(b.normal[2])) > cos_horiz)
                ct, ov = (cos_h, off_h) if both_h else (cos_thresh, off)
                if abs(float(np.dot(a.normal, b.normal))) < ct:
                    continue
                da = abs(float(np.dot(a.normal, b.centroid) + a.d))
                db = abs(float(np.dot(b.normal, a.centroid) + b.d))
                if max(da, db) > ov:
                    continue
                idx = np.unique(np.concatenate([a.inlier_idx, b.inlier_idx]))
                n, d, c, rms = _refit_plane(pts[idx], a.normal)
                seeded = a.seeded_role or b.seeded_role
                planes[i] = PlaneFit(normal=n, d=d, inlier_idx=idx, centroid=c, rms_m=rms,
                                      seeded_role=seeded)
                planes.pop(j)
                n_merges += 1
                changed = True
                break
            if changed:
                break
    # biggest surface first, so `top_0` is the real top
    planes.sort(key=lambda pl: -len(pl.inlier_idx))
    return planes, n_merges


def merge_parallel_vertical(
    planes: list[PlaneFit],
    pts: np.ndarray,
    angle_deg: float = 10.0,
    offset_mm: float = 15.0,
) -> tuple[list[PlaneFit], int]:
    """(F7) Merge near-parallel *vertical* planes offset from each other by up
    to `offset_mm`.

    `merge_coplanar` only merges (near-)coincident planes (tight offset
    tolerance). A real fabric partition panel is not perfectly flat, so
    iterative RANSAC often splits one physical panel into several vertical
    fragments whose normals agree but whose offsets differ by up to ~1 cm --
    more than `merge_coplanar`'s tolerances allow. Restricted to planes within
    20 deg of vertical (|n_z| < cos 70 deg) so it never touches the
    horizontal top/bottom merge behaviour above.
    """
    planes = list(planes)
    cos_thresh = np.cos(np.radians(angle_deg))
    off = offset_mm / 1000.0
    vert_cos = np.cos(np.radians(70.0))
    n_merges = 0
    changed = True
    while changed and len(planes) > 1:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                if abs(float(a.normal[2])) >= vert_cos or abs(float(b.normal[2])) >= vert_cos:
                    continue
                if abs(float(np.dot(a.normal, b.normal))) < cos_thresh:
                    continue
                da = abs(float(np.dot(a.normal, b.centroid) + a.d))
                db = abs(float(np.dot(b.normal, a.centroid) + b.d))
                if max(da, db) > off:
                    continue
                idx = np.unique(np.concatenate([a.inlier_idx, b.inlier_idx]))
                n, d, c, rms = _refit_plane(pts[idx], a.normal)
                seeded = a.seeded_role or b.seeded_role
                planes[i] = PlaneFit(normal=n, d=d, inlier_idx=idx, centroid=c, rms_m=rms,
                                      seeded_role=seeded)
                planes.pop(j)
                n_merges += 1
                changed = True
                break
            if changed:
                break
    planes.sort(key=lambda pl: -len(pl.inlier_idx))
    return planes, n_merges


def label_roles(planes: list[PlaneFit], cam_centres: np.ndarray, pts: np.ndarray) -> list[str]:
    """Role labels per plan (d): top / bottom / front / side / other."""
    if len(planes) == 0:
        return []

    cam_centres = np.asarray(cam_centres, dtype=np.float64)
    cam_xy_mean = cam_centres[:, :2].mean(axis=0)

    z = pts[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    top_z_thresh = z_min + 0.75 * (z_max - z_min)
    # A cloud with no legs/underside (< 5 cm of z relief) is a single slab: its
    # z spread is depth noise, not structure, so the "top quartile" gate would
    # reject the one genuine top face.  Any up-facing plane in such a cloud is
    # the top.
    if (z_max - z_min) < 0.05:
        top_z_thresh = z_min - 1.0

    roles: list = [None] * len(planes)
    vertical_idxs = []

    # (F1-C) The mount surface is the plane the card lies on, and the world frame
    # puts the card centre at the origin with that plane at z ~= 0.  So any plane
    # that passes within 8 mm of the origin with a normal within 20 deg of +Z *is*
    # the top, whatever its size or where it sits in the cloud's z range.  This is
    # what rescues a big desk plane that the z-quartile gate mislabels as "other".
    # (F7) A seeded surface is already unconditionally `top` (below); once one
    # exists, it *is* the mount surface, so no other -- unmerged, leftover --
    # near-z=0 plane may also claim "top" (a stray few-hundred-point fragment
    # that RANSAC turns up separately from the seed would otherwise pass the
    # exact same origin/slab checks and produce a spurious second `top`).
    has_seeded_top = any(pl.seeded_role == "top" for pl in planes)

    origin_top = [
        i
        for i, pl in enumerate(planes)
        if abs(float(pl.d)) < 0.008 and abs(float(pl.normal[2])) > np.cos(np.radians(20.0))
    ]
    if origin_top and not has_seeded_top:
        top_z_thresh = z_min - 1.0   # bypass the z-quartile gate entirely

    for i, pl in enumerate(planes):
        if pl.seeded_role:
            roles[i] = pl.seeded_role
            continue
        cosang = float(np.clip(pl.normal[2], -1.0, 1.0))
        a = np.degrees(np.arccos(cosang))
        if not has_seeded_top and i in origin_top:
            roles[i] = "top"
        elif not has_seeded_top and a < 20.0 and pl.centroid[2] >= top_z_thresh:
            roles[i] = "top"
        elif a > 160.0:
            roles[i] = "bottom"
        elif 70.0 <= a <= 110.0:
            vertical_idxs.append(i)
        else:
            roles[i] = "other"

    if vertical_idxs:
        info = []
        for i in vertical_idxs:
            pl = planes[i]
            direction = cam_xy_mean - pl.centroid[:2]
            facing = bool(np.dot(pl.normal[:2], direction) > 0)
            long_ext, _short_ext = _plane_extent(pts[pl.inlier_idx], pl.normal, pl.centroid)
            info.append((i, facing, long_ext))

        facing_candidates = [c for c in info if c[1]]
        pool = facing_candidates if facing_candidates else info
        front_i = max(pool, key=lambda c: c[2])[0]
        for i, _facing, _ext in info:
            roles[i] = "front" if i == front_i else "side"

    return [r if r is not None else "other" for r in roles]


def plane_polygon(
    xyz: np.ndarray, n: np.ndarray, d: float, simplify: float = 0.005
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto the plane basis, take the convex hull, simplify.

    Returns (poly2d [K,2] centred at the projected centroid, basis (3x3 rows
    u, v, n), poly3d [K,3]).
    """
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    u, v, n = _make_basis(n)

    xyz = np.asarray(xyz, dtype=np.float64)
    signed_dist = xyz @ n + d
    proj = xyz - signed_dist[:, None] * n
    centroid = proj.mean(axis=0)

    pts2d_all = np.stack([(proj - centroid) @ u, (proj - centroid) @ v], axis=1)

    basis = np.stack([u, v, n], axis=0)

    if len(pts2d_all) < 3:
        poly2d = pts2d_all.copy()
        poly3d = centroid + poly2d[:, 0:1] * u + poly2d[:, 1:2] * v
        return poly2d, basis, poly3d

    pts2d_f32 = pts2d_all.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts2d_f32)
    approx = cv2.approxPolyDP(hull, epsilon=float(simplify), closed=True)
    poly2d = approx.reshape(-1, 2).astype(np.float64)

    poly3d = centroid[None, :] + poly2d[:, 0:1] * u[None, :] + poly2d[:, 1:2] * v[None, :]
    return poly2d, basis, poly3d


def gravity_obb(pts: np.ndarray) -> dict:
    """Gravity-aligned OBB via minAreaRect on XY + z-range."""
    pts = np.asarray(pts, dtype=np.float64)
    xy = pts[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
    angle = np.radians(angle_deg)
    ax_w = np.array([np.cos(angle), np.sin(angle), 0.0])
    ax_h = np.array([-np.sin(angle), np.cos(angle), 0.0])
    ax_z = np.array([0.0, 0.0, 1.0])

    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
    z_extent = z_max - z_min
    centre = np.array([cx, cy, (z_min + z_max) / 2.0])

    if w >= h:
        axes = [ax_w.tolist(), ax_h.tolist(), ax_z.tolist()]
        extents = [float(w), float(h), float(z_extent)]
    else:
        axes = [ax_h.tolist(), ax_w.tolist(), ax_z.tolist()]
        extents = [float(h), float(w), float(z_extent)]

    return {"centre": centre.tolist(), "axes": axes, "extents_m": extents}


def extract_surfaces(
    pcd: o3d.geometry.PointCloud,
    cam_centres: np.ndarray,
    max_planes: int = 8,
    dist: float = 0.002,
    normal_deg: float = 45.0,
    min_inliers: int = 200,
    min_frac: float = 0.005,
    seed: int = 0,
    simplify: float = 0.005,
    merge_angle_deg: float = 5.0,
    merge_offset_mm: float = 6.0,
    seed_mount: bool = True,
    tol_m: float = 0.003,
) -> tuple[list, dict | None, list]:
    """Run the full primitives pipeline: plane fit -> roles -> polygons/OBB."""
    warnings: list = []
    pts = np.asarray(pcd.points)

    # (F7) Seed the mount/top surface directly instead of leaving it to
    # iterative RANSAC. The cloud is always in the world frame here (card
    # plane at z=0, card centre at the origin), so the |z|<6mm band clustered
    # and matched to the origin *is* the top, whatever it's made of. Removing
    # its points before RANSAC means a fabric partition's vertical panel
    # fragments can no longer eat the whole max_planes budget before a top is
    # ever found (partition_e: 6 side/front fragments, no top).
    seed_plane, remaining_idx = (None, np.arange(len(pts)))
    if seed_mount and len(pts) > 0:
        seed_plane, remaining_idx = _seed_top_surface(
            pcd, pts, cam_centres, tol_m=tol_m
        )

    ransac_pcd = pcd
    if seed_plane is not None and len(remaining_idx) < len(pts):
        ransac_pcd = pcd.select_by_index(remaining_idx.tolist())
        warnings.append(f"seeded_top_points={len(seed_plane.inlier_idx)}")

    # (N3) RANSAC inlier distance follows the noise as well (floor = the old
    # fixed 2 mm, so low-noise scenes fit exactly the same planes).
    dist = max(float(dist), 0.5 * float(tol_m))
    planes = fit_planes(
        ransac_pcd,
        max_planes=max_planes,
        dist=dist,
        normal_deg=normal_deg,
        min_inliers=min_inliers,
        min_frac=min_frac,
        seed=seed,
        cam_centres=cam_centres,
    )

    if not planes and seed_plane is None:
        warnings.append("no_planes_fitted")
        planes = fit_planes(
            ransac_pcd,
            max_planes=max_planes,
            dist=0.008,
            normal_deg=35.0,
            min_inliers=150,
            min_frac=min_frac,
            seed=seed,
            cam_centres=cam_centres,
        )

    if ransac_pcd is not pcd:
        for pl in planes:
            pl.inlier_idx = remaining_idx[pl.inlier_idx]

    if seed_plane is not None:
        planes = [seed_plane] + planes

    if not planes:
        obb = gravity_obb(pts) if len(pts) > 0 else None
        return [], obb, warnings

    n_before = len(planes)
    planes, n_merges = merge_coplanar(planes, pts, angle_deg=merge_angle_deg,
                                      offset_mm=merge_offset_mm)
    if n_merges:
        warnings.append(f"coplanar_merged={n_before}->{len(planes)}")

    n_before_v = len(planes)
    planes, n_vmerges = merge_parallel_vertical(planes, pts)
    if n_vmerges:
        warnings.append(f"parallel_vertical_merged={n_before_v}->{len(planes)}")

    roles = label_roles(planes, cam_centres, pts)

    surfaces = []
    for k, (pl, role) in enumerate(zip(planes, roles)):
        idx = pl.inlier_idx
        xyz = pts[idx]

        # (F3) drop stray points (thin streaks bridged into the RANSAC inlier
        # set at a sparse corner) before computing polygon/extent/n_points/rms.
        keep, n_dropped = _largest_planar_cluster(xyz, pl.normal, pl.centroid)
        if n_dropped:
            idx = idx[keep]
            xyz = xyz[keep]
            warnings.append(f"{role}_{k}: n_points_dropped_as_stray={n_dropped}")

        poly2d, basis, poly3d = plane_polygon(xyz, pl.normal, pl.d, simplify=simplify)
        u, v, _n = basis
        pts2d = np.stack([(xyz - pl.centroid) @ u, (xyz - pl.centroid) @ v], axis=1)
        long_ext, short_ext = _pca_extent(pts2d, pct=(0.5, 99.5))

        rms_dists = xyz @ pl.normal + pl.d
        rms_mm = float(np.sqrt(np.mean(rms_dists ** 2)) * 1000.0)

        surfaces.append(
            Surface(
                id=f"{role}_{k}",
                role=role,
                normal=pl.normal.tolist(),
                centroid=pl.centroid.tolist(),
                extent_m=[float(long_ext), float(short_ext)],
                polygon_2d=poly2d.tolist(),
                polygon_3d=poly3d.tolist(),
                planarity_rms_mm=rms_mm,
                n_points=int(len(idx)),
            )
        )

    obb = gravity_obb(pts)
    return surfaces, obb, warnings


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from tests.synthetic import make_table, make_cameras

    rng = np.random.default_rng(0)
    xyz, _true_planes = make_table(rng=rng)
    cams = make_cameras(12)
    cam_centres = cams[:, :3, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    surfaces, obb, warnings = extract_surfaces(pcd, cam_centres, seed=0)

    print(f"synthetic table: {len(xyz)} points, {len(surfaces)} surfaces, warnings={warnings}")
    print(f"{'id':<10}{'role':<8}{'normal':<28}{'extent_m':<22}{'n_pts':<8}{'rms_mm':<8}")
    for s in surfaces:
        n = np.round(s.normal, 3).tolist()
        e = np.round(s.extent_m, 4).tolist()
        print(f"{s.id:<10}{s.role:<8}{str(n):<28}{str(e):<22}{s.n_points:<8}{s.planarity_rms_mm:<8.3f}")
    print("obb:", obb)
