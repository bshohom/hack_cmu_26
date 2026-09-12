"""Table-top plane -> metric Z-up world frame.  Plan section "(c) frame.py".

World frame convention (``surfcap/types.py``): metres, Z-up (gravity), origin =
credit-card centre projected onto the table-top plane, X = card long-edge
direction projected onto the plane, Y = Z x X.

All transforms are 4x4 sim(3) matrices ``T`` acting as ``p' = s * R (p - o)``,
i.e. ``T[:3,:3] = s R`` and ``T[:3,3] = -s R o``.
"""
from __future__ import annotations

import numpy as np

from .scale import _plane_basis, _plane_from_points

__all__ = [
    "fit_table_plane",
    "build_world_transform",
    "apply_sim3",
    "apply_sim3_to_c2w",
    "build_frame_dict",
]

CARD_LONG_M = 0.08560


# ----------------------------------------------------------------------------- plane


def _ransac_plane_seeded(
    pts: np.ndarray, thresh: float, iters: int = 2000, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    n_pts = pts.shape[0]
    n, d = _plane_from_points(pts)
    best = np.abs(pts @ n + d) <= thresh
    best_cnt = int(best.sum())
    rng = np.random.default_rng(seed)
    if n_pts >= 3:
        idx = rng.integers(0, n_pts, size=(int(iters), 3))
        for tri in idx:
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                continue
            p0, p1, p2 = pts[tri]
            nv = np.cross(p1 - p0, p2 - p0)
            nl = np.linalg.norm(nv)
            if nl < 1e-12:
                continue
            nv = nv / nl
            dv = float(-nv @ p0)
            inl = np.abs(pts @ nv + dv) <= thresh
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt, best = cnt, inl
    if best_cnt >= 3:
        n, d = _plane_from_points(pts[best])
        best = np.abs(pts @ n + d) <= thresh
        if int(best.sum()) >= 3:
            n, d = _plane_from_points(pts[best])
    return n, d, best


def fit_table_plane(
    table_xyz: np.ndarray,
    card_corners_w: np.ndarray | None,
    cam_centres: np.ndarray,
    s: float,
    dist_m: float = 0.004,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, list[str]]:
    """RANSAC the table-top plane in the *unscaled* recon frame.

    Returns ``(n, d, inlier_idx, warnings)`` where ``n`` is a unit normal pointing
    toward the cameras and ``n . p + d = 0``.  ``inlier_idx`` indexes ``table_xyz``.
    The inlier threshold is ``dist_m / s`` so it stays 4 mm in metric terms.
    """
    warnings: list[str] = []
    pts = np.asarray(table_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("fit_table_plane needs >= 3 points")
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    thresh = float(dist_m) / s

    cams = np.asarray(cam_centres, dtype=np.float64).reshape(-1, 3)
    corners = None if card_corners_w is None else np.asarray(card_corners_w, float).reshape(4, 3)

    def _toward_cams(nv, dv):
        """Flip (n, d) so the normal points at the camera cloud."""
        ref = cams.mean(axis=0) if cams.size else pts.mean(axis=0) + nv
        return (-nv, -dv) if float(nv @ ref + dv) < 0 else (nv, dv)

    # the card itself defines the surface: 4 coplanar corners lying ON the mount
    n_card = d_card = None
    if corners is not None:
        n_card, d_card = _plane_from_points(corners)
        n_card, d_card = _toward_cams(n_card, float(d_card))

    # candidate restriction: near the card AND near the card's own plane, so a
    # big perpendicular panel (cubicle fabric, wall) can never out-vote the cap
    idx_all = np.arange(pts.shape[0])
    idx_c = idx_all
    if corners is not None:
        cc = corners.mean(axis=0)
        radius = 1.5 * CARD_LONG_M / s
        band = 0.015 / s  # +- 15 mm of the card plane
        near = (np.linalg.norm(pts - cc, axis=1) <= radius) & (
            np.abs(pts @ n_card + d_card) <= band
        )
        if int(near.sum()) >= 50:
            idx_c = idx_all[near]
        else:
            warnings.append("table_plane_card_neighbourhood_sparse")

    fit_pts = pts[idx_c]
    if corners is not None:
        fit_pts = np.concatenate([fit_pts, corners], axis=0)

    # keep the 2000-iteration RANSAC cheap on very large masks (deterministic stride)
    if fit_pts.shape[0] > 20000:
        fit_pts = fit_pts[:: int(np.ceil(fit_pts.shape[0] / 20000))]
    n, d, inl = _ransac_plane_seeded(fit_pts, thresh, iters=2000, seed=seed)
    n, d = _toward_cams(n, float(d))

    # tightened refit: RANSAC maximises inliers, so it happily tilts a few degrees
    # to swallow the top edge of an adjoining panel.  Halving the band pulls the
    # plane back onto the surface itself -- but only if most inliers survive, so a
    # genuinely noisy cloud keeps the wider fit.
    keep0 = np.abs(fit_pts @ n + d) <= thresh
    if int(keep0.sum()) >= 3:
        keep1 = np.abs(fit_pts @ n + d) <= 0.5 * thresh
        if int(keep1.sum()) >= max(3, 0.5 * int(keep0.sum())):
            n2, d2 = _plane_from_points(fit_pts[keep1])
            n2, d2 = _toward_cams(n2, float(d2))
            if float(n2 @ n) > 0.99:  # < 8 deg: a refinement, never a different plane
                n, d = n2, float(d2)

    # accept the RANSAC plane only if it agrees with the card plane
    if corners is not None:
        cos = float(np.clip(n @ n_card, -1.0, 1.0))
        ang_deg = float(np.rad2deg(np.arccos(cos)))
        centre_mm = abs(float(corners.mean(axis=0) @ n + d)) * s * 1000.0
        if ang_deg < 5.0 and centre_mm < 5.0:
            warnings.append("table_plane_refined_from_ransac")
        else:
            warnings.append("table_plane_from_card_only")
            n, d = n_card, float(d_card)

    inlier_idx = idx_all[np.abs(pts @ n + d) <= thresh]
    if inlier_idx.size < 3:
        inlier_idx = idx_c
    return n, float(d), inlier_idx, warnings


# ----------------------------------------------------------------------------- transform


def build_world_transform(
    n: np.ndarray,
    d: float,
    card_corners_w: np.ndarray | None,
    cam_centres: np.ndarray,
    s: float,
    table_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the 4x4 sim(3) ``world_from_recon`` (``p' = s R (p - o)``).

    ``table_xyz`` should be the *plane inliers*; it is only needed for the
    no-card fallback, where the origin becomes their centroid and X the longest
    axis of their in-plane 2D PCA (warning ``frame_x_from_table_pca``).
    """
    warnings: list[str] = []
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    d = float(d)
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        s = 1.0

    def proj(p):
        p = np.asarray(p, dtype=np.float64)
        return p - (p @ n + d)[..., None] * n

    e1, e2 = _plane_basis(n)

    if card_corners_w is not None:
        c = proj(np.asarray(card_corners_w, dtype=np.float64).reshape(4, 3))
        o = c.mean(axis=0)
        v0 = c[1] - c[0]
        v1 = c[2] - c[3]
        if float(v0 @ v1) < 0:
            v1 = -v1
        x = v0 / max(np.linalg.norm(v0), 1e-12) + v1 / max(np.linalg.norm(v1), 1e-12)
    else:
        warnings.append("frame_x_from_table_pca")
        if table_xyz is not None and np.asarray(table_xyz).size >= 9:
            p = proj(np.asarray(table_xyz, dtype=np.float64).reshape(-1, 3))
            o = p.mean(axis=0)
            rel = p - o
            uv = np.stack([rel @ e1, rel @ e2], axis=1)
            cov = uv.T @ uv / max(uv.shape[0], 1)
            w, V = np.linalg.eigh(cov)
            ax = V[:, int(np.argmax(w))]
            x = ax[0] * e1 + ax[1] * e2
        else:
            warnings.append("frame_origin_arbitrary")
            o = -d * n
            x = e1.copy()

    x = x - (x @ n) * n
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        x = e1.copy()
        nx = 1.0
    x = x / nx
    y = np.cross(n, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, n)
    x = x / np.linalg.norm(x)

    R = np.stack([x, y, n], axis=0)  # rows: world axes expressed in recon frame
    det = float(np.linalg.det(R))
    assert abs(det - 1.0) < 1e-6, f"det(R) = {det}, expected +1"

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = -s * (R @ o)
    return T, warnings


def apply_sim3(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 similarity transform to an [M,3] point array."""
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return p @ T[:3, :3].T + T[:3, 3][None, :]


def apply_sim3_to_c2w(T: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Apply a 4x4 sim(3) to [N,4,4] cam-to-world poses, keeping R orthonormal."""
    T = np.asarray(T, dtype=np.float64)
    poses = np.asarray(c2w, dtype=np.float64)
    single = poses.ndim == 2
    poses = poses.reshape(-1, 4, 4).copy()
    A = T[:3, :3]
    s = float(np.cbrt(max(abs(np.linalg.det(A)), 1e-300)))
    Rw = A / s if s > 0 else A
    U, _, Vt = np.linalg.svd(Rw)
    Rw = U @ Vt
    if np.linalg.det(Rw) < 0:
        U[:, -1] *= -1
        Rw = U @ Vt
    out = poses.copy()
    for i in range(poses.shape[0]):
        Rc = poses[i, :3, :3]
        Rn = Rw @ Rc
        U2, _, Vt2 = np.linalg.svd(Rn)
        Rn = U2 @ Vt2
        if np.linalg.det(Rn) < 0:
            U2[:, -1] *= -1
            Rn = U2 @ Vt2
        out[i, :3, :3] = Rn
        out[i, :3, 3] = A @ poses[i, :3, 3] + T[:3, 3]
        out[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return out[0] if single else out


def build_frame_dict(T: np.ndarray, units: str, origin_desc: str | None = None) -> dict:
    """``Target.frame`` payload."""
    T = np.asarray(T, dtype=np.float64)
    if origin_desc is None:
        origin_desc = "credit-card centre projected onto the table-top plane"
    return {
        "units": units,
        "up": [0.0, 0.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "origin_desc": origin_desc,
        "sim3_world_from_recon": T.tolist(),
    }


# ----------------------------------------------------------------------------- self-check

if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from tests.synthetic import make_cameras, make_card, make_table, random_sim3
    from tests.synthetic import apply_sim3 as ref_apply

    rng = np.random.default_rng(11)
    table_xyz, _ = make_table(rng=rng)
    card_corners, _ = make_card(centre=(0.1, 0.05, 0.0), yaw_deg=10.0, rng=rng)
    cams = make_cameras(n=6, radius=0.8, height=0.6, look_at=(0.1, 0.05, 0.0))
    Tsim = random_sim3(rng)
    s_sim = float(np.linalg.norm(Tsim[:3, 0]))
    s_true = 1.0 / s_sim  # metres per recon unit

    table_r = ref_apply(Tsim, table_xyz)
    corners_r = ref_apply(Tsim, card_corners)
    cam_r = ref_apply(Tsim, cams[:, :3, 3])

    n, d, inl, w1 = fit_table_plane(table_r, corners_r, cam_r, s_true)
    T, w2 = build_world_transform(n, d, corners_r, cam_r, s_true, table_xyz=table_r[inl])

    top = table_xyz[np.abs(table_xyz[:, 2]) < 0.004]
    top_w = apply_sim3(T, ref_apply(Tsim, top))
    nw, _ = _plane_from_points(top_w)
    if nw[2] < 0:
        nw = -nw
    origin_w = apply_sim3(T, ref_apply(Tsim, np.array([[0.1, 0.05, 0.0]])))[0]
    ldir = np.array([np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10)), 0.0])
    ldir_w = (T[:3, :3] / s_true) @ (Tsim[:3, :3] / s_sim) @ ldir

    print("=== surfcap.frame self-check (synthetic, GPU-free) ===")
    print(f"sim3 scale (recon units per m) : {s_sim:.6f}   s = {s_true:.6f} m/unit")
    print(f"plane inliers                  : {inl.size} / {table_r.shape[0]}")
    print(f"table-top z rms (mm)           : {1000 * np.sqrt(np.mean(top_w[:, 2] ** 2)):.4f}")
    print(f"top normal                     : {np.round(nw, 6).tolist()}")
    print(f"angle(top normal, +Z) (deg)    : {np.rad2deg(np.arccos(np.clip(nw[2], -1, 1))):.4f}")
    print(f"origin error (mm)              : {1000 * np.linalg.norm(origin_w):.4f}")
    ang = np.rad2deg(np.arccos(np.clip(abs(ldir_w[0]), -1, 1)))
    print(f"angle(X, card long edge) (deg) : {ang:.4f}")
    print(f"det(R)                         : {np.linalg.det(T[:3, :3] / s_true):.9f}")
    print(f"warnings                       : {w1 + w2}")

    Tn, wn = build_world_transform(n, d, None, cam_r, s_true, table_xyz=table_r[inl])
    print(f"no-card fallback warnings      : {wn}")
    cams_r = cams.copy()
    for i in range(cams.shape[0]):
        cams_r[i, :3, :3] = (Tsim[:3, :3] / s_sim) @ cams[i, :3, :3]
        cams_r[i, :3, 3] = Tsim[:3, :3] @ cams[i, :3, 3] + Tsim[:3, 3]
    c2w_w = apply_sim3_to_c2w(T, cams_r)
    print(f"c2w det(R') min/max            : "
          f"{min(np.linalg.det(c2w_w[i, :3, :3]) for i in range(len(cams_r))):.9f} / "
          f"{max(np.linalg.det(c2w_w[i, :3, :3]) for i in range(len(cams_r))):.9f}")
