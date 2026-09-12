"""Metric scale from a credit-card reference (ISO ID-1, 85.60 x 53.98 mm).

GPU-free: numpy + OpenCV only.  Implements plan section "(b) scale.py".

Pipeline per view
-----------------
1. ``card_corners_2d``  : card mask -> largest external contour -> ``minAreaRect``
   -> ``boxPoints`` -> ``cornerSubPix`` refinement.  Corners come back ordered so
   that ``0->1`` and ``2->3`` are the LONG edges and the loop is counter-clockwise
   (positive shoelace area in pixel coordinates).
2. ``card_corners_3d``  : lift those 4 pixel corners into the recon world frame
   using the view's pointmap.
3. ``estimate_scale``   : six length measurements per view (2 long, 2 short,
   2 diagonals) -> per-view scale, then a weighted-median aggregate.

Units: ``scale`` is *metres per recon unit* (multiply recon coordinates by it).
"""
from __future__ import annotations

import numpy as np

try:  # OpenCV is required; keep the import error readable.
    import cv2
except Exception as _e:  # pragma: no cover
    raise ImportError("surfcap.scale requires OpenCV (cv2)") from _e

from .types import CARD_MM, ScaleResult

__all__ = [
    "card_corners_2d",
    "card_corners_3d",
    "estimate_scale",
]

# ----------------------------------------------------------------------------- helpers


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _order_corners(box: np.ndarray) -> np.ndarray:
    """Roll/flip a 4x2 ``boxPoints`` array so 0->1 and 2->3 are the long edges
    and the polygon winds counter-clockwise (positive shoelace)."""
    box = np.asarray(box, dtype=np.float64).reshape(4, 2)
    d01 = np.linalg.norm(box[1] - box[0])
    d12 = np.linalg.norm(box[2] - box[1])
    if d01 < d12:
        box = np.roll(box, -1, axis=0)
    if _shoelace(box) < 0:
        # reverse winding while keeping (0,1) and (2,3) as the long edges
        box = box[[1, 0, 3, 2]]
    return box.astype(np.float32)


def _plane_from_points(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through ``pts``.  Returns (unit n, d) with n.p + d = 0."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([0.0, 0.0, 1.0]), -float(c[2])
    n = n / nn
    return n, float(-n @ c)


def _ransac_plane(
    pts: np.ndarray,
    thresh: float | None = None,
    iters: int = 500,
    seed: int = 0,
    min_inliers: int = 3,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Small seeded numpy RANSAC plane fit.

    ``thresh`` defaults to a scale-free robust estimate (3 x MAD of the residuals
    of a plain least-squares fit), because at this stage the reconstruction scale
    is exactly the unknown we are trying to measure.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n_pts = pts.shape[0]
    if n_pts < 3:
        raise ValueError("need >= 3 points for a plane")

    n0, d0 = _plane_from_points(pts)
    if thresh is None:
        r = np.abs(pts @ n0 + d0)
        mad = float(np.median(np.abs(r - np.median(r))))
        thresh = max(3.0 * 1.4826 * mad, 1e-9)

    rng = np.random.default_rng(seed)
    best_inl = np.abs(pts @ n0 + d0) <= thresh
    best_cnt = int(best_inl.sum())
    for _ in range(int(iters)):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        nv = np.cross(p1 - p0, p2 - p0)
        nl = np.linalg.norm(nv)
        if nl < 1e-12:
            continue
        nv = nv / nl
        dv = float(-nv @ p0)
        inl = np.abs(pts @ nv + dv) <= thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl

    if best_cnt < max(3, min_inliers):
        return n0, d0, np.abs(pts @ n0 + d0) <= thresh
    n, d = _plane_from_points(pts[best_inl])
    best_inl = np.abs(pts @ n + d) <= thresh
    if int(best_inl.sum()) >= 3:
        n, d = _plane_from_points(pts[best_inl])
    return n, d, best_inl


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _weighted_median(vals, wts) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    order = np.argsort(vals)
    v, w = vals[order], wts[order]
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0:
        return float(np.median(v))
    c = np.cumsum(w) / tot
    i = int(np.searchsorted(c, 0.5))
    return float(v[min(i, v.size - 1)])


# ----------------------------------------------------------------------------- 2D corners

# (F1-D2) relaxed corner-acceptance gates: the original 25 px / 0.85 pair threw
# away 10 of 14 good SAM card masks on the desk scene (small, oblique card).
_SHORT_SIDE_MIN_PX = 12.0
_FILL_MIN = 0.75
# (F5) a card whose mask touches the edge of a text-prompt segmentation cap
# (e.g. the "cubicle partition" mask boundary) gets clipped into a non-rectangle
# by that boundary, so its minAreaRect fill ratio reads low even though the
# corners we can see are still good.  Only relax the gate when the rect's
# aspect is close to the card's own (85.60 / 53.98 ~= 1.586) -- that is the
# signature of "mostly-rectangular card, one side clipped", not a blob.
_FILL_MIN_RELAXED = 0.65
_ASPECT_RELAX_RANGE = (1.4, 1.8)


def _contour_quad(cnt: np.ndarray) -> np.ndarray | None:
    """Approximate a contour by a 4-gon (``approxPolyDP`` with a growing epsilon)."""
    peri = float(cv2.arcLength(cnt, True))
    if peri <= 0:
        return None
    for frac in np.linspace(0.005, 0.10, 20):
        ap = cv2.approxPolyDP(cnt, frac * peri, True)
        if ap.shape[0] == 4:
            return ap.reshape(4, 2).astype(np.float64)
        if ap.shape[0] < 4:
            break
    return None


def _match_quad(box: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Permute ``quad`` (4,2) onto ``box``'s ordering by nearest-corner matching."""
    box = np.asarray(box, dtype=np.float64).reshape(4, 2)
    dist = np.linalg.norm(box[:, None, :] - quad[None, :, :], axis=2)
    order = np.argmin(dist, axis=1)
    if len(set(order.tolist())) != 4:
        return None
    diag = float(np.linalg.norm(box[0] - box[2]))
    if float(np.max(dist[np.arange(4), order])) > 0.35 * diag:
        return None
    return quad[order].astype(np.float32)




def card_corners_2d(mask: np.ndarray, gray: np.ndarray, reasons: list | None = None) -> np.ndarray | None:
    """Card mask (+ grayscale image) -> (4,2) float32 pixel corners, or ``None``.

    Rejects when the contour fills < 85 % of its ``minAreaRect`` or when the short
    side is under 25 px.  ``cornerSubPix`` ((7,7) window) refines each corner; a
    refined corner is accepted only if it moved less than 4 px, otherwise the raw
    ``boxPoints`` corner is kept.
    """
    def _rej(tag):
        if reasons is not None:
            reasons.append(tag)
        return None

    if mask is None:
        return _rej("no_mask")
    m = np.asarray(mask)
    if m.ndim != 2 or m.size == 0 or not m.any():
        return _rej("empty_mask")
    m8 = (m > 0).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _rej("no_contour")
    cnt = max(cnts, key=cv2.contourArea)
    if cnt.shape[0] < 4:
        return _rej("contour_lt_4pts")

    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    rect_area = float(rw) * float(rh)
    if rect_area <= 0:
        return _rej("degenerate_rect")
    fill = float(cv2.contourArea(cnt)) / rect_area
    if fill < _FILL_MIN:
        short, long_ = min(float(rw), float(rh)), max(float(rw), float(rh))
        aspect = long_ / short if short > 1e-9 else float("inf")
        aspect_ok = _ASPECT_RELAX_RANGE[0] <= aspect <= _ASPECT_RELAX_RANGE[1]
        if aspect_ok and fill >= _FILL_MIN_RELAXED:
            if reasons is not None:
                reasons.append(
                    f"fill={fill:.2f}<{_FILL_MIN} relaxed_to={_FILL_MIN_RELAXED} "
                    f"(aspect={aspect:.2f} in {_ASPECT_RELAX_RANGE}, likely mask-edge clip)"
                )
        else:
            return _rej(
                f"fill={fill:.2f}<{_FILL_MIN} (aspect={aspect:.2f}, not in "
                f"{_ASPECT_RELAX_RANGE} -> not relaxed)"
            )
    if min(float(rw), float(rh)) < _SHORT_SIDE_MIN_PX:
        return _rej(f"short_side={min(rw, rh):.0f}px<{_SHORT_SIDE_MIN_PX:.0f}")

    box = _order_corners(cv2.boxPoints(rect))

    # `minAreaRect` is the *bounding* rectangle: under perspective the card images
    # as a general quadrilateral and the bounding rect is systematically too large
    # (2-4 % on a close-up).  Snap each rect corner onto the contour's own 4-gon.
    quad = _contour_quad(cnt)
    if quad is not None:
        matched = _match_quad(box, quad)
        if matched is not None:
            box = matched

    if gray is not None:
        g = np.asarray(gray)
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        try:
            ref = box.reshape(-1, 1, 2).astype(np.float32).copy()
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
            cv2.cornerSubPix(g, ref, (7, 7), (-1, -1), crit)
            ref = ref.reshape(4, 2)
            shift = np.linalg.norm(ref - box, axis=1)
            # a corner that moved > 4 px means subpix latched onto a different
            # feature -> keep the unrefined corner rather than reject the view
            ok = np.isfinite(ref).all(axis=1) & (shift < 4.0)
            box[ok] = ref[ok]
            if reasons is not None and (~ok).any():
                reasons.append(f"subpix_fallback={int((~ok).sum())}")
        except cv2.error:
            pass

    return box.astype(np.float32)


# ----------------------------------------------------------------------------- 3D corners


def card_corners_3d(
    corners2d: np.ndarray,
    pointmap_i: np.ndarray,
    valid_i: np.ndarray,
    inset_px: int = 6,
    win: int = 5,
    cam_centre: np.ndarray | None = None,
    min_plane_inliers: int = 200,
    seed: int = 0,
) -> np.ndarray | None:
    """Lift the 4 pixel corners into the recon world frame -> (4,3), or ``None``.

    The card plane is RANSAC-fitted on the conf-valid 3D points inside the card
    rectangle eroded by 5 px (the rectangle *is* the eroded card mask here, which
    keeps this function self-contained), requiring >= ``min_plane_inliers``.

    Per corner a ``win`` x ``win`` window centred ``inset_px`` toward the rect
    centre gives a conf-valid median 3D point; the corner is rejected if the
    window's depth spread exceeds 3 % of its depth (depth = distance to
    ``cam_centre`` when supplied).

    The returned corner positions themselves come from the pixel->card-plane
    homography fitted on the same eroded-region correspondences (see module note
    in the task report): evaluating it at the *true* corner pixels removes the
    systematic shrink that using the inset window centres directly would bake in.
    """
    if corners2d is None:
        return None
    c2d = np.asarray(corners2d, dtype=np.float64).reshape(4, 2)
    pm = np.asarray(pointmap_i, dtype=np.float64)
    vd = np.asarray(valid_i).astype(bool)
    if pm.ndim != 3 or pm.shape[2] != 3:
        return None
    H, W = pm.shape[:2]
    if vd.shape != (H, W):
        return None

    # --- eroded card region -------------------------------------------------
    poly = np.round(c2d).astype(np.int32)
    filled = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(filled, poly, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # ~5 px erosion
    eroded = cv2.erode(filled, k) > 0
    sel = eroded & vd & np.isfinite(pm).all(axis=2)
    n_sel = int(sel.sum())
    if n_sel < min_plane_inliers:
        return None

    ys, xs = np.nonzero(sel)
    pts = pm[ys, xs]
    if n_sel > 6000:  # subsample for speed, deterministic
        step = int(np.ceil(n_sel / 6000))
        ys, xs, pts = ys[::step], xs[::step], pts[::step]

    n, d, inl = _ransac_plane(pts, iters=300, seed=seed, min_inliers=min_plane_inliers)
    if int(inl.sum()) < min(min_plane_inliers, pts.shape[0]):
        if int(inl.sum()) < 3:
            return None

    # --- per-corner conf-valid window median + depth-spread gate ------------
    centre2d = c2d.mean(axis=0)
    half = int(win) // 2
    med_pts = []
    for i in range(4):
        v = centre2d - c2d[i]
        nv = np.linalg.norm(v)
        p = c2d[i] + (v / nv) * float(inset_px) if nv > 1e-9 else c2d[i]
        cx, cy = int(round(p[0])), int(round(p[1]))
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        sub_v = vd[y0:y1, x0:x1]
        sub_p = pm[y0:y1, x0:x1][sub_v]
        sub_p = sub_p[np.isfinite(sub_p).all(axis=1)]
        if sub_p.shape[0] < 3:
            return None
        med = np.median(sub_p, axis=0)
        ref = np.asarray(cam_centre, float) if cam_centre is not None else None
        if ref is not None:
            dist = np.linalg.norm(sub_p - ref, axis=1)
            depth = float(np.median(dist))
            if depth > 1e-12 and float(np.std(dist)) > 0.03 * depth:
                return None
        else:
            # scale-free fallback: reject only a window straddling a depth jump
            spread = float(np.linalg.norm(np.std(sub_p, axis=0)))
            card_long_units = float(np.linalg.norm(med - np.median(pts, axis=0))) + 1e-12
            if spread > 0.25 * card_long_units:
                return None
        med_pts.append(med)
    med_pts = np.asarray(med_pts)

    # --- pixel -> card-plane homography -------------------------------------
    e1, e2 = _plane_basis(n)
    o = -d * n  # a point on the plane
    rel = pts[inl] - o
    uv = np.stack([rel @ e1, rel @ e2], axis=1)
    pix = np.stack([xs[inl], ys[inl]], axis=1).astype(np.float64)
    if uv.shape[0] < 8:
        Hm = None
    else:
        span = float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0))) + 1e-12
        Hm, _ = cv2.findHomography(
            pix.astype(np.float32), uv.astype(np.float32), cv2.RANSAC, 0.02 * span
        )

    if Hm is None:
        # fall back to the inset window medians projected onto the plane
        out = med_pts - (med_pts @ n + d)[:, None] * n[None, :]
        return out.astype(np.float64)

    hp = np.concatenate([c2d, np.ones((4, 1))], axis=1) @ np.asarray(Hm).T
    w = hp[:, 2]
    if not np.all(np.abs(w) > 1e-12):
        out = med_pts - (med_pts @ n + d)[:, None] * n[None, :]
        return out.astype(np.float64)
    uvc = hp[:, :2] / w[:, None]
    corners3d = o[None, :] + uvc[:, 0:1] * e1[None, :] + uvc[:, 1:2] * e2[None, :]

    # sanity: homography corners must stay near the window medians
    if float(np.max(np.linalg.norm(corners3d - med_pts, axis=1))) > 0.5 * float(
        np.linalg.norm(med_pts[0] - med_pts[2])
    ):
        out = med_pts - (med_pts @ n + d)[:, None] * n[None, :]
        return out.astype(np.float64)

    return corners3d.astype(np.float64)


# ----------------------------------------------------------------------------- aggregate


def estimate_scale(
    masks_card: np.ndarray,
    pointmaps: np.ndarray,
    valids: np.ndarray,
    grays: list[np.ndarray],
    card_mm: tuple[float, float] = CARD_MM,
    cam_centres: np.ndarray | None = None,
    resid_hard_max_mm: float = 12.0,
    resid_soft_mm: float = 3.0,
    scale_outlier_frac: float = 0.10,
    rms_ok_mm: float = 4.0,
    min_views: int = 3,
) -> tuple[ScaleResult, list[str]]:
    """Estimate metres-per-recon-unit from the card seen in one or more views.

    Two-stage gate (F5): the old single hard ``resid_mm > 4`` cutoff threw away
    views whose per-view scales agreed to within a few percent just because the
    absolute mm residual was a bit high (small/far card -> large px noise ->
    large mm residual at a *consistent* scale).  Instead:

    (a) hard-reject only ``resid_mm > resid_hard_max_mm`` (12 mm) -- this still
        catches genuinely wrong corner correspondences;
    (b) weight the survivors by ``area_px / (1 + (resid_mm / resid_soft_mm)^2)``
        (bigger/cleaner card silhouettes and lower-noise views count for more)
        and take the weighted median scale ``s``;
    (c) drop views whose own scale disagrees with ``s`` by more than
        ``scale_outlier_frac`` (10 %) -- this is the real "wrong card" filter;
    (d) ``rms_mm`` is the weighted RMS residual (same weights) over the kept
        views; ``reliable`` requires >= ``min_views`` kept and ``rms_mm <=
        rms_ok_mm``.
    """
    warnings: list[str] = []
    long_m = float(card_mm[0]) / 1000.0
    short_m = float(card_mm[1]) / 1000.0
    diag_m = float(np.hypot(long_m, short_m))
    known = np.array([long_m, long_m, short_m, short_m, diag_m, diag_m])
    pairs = [(0, 1), (2, 3), (1, 2), (3, 0), (0, 2), (1, 3)]

    n_frames = len(masks_card)
    per_view: dict = {}
    cand: list[dict] = []

    for i in range(n_frames):
        mask = np.asarray(masks_card[i])
        if not mask.any():
            continue
        gray = grays[i] if grays is not None and i < len(grays) else None
        reasons: list = []
        c2d = card_corners_2d(mask, gray, reasons=reasons)
        if c2d is None:
            per_view[int(i)] = {"rejected": "corners_2d", "why": reasons}
            continue
        cc = None
        if cam_centres is not None:
            cc = np.asarray(cam_centres)[i][:3]
        c3d = card_corners_3d(
            c2d, pointmaps[i], valids[i], inset_px=6, win=5, cam_centre=cc, seed=i
        )
        if c3d is None:
            per_view[int(i)] = {"rejected": "corners_3d"}
            continue

        meas = np.array([np.linalg.norm(c3d[a] - c3d[b]) for a, b in pairs])
        if not np.all(np.isfinite(meas)) or np.any(meas <= 1e-12):
            per_view[int(i)] = {"rejected": "degenerate"}
            continue
        s_i = float(np.median(known / meas))
        resid_mm = float(np.sqrt(np.mean((meas * s_i * 1000.0 - known * 1000.0) ** 2)))
        area_px = float(np.count_nonzero(mask))
        rec = {
            "view": int(i),
            "scale": s_i,
            "resid_mm": resid_mm,
            "area_px": area_px,
            "corners_w": c3d,
            "meas": meas,
        }
        per_view[int(i)] = {
            "scale": s_i,
            "resid_mm": resid_mm,
            "area_px": area_px,
            "kept": False,
            "why": reasons,
        }
        if resid_mm > resid_hard_max_mm:
            per_view[int(i)]["rejected"] = "resid_mm"
            continue
        cand.append(rec)

    if not cand:
        warnings.append("scale_unreliable_n_views=0")
        return (
            ScaleResult(
                scale=1.0,
                rms_mm=float("inf"),
                n_views=0,
                reliable=False,
                card_corners_w=None,
                per_view=per_view,
            ),
            warnings,
        )

    s_vals = np.array([c["scale"] for c in cand])
    resid_vals = np.array([c["resid_mm"] for c in cand])
    area_vals = np.array([c["area_px"] for c in cand])
    weights = area_vals / (1.0 + (resid_vals / resid_soft_mm) ** 2)
    for c, w_i in zip(cand, weights):
        c["weight"] = float(w_i)
        per_view[c["view"]]["weight"] = float(w_i)

    s = _weighted_median(s_vals, weights)

    keep = np.abs(s_vals / s - 1.0) <= scale_outlier_frac
    if keep.any():
        kept = [c for c, k in zip(cand, keep) if k]
        kept_w = np.array([c["weight"] for c in kept])
        s = _weighted_median(np.array([c["scale"] for c in kept]), kept_w)
    else:  # pragma: no cover - all views disagree with their own median
        kept = cand
        kept_w = weights

    for c in cand:
        per_view[c["view"]]["kept"] = c in kept
        if c not in kept:
            per_view[c["view"]]["rejected"] = "scale_outlier"

    # weighted RMS residual (mm) over the kept views, using the same per-view
    # weights (not per-measurement) so a view with 6 length measurements
    # doesn't get 6x the influence of one with fewer valid corners.
    kept_resid = np.array([c["resid_mm"] for c in kept])
    tot_w = float(kept_w.sum())
    if tot_w > 0:
        rms_mm = float(np.sqrt(np.sum(kept_w * kept_resid**2) / tot_w))
    else:  # pragma: no cover - degenerate all-zero-weight case
        rms_mm = float(np.sqrt(np.mean(kept_resid**2)))
    n_views = len(kept)
    best = min(kept, key=lambda c: c["resid_mm"])

    reliable = bool(n_views >= min_views and rms_mm <= rms_ok_mm)
    if n_views < min_views:
        warnings.append(f"scale_unreliable_n_views={n_views}")
    if rms_mm > rms_ok_mm:
        warnings.append(f"scale_rms_mm={rms_mm:.2f}")

    return (
        ScaleResult(
            scale=float(s),
            rms_mm=rms_mm,
            n_views=n_views,
            reliable=reliable,
            card_corners_w=np.asarray(best["corners_w"], dtype=np.float64),
            per_view=per_view,
        ),
        warnings,
    )


# ----------------------------------------------------------------------------- self-check

if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.test_scale_frame import build_scene  # noqa: E402

    sc = build_scene(seed=7)
    res, warns = estimate_scale(
        sc["masks_card"], sc["pointmaps"], sc["valids"], sc["grays"],
        cam_centres=sc["cam_centres"],
    )
    true_s = sc["scale_true"]
    print("=== surfcap.scale self-check (synthetic, GPU-free) ===")
    print(f"true  scale (m per recon unit) : {true_s:.9f}")
    print(f"est.  scale (m per recon unit) : {res.scale:.9f}")
    print(f"relative error                 : {100.0 * (res.scale / true_s - 1.0):+.4f} %")
    print(f"rms_mm                         : {res.rms_mm:.3f}")
    print(f"n_views / reliable             : {res.n_views} / {res.reliable}")
    print(f"warnings                       : {warns}")
    e = np.linalg.norm(res.card_corners_w - sc["card_corners_recon"], axis=1) * true_s * 1000
    print(f"corner err vs truth (mm)       : {np.round(e, 3).tolist()}")
    for k, v in sorted(res.per_view.items()):
        print(f"  view {k}: {v}")
