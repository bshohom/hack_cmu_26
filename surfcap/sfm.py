"""surfcap.sfm -- hybrid SfM (pycolmap) + monocular depth + TSDF fusion.

Motivation (E1).  The default `da3` recon mode takes both the poses and the depth from one
feed-forward Depth Anything 3 pass and fuses the per-view point maps by concatenation
(`recon.fuse_views`).  That union is as noisy as the worst view: nothing averages the
per-view depth error out, so small features (a 5 mm lip, a drawer seam) drown in a 3-4 mm
thick point band.

This module builds the alternative:

  1. `run_sfm`      -- classical SIFT SfM on the same 1024-px frames (pycolmap, CPU; the
                       installed build has `has_cuda == False`, so *sparse* mapping only,
                       never `patch_match_stereo`).  Gives geometrically-consistent poses,
                       one shared pinhole camera, and a sparse point cloud with per-image
                       2D observations.
  2. `solve_scale_shift` / `align_depth_to_sparse`
                    -- DA3 depth is affine-ambiguous.  For every registered view we solve a
                       robust (Huber IRLS) least-squares scale+shift **on inverse depth**
                       against the sparse points visible in that view, which puts every
                       view's depth into one common SfM unit.
  3. `tsdf_fuse`    -- Open3D ScalableTSDFVolume integrates the aligned depth maps with the
                       SfM poses.  The truncated signed distance field *averages* the views
                       instead of stacking them, which is what recovers the thin features.

Conventions match the rest of surfcap: OpenCV pinhole, `w2c` 4x4 world-to-camera,
`c2w` cam-to-world with det(R) = +1, z-depth (not ray distance).
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SfmResult",
    "run_sfm",
    "solve_scale_shift",
    "solve_grid_correction",
    "undistort_maps",
    "apply_grid_correction",
    "align_depth_to_sparse",
    "depth_edge_mask",
    "geometric_consistency_filter",
    "tsdf_fuse",
    "MIN_REGISTERED",
]

MIN_REGISTERED = 6          # fewer registered views than this -> caller falls back to `da3`
SFM_TIME_BUDGET_S = 90.0    # wall-clock cap for the whole SfM stage
MIN_ANCHORS = 15            # views with fewer anchors keep the global median (a, b)

# ---- E1b/E1c: spatially varying (grid) inverse-depth correction -----------------------
# E1c makes the grid finer (6 x 4 instead of 4 x 3): after E1b the residual was a ~3.5 mm
# *dish* along the image y axis, which a 3-cell-tall bilinear field cannot represent (it is
# quadratic and 3 cells give the curvature a single interior knot).  Anchor-starved views
# step back down the ladder (6x4 -> 4x3 -> affine) instead of overfitting.
GRID_NX = 6                 # cells across the image (nodes = GRID_NX + 1)
GRID_NY = 4                 # cells down the image
GRID_NX_COARSE = 4          # fallback grid when GRID_MIN_ANCHORS <= n < GRID_FINE_MIN
GRID_NY_COARSE = 3
GRID_FINE_MIN_ANCHORS = 60  # below this the coarse 4 x 3 grid is used instead
GRID_MIN_ANCHORS = 40       # below this a view keeps the global affine fit
# (E1c) The 2nd-order prior penalises *curvature*, which is the shape of the very dish the
# finer grid exists to capture, so halving it to 0.15 does let the synthetic fit through
# (0.80 -> 0.36 mm on a 3.4 mm dish).  On the real captures it overfits: table_a top rms
# went 1.80 -> 2.38 mm at 0.15.  Kept at the E1b value; the ~0.8 mm the fine grid leaves on
# the synthetic dish is this prior, deliberately.
GRID_LAMBDA = 0.30          # Tikhonov (2nd-order) smoothness between grid nodes
GRID_REF_CELLS = 12         # the 4 x 3 grid GRID_LAMBDA was tuned on (E1b)
GRID_FIRST_ORDER = 0.25     # weight of the 1st-order term relative to the 2nd-order one
GRID_PRIOR = 0.05           # pull of the global affine solution on every node
EDGE_LOG_RANGE = 0.05       # 3x3 log-depth range above this = depth discontinuity
GEOM_K = 4                  # (N2) neighbouring views consulted per pixel
GEOM_REL_TOL = 0.02         # relative depth disagreement a neighbour may still confirm
GEOM_MIN_AGREE = 2          # neighbours that must confirm a pixel
GEOM_PX_TOL = 2.0           # round-trip reprojection error, pixels
GEOM_MIN_KEPT = 0.15        # (N3) no view may keep less than this; its tol is relaxed
GEOM_MAX_REL_TOL = 0.06     # (N3) cap on that relaxation

# ---- E1c: radial / pinhole split ------------------------------------------------------
# The SfM pass now solves a SIMPLE_RADIAL camera (f, cx, cy, k1) with
# `ba_refine_extra_params`: the phone lens distortion is real, and letting BA absorb it is
# what fixed the metric scale in the E1b ablation (card 85.7 mm, height 0.749 m).  But the
# depth half of the pipeline -- grid alignment, back-projection, TSDF integration, Open3D
# PinholeCameraIntrinsic -- has no distortion model at all, so E1b's radial run fed a
# *distorted* depth grid to a pinhole integrator and top rms got worse (1.75 -> 2.39 mm).
# The split implemented here: keep the radial poses, intrinsics and sparse points, then
# undistort every view's depth / conf / RGB (and the SAM masks, in the pipeline) onto the
# same pinhole K before anything downstream touches them, and push the sparse anchor pixels
# through the same map, so everything after this point sees one consistent pinhole camera.
#
# E1c A/B, both arms on the 6 x 4 grid (top plane rms, mm):
#                       table_a   partition_e   card long (truth 85.60)      registered
#   SIMPLE_PINHOLE       1.80        1.95        85.3 / 85.4                 10/10, 7/8
#   SIMPLE_RADIAL split  1.76        2.41        85.1 / 85.6                 10/10, 8/8
# The split fixes the E1b radial regression on table_a (2.39 -> 1.76: rectifying the depth
# really was the missing half) and registers every partition view, but k1 is 15x larger on
# the partition capture (-0.032 vs -0.002) and rectifying that much distortion with NEAREST
# depth costs more than the pose accuracy buys -- top rms 1.95 -> 2.41.  So pinhole stays
# the default; SURFCAP_SFM_CAMERA=SIMPLE_RADIAL turns the split on.
import os as _os
SFM_CAMERA_MODEL = _os.environ.get("SURFCAP_SFM_CAMERA", "SIMPLE_PINHOLE")


# --------------------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------------------
@dataclass
class SfmResult:
    """Sparse SfM for a subset of `frames`.

    `idx` maps row j of every array here back to the caller's frame list:
    `w2c[j]` / `K[j]` are the pose/intrinsics of `frames[idx[j]]`.
    """

    idx: list[int]                  # registered frame indices, ascending
    w2c: np.ndarray                 # [M,4,4] world-to-camera, float64
    K: np.ndarray                   # [M,3,3] in the frames' own pixel units
    obs_xy: list[np.ndarray]        # per registered view: [n_j,2] keypoint pixel coords
    obs_xyz: list[np.ndarray]       # per registered view: [n_j,3] SfM world points
    points_xyz: np.ndarray          # [P,3] all sparse points (SfM units)
    n_registered: int
    n_input: int
    reproj_px: float
    seconds: float
    camera_model: str = "SIMPLE_PINHOLE"
    k1: float = 0.0                 # SIMPLE_RADIAL distortion coefficient (0 = pinhole)
    warnings: list[str] = field(default_factory=list)

    @property
    def c2w(self) -> np.ndarray:
        return np.stack([np.linalg.inv(w) for w in self.w2c], axis=0)


def _nearest_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


# --------------------------------------------------------------------------------------
# 1. sparse SfM
# --------------------------------------------------------------------------------------
def run_sfm(
    frames: list,
    exif_f_px: float | None = None,
    work_dir: str | Path = "sfm_work",
    max_num_features: int = 4096,
    time_budget_s: float = SFM_TIME_BUDGET_S,
    seed: int = 0,
) -> SfmResult | None:
    """SIFT SfM over `frames` (their in-memory 1024-px RGB), CPU only.

    Returns None when fewer than MIN_REGISTERED frames register -- the caller then warns
    `sfm_failed` and falls back to the `da3` recon mode.
    """
    import cv2

    try:
        import pycolmap
    except Exception as exc:  # pragma: no cover -- pycolmap is installed in this env
        print(f"[sfm] pycolmap unavailable: {exc}")
        return None

    t0 = time.time()
    warnings: list[str] = []
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for i, f in enumerate(frames):
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(img_dir / name), f.rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        names.append(name)
    H, W = frames[0].rgb.shape[:2]

    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = SFM_CAMERA_MODEL
    radial = SFM_CAMERA_MODEL == "SIMPLE_RADIAL"
    if exif_f_px is not None and np.isfinite(exif_f_px) and exif_f_px > 0:
        reader.camera_params = f"{float(exif_f_px)},{W / 2.0},{H / 2.0}" + (",0.0" if radial else "")
    else:
        warnings.append("sfm_no_exif_focal_prior")

    fopts = pycolmap.FeatureExtractionOptions()
    fopts.sift.max_num_features = int(max_num_features)
    fopts.use_gpu = False
    try:
        pycolmap.extract_features(
            database_path=db,
            image_path=img_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,   # one shared camera for all views
            reader_options=reader,
            extraction_options=fopts,
            device=pycolmap.Device.cpu,
        )
        t_feat = time.time() - t0
        pycolmap.match_exhaustive(database_path=db, device=pycolmap.Device.cpu)
        t_match = time.time() - t0

        mopts = pycolmap.IncrementalPipelineOptions()
        mopts.num_threads = -1
        mopts.random_seed = int(seed)
        mopts.min_model_size = 3
        mopts.multiple_models = True
        mopts.ba_refine_principal_point = False
        # prior focal from EXIF: let BA polish it but do not let it run away
        mopts.ba_refine_focal_length = True
        mopts.ba_refine_extra_params = radial
        remaining = max(10.0, time_budget_s - (time.time() - t0))
        mopts.max_runtime_seconds = int(max(10, round(remaining)))
        out = work / "sparse"
        out.mkdir(parents=True, exist_ok=True)
        recs = pycolmap.incremental_mapping(
            database_path=db, image_path=img_dir, output_path=out, options=mopts
        )
    except Exception as exc:  # noqa: BLE001 -- SfM is optional, never crash the pipeline
        print(f"[sfm] failed: {type(exc).__name__}: {exc}")
        return None

    if not recs:
        print("[sfm] no reconstruction produced")
        return None
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(
        f"[sfm] feat {t_feat:.1f}s match {t_match - t_feat:.1f}s map "
        f"{time.time() - t0 - t_match:.1f}s -> {len(recs)} model(s), "
        f"largest {rec.num_reg_images()} images / {rec.num_points3D()} points"
    )

    # ---- harvest poses, K and per-view observations -----------------------------------
    name_to_idx = {n: i for i, n in enumerate(names)}
    k1_vals: list[float] = []
    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for image_id in rec.reg_image_ids():
        im = rec.image(image_id)
        fi = name_to_idx.get(im.name)
        if fi is None:
            continue
        cam = rec.camera(im.camera_id)
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        params = np.asarray(cam.params, dtype=np.float64).ravel()
        # SIMPLE_RADIAL params are (f, cx, cy, k1); SIMPLE_PINHOLE has no k1
        k1_vals.append(float(params[3]) if params.size >= 4 else 0.0)
        rigid = im.cam_from_world
        if callable(rigid):
            rigid = rigid()
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = np.asarray(rigid.matrix(), dtype=np.float64)
        if np.linalg.det(w2c[:3, :3]) <= 0:                  # numerical hygiene
            w2c[:3, :3] = _nearest_rotation(w2c[:3, :3])

        xy, xyz = [], []
        for p2d in im.points2D:
            if not p2d.has_point3D():
                continue
            xy.append(np.asarray(p2d.xy, dtype=np.float64))
            xyz.append(np.asarray(rec.point3D(p2d.point3D_id).xyz, dtype=np.float64))
        rows.append(
            (
                fi,
                w2c,
                K,
                np.asarray(xy, dtype=np.float64).reshape(-1, 2),
                np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
            )
        )

    rows.sort(key=lambda r: r[0])
    if len(rows) < MIN_REGISTERED:
        print(f"[sfm] only {len(rows)} frames registered (< {MIN_REGISTERED})")
        return None

    pts = np.asarray([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    res = SfmResult(
        idx=[r[0] for r in rows],
        w2c=np.stack([r[1] for r in rows], axis=0),
        K=np.stack([r[2] for r in rows], axis=0),
        obs_xy=[r[3] for r in rows],
        obs_xyz=[r[4] for r in rows],
        points_xyz=pts,
        n_registered=len(rows),
        n_input=len(frames),
        reproj_px=float(rec.compute_mean_reprojection_error()),
        seconds=float(time.time() - t0),
        camera_model=str(SFM_CAMERA_MODEL),
        k1=float(np.median(k1_vals)) if k1_vals else 0.0,
        warnings=warnings,
    )
    if res.seconds > time_budget_s:
        res.warnings.append(f"sfm_over_budget={res.seconds:.0f}s")
    return res


# --------------------------------------------------------------------------------------
# 1b. radial -> pinhole rectification   [E1c]
# --------------------------------------------------------------------------------------
def undistort_maps(K: np.ndarray, k1: float, W: int, H: int):
    """cv2 remap tables that take a SIMPLE_RADIAL image to the pinhole camera `K`.

    `P = K` on purpose: the rectified image keeps the *same* intrinsics the SfM solve
    reported, so nothing downstream has to re-derive a focal length -- only the pixel
    grid changes.  Returns (map1, map2) for `cv2.remap`, or None when k1 is negligible.
    """
    import cv2

    if not np.isfinite(k1) or abs(float(k1)) < 1e-9:
        return None
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.array([float(k1), 0.0, 0.0, 0.0], dtype=np.float64)
    return cv2.initUndistortRectifyMap(K, dist, None, K, (int(W), int(H)), cv2.CV_32FC1)


def undistort_image(img: np.ndarray, maps, nearest: bool = False) -> np.ndarray:
    """Apply `undistort_maps` output to one image (depth/conf/RGB/mask)."""
    import cv2

    if maps is None:
        return img
    src = np.asarray(img)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if src.dtype == bool:
        out = cv2.remap(src.astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out.astype(bool)
    out = cv2.remap(np.ascontiguousarray(src), maps[0], maps[1], interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.astype(src.dtype, copy=False)


def undistort_points(xy: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    """Map distorted keypoint pixels to their position in the rectified (pinhole K) image."""
    import cv2

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.size == 0 or not np.isfinite(k1) or abs(float(k1)) < 1e-9:
        return xy
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.array([float(k1), 0.0, 0.0, 0.0], dtype=np.float64)
    out = cv2.undistortPoints(xy.reshape(-1, 1, 2), K, dist, P=K)
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


# --------------------------------------------------------------------------------------
# 2. robust inverse-depth scale + shift
# --------------------------------------------------------------------------------------
def solve_scale_shift(
    d_pred: np.ndarray,
    d_ref: np.ndarray,
    iters: int = 3,
    huber_k: float = 1.345,
    trim_sigma: float = 3.0,
) -> tuple[float, float, float]:
    """Robust least squares for (a, b) in   1/d_ref ~= a * (1/d_pred) + b.

    `d_pred` is the monocular (DA3) z-depth at each anchor pixel, `d_ref` the z-depth of the
    corresponding triangulated SfM point in that camera.  Working on *inverse* depth is what
    makes this a linear problem and matches the affine ambiguity of a scale-invariant
    monocular depth head.

    Robustness is Huber IRLS: `iters` reweighting passes around an initial plain LS fit,
    with the Huber threshold set from the MAD of the residuals each pass, plus a hard
    trim at `trim_sigma` * MAD-sigma.  The trim matters: SfM depth outliers are one-sided
    (a mis-triangulated point sits *behind* the surface, never in front), and plain Huber
    -- which only downweights, never rejects -- still carries a ~1 % bias from them,
    while the trimmed version lands within 0.1 %.

    Returns (a, b, resid) where `resid` is the weighted RMS inverse-depth residual.
    """
    x = 1.0 / np.asarray(d_pred, dtype=np.float64).ravel()
    y = 1.0 / np.asarray(d_ref, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return 1.0, 0.0, float("inf")

    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(max(1, iters) + 1):
        sw = np.sqrt(w)[:, None]
        try:
            sol, *_ = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        a, b = float(sol[0]), float(sol[1])
        r = A @ sol - y
        mad = float(np.median(np.abs(r - np.median(r))))
        s = 1.4826 * mad if mad > 0 else (float(np.std(r)) or 1e-9)
        u = np.abs(r) / (huber_k * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        if trim_sigma and trim_sigma > 0:
            w = np.where(np.abs(r) > trim_sigma * s, 0.0, w)

    r = A @ np.array([a, b]) - y
    resid = float(np.sqrt(np.average(r**2, weights=w))) if w.sum() > 0 else float("inf")
    return a, b, resid


def align_depth_to_sparse(
    depth: np.ndarray,
    valid: np.ndarray,
    sfm: SfmResult,
    j: int,
    min_anchors: int = MIN_ANCHORS,
    grid: bool = True,
    grid_min_anchors: int = GRID_MIN_ANCHORS,
    xy: np.ndarray | None = None,
) -> dict:
    """Anchors for registered view `j`: sparse points projected into that view.

    Returns {a, b, n_anchors, resid_invdepth, resid_mm} -- `resid_mm` is the *median
    absolute* depth error of the aligned depth at the anchors, in **SfM units x 1000**
    (the SfM scale is arbitrary until the card fixes it downstream, so read it as a
    relative number; divide by the TSDF `unit_per_m` for millimetres).  Median rather
    than RMS on purpose: a handful of mis-triangulated sparse points would otherwise
    dominate the number and hide the alignment quality it is meant to report.
    """
    H, W = depth.shape[:2]
    # `xy` may be overridden with the *rectified* keypoint pixels when the caller has
    # undistorted `depth` (E1c radial/pinhole split); the 3D points are unaffected.
    xy = sfm.obs_xy[j] if xy is None else np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xyz = sfm.obs_xyz[j]
    out = {"a": float("nan"), "b": float("nan"), "n_anchors": 0,
           "resid_invdepth": float("inf"), "resid_mm": float("inf"),
           "grid": False, "resid_mm_affine": float("inf"),
           "a_nodes": None, "b_nodes": None,
           "nx": GRID_NX, "ny": GRID_NY}
    if xy.shape[0] == 0:
        return out

    w2c = sfm.w2c[j]
    z_ref = (xyz @ w2c[:3, :3].T + w2c[:3, 3][None, :])[:, 2]      # z-depth in that camera
    u = np.round(xy[:, 0] - 0.5).astype(int)
    v = np.round(xy[:, 1] - 0.5).astype(int)
    ok = (
        (u >= 0) & (u < W) & (v >= 0) & (v < H)
        & np.isfinite(z_ref) & (z_ref > 1e-6)
    )
    if not ok.any():
        return out
    u, v, z_ref = u[ok], v[ok], z_ref[ok]
    d_pred = depth[v, u].astype(np.float64)
    ok2 = np.isfinite(d_pred) & (d_pred > 1e-6) & valid[v, u]
    if ok2.sum() < 4:
        ok2 = np.isfinite(d_pred) & (d_pred > 1e-6)              # ignore the conf gate
    d_pred, z_ref, u, v = d_pred[ok2], z_ref[ok2], u[ok2], v[ok2]
    n = int(d_pred.size)
    if n < 4:
        return out

    def _median_mm(inv):
        good = inv > 1e-9
        dz = np.abs(1.0 / inv[good] - z_ref[good]) if good.any() else np.array([np.inf])
        return float(np.median(dz) * 1000.0)

    a, b, resid = solve_scale_shift(d_pred, z_ref)
    resid_affine_mm = _median_mm(a / d_pred + b)
    out.update(
        a=a, b=b, n_anchors=n, resid_invdepth=resid,
        resid_mm=resid_affine_mm, resid_mm_affine=resid_affine_mm,
        enough=n >= min_anchors,
    )

    # (E1b) spatially varying refit -- only with enough anchors to constrain the grid,
    # and only kept if it actually beats the affine fit at the anchors.
    # (E1c) resolution ladder: the fine 6 x 4 grid only when the view carries enough
    # anchors to constrain 35 nodes, the E1b 4 x 3 grid down to `grid_min_anchors`, the
    # plain affine fit below that.  Each rung is kept only if it beats the affine fit.
    if grid and n >= grid_min_anchors:
        nx, ny = (
            (GRID_NX, GRID_NY) if n >= GRID_FINE_MIN_ANCHORS
            else (GRID_NX_COARSE, GRID_NY_COARSE)
        )
        uv = np.stack([u + 0.5, v + 0.5], axis=1).astype(np.float64)
        a_nodes, b_nodes, gresid = solve_grid_correction(uv, d_pred, z_ref, W, H, nx, ny)
        idx_n, wts_n = _bilinear_weights(uv, W, H, nx, ny)
        inv_g = (a_nodes[idx_n] * wts_n).sum(1) / d_pred + (b_nodes[idx_n] * wts_n).sum(1)
        gm = _median_mm(inv_g)
        if np.isfinite(gm) and gm < resid_affine_mm:
            out.update(
                grid=True, resid_mm=gm, resid_invdepth=gresid,
                a_nodes=a_nodes, b_nodes=b_nodes, nx=nx, ny=ny,
            )
    return out


def apply_scale_shift(depth: np.ndarray, a: float, b: float) -> np.ndarray:
    """depth -> 1 / (a/depth + b), with non-positive results marked as 0."""
    d = np.asarray(depth, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = a / d + b
        out = np.where(inv > 1e-9, 1.0 / inv, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# 2b. spatially varying (grid) inverse-depth correction   [E1b]
# --------------------------------------------------------------------------------------
# A single (a, b) per view can only remove a *global* affine offset of the monocular
# inverse depth.  What is left on these captures is a long-wavelength bow of +/-10-15 mm
# across the frame (iPhone wide-lens distortion + the depth head's own field curvature),
# which shows up as a bowed TSDF top surface.  So instead of two numbers we fit two
# bilinear *fields*
#
#       1/d_corrected(u, v) = a(u, v) / d_pred(u, v) + b(u, v)
#
# on a coarse GRID_NX x GRID_NY cell grid (nodes at the cell corners), jointly, by the
# same Huber IRLS, plus a Tikhonov prior that (i) penalises differences between
# neighbouring nodes and (ii) pulls every node towards the global affine solution.  The
# prior is scaled by sqrt(n_anchors / n_cells), i.e. by the anchor count an *average*
# cell would hold, so a cell with far fewer anchors than average is dominated by its
# neighbours' values instead of fitting its own handful of points.

def _bilinear_weights(
    uv: np.ndarray, W: int, H: int, nx: int = GRID_NX, ny: int = GRID_NY
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear node weights for pixel coords `uv` [n,2].

    Returns (idx [n,4] node indices into a row-major (ny+1) x (nx+1) node grid,
    wts [n,4]).
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    s = np.clip(uv[:, 0] / max(W, 1) * nx, 0.0, nx - 1e-9)
    t = np.clip(uv[:, 1] / max(H, 1) * ny, 0.0, ny - 1e-9)
    i0 = np.floor(s).astype(int)
    j0 = np.floor(t).astype(int)
    fs = s - i0
    ft = t - j0
    n_nodes_x = nx + 1
    idx = np.stack(
        [
            j0 * n_nodes_x + i0,
            j0 * n_nodes_x + i0 + 1,
            (j0 + 1) * n_nodes_x + i0,
            (j0 + 1) * n_nodes_x + i0 + 1,
        ],
        axis=1,
    )
    wts = np.stack([(1 - fs) * (1 - ft), fs * (1 - ft), (1 - fs) * ft, fs * ft], axis=1)
    return idx, wts


def _node_edges(nx: int, ny: int) -> list[tuple[int, int]]:
    """4-neighbour edges of the (ny+1) x (nx+1) node lattice."""
    nnx = nx + 1
    edges = []
    for j in range(ny + 1):
        for i in range(nnx):
            k = j * nnx + i
            if i + 1 < nnx:
                edges.append((k, k + 1))
            if j + 1 <= ny:
                edges.append((k, k + nnx))
    return edges


def _node_laplacians(nx: int, ny: int) -> list[tuple[int, int, int]]:
    """Collinear node triples (i-1, i, i+1) along rows and columns.

    Second-order smoothness is the right prior here: the error we are modelling is a
    *bow*, which is dominated by a linear ramp across the frame, and a first-order
    (difference) prior shrinks exactly that ramp towards zero.  Penalising curvature
    instead lets any affine-in-(u,v) correction through for free while still keeping
    an anchor-starved cell from inventing its own value -- it gets the linear
    extrapolation of its neighbours.
    """
    nnx = nx + 1
    tri = []
    for j in range(ny + 1):
        for i in range(1, nnx - 1):
            k = j * nnx + i
            tri.append((k - 1, k, k + 1))
    for j in range(1, ny):
        for i in range(nnx):
            k = j * nnx + i
            tri.append((k - nnx, k, k + nnx))
    return tri


def solve_grid_correction(
    uv: np.ndarray,
    d_pred: np.ndarray,
    d_ref: np.ndarray,
    W: int,
    H: int,
    nx: int = GRID_NX,
    ny: int = GRID_NY,
    lam: float = GRID_LAMBDA,
    prior: float = GRID_PRIOR,
    g1: float = GRID_FIRST_ORDER,
    iters: int = 3,
    huber_k: float = 1.345,
    trim_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit bilinear fields a(u,v), b(u,v) so that 1/d_ref ~= a/d_pred + b.

    Returns (a_nodes [(ny+1)*(nx+1)], b_nodes [...], weighted rms inverse-depth residual).
    Falls back to the constant (global affine) fields when the system is degenerate.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x = 1.0 / np.asarray(d_pred, dtype=np.float64).ravel()
    y = 1.0 / np.asarray(d_ref, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    uv, x, y = uv[ok], x[ok], y[ok]
    n = int(x.size)
    n_nodes = (nx + 1) * (ny + 1)

    a0, b0, resid0 = solve_scale_shift(1.0 / np.maximum(x, 1e-12), 1.0 / np.maximum(y, 1e-12))
    a_flat = np.full(n_nodes, a0)
    b_flat = np.full(n_nodes, b0)
    # The Tikhonov + affine-prior blocks alone make the normal equations full rank, so a
    # view only needs enough anchors to say something beyond the prior; requiring 2 per
    # node (the E1b rule) would have locked the 6 x 4 grid out at its own anchor gate.
    if n < max(8, n_nodes // 2):
        return a_flat, b_flat, resid0

    idx, wts = _bilinear_weights(uv, W, H, nx, ny)
    # data block: row i has wts[i, k] * x[i] on a-node idx[i, k] and wts[i, k] on b-node
    rows = np.repeat(np.arange(n), 4)
    A_data = np.zeros((n, 2 * n_nodes), dtype=np.float64)
    np.add.at(A_data, (rows, idx.ravel()), (wts * x[:, None]).ravel())
    np.add.at(A_data, (rows, n_nodes + idx.ravel()), wts.ravel())

    # prior blocks.  s_x converts an `a` difference into the inverse-depth error it causes,
    # so a- and b-smoothness are penalised by their *effect*, not by their raw magnitude.
    s_x = float(np.sqrt(np.mean(x**2))) or 1.0
    cells = max(nx * ny, 1)
    # (E1c) `lam * sqrt(n / cells)` balanced prior against data *per cell*, so refining
    # 4x3 -> 6x4 would have quietly weakened the prior by sqrt(2) exactly where the cells
    # got hungrier.  Referencing the tuned 4 x 3 grid keeps the absolute node prior fixed,
    # so an anchor-starved cell in the finer grid still inherits its neighbours' values.
    g = lam * float(np.sqrt(max(n, 1) / GRID_REF_CELLS))
    edges = _node_edges(nx, ny)
    tri = _node_laplacians(nx, ny)
    blocks = []
    for (p, q) in edges:                        # weak first-order: keeps the fit bounded
        r_a = np.zeros(2 * n_nodes); r_a[p] = g1 * g * s_x; r_a[q] = -g1 * g * s_x
        r_b = np.zeros(2 * n_nodes)
        r_b[n_nodes + p] = g1 * g; r_b[n_nodes + q] = -g1 * g
        blocks += [r_a, r_b]
    for (p, q, r) in tri:                       # second-order: penalise curvature only
        r_a = np.zeros(2 * n_nodes)
        r_a[p] = g * s_x; r_a[q] = -2 * g * s_x; r_a[r] = g * s_x
        r_b = np.zeros(2 * n_nodes)
        r_b[n_nodes + p] = g; r_b[n_nodes + q] = -2 * g; r_b[n_nodes + r] = g
        blocks += [r_a, r_b]
    A_sm = np.stack(blocks, axis=0) if blocks else np.zeros((0, 2 * n_nodes))
    b_sm = np.zeros(A_sm.shape[0])

    gp = prior * g
    A_pr = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=np.float64)
    A_pr[np.arange(n_nodes), np.arange(n_nodes)] = gp * s_x
    A_pr[n_nodes + np.arange(n_nodes), n_nodes + np.arange(n_nodes)] = gp
    b_pr = np.concatenate([np.full(n_nodes, gp * s_x * a0), np.full(n_nodes, gp * b0)])

    w = np.ones(n)
    sol = np.concatenate([a_flat, b_flat])
    for _ in range(max(1, iters) + 1):
        sw = np.sqrt(w)[:, None]
        M = np.vstack([A_data * sw, A_sm, A_pr])
        rhs = np.concatenate([y * sw[:, 0], b_sm, b_pr])
        try:
            sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            return a_flat, b_flat, resid0
        r = A_data @ sol - y
        mad = float(np.median(np.abs(r - np.median(r))))
        s = 1.4826 * mad if mad > 0 else (float(np.std(r)) or 1e-9)
        u = np.abs(r) / (huber_k * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        if trim_sigma and trim_sigma > 0:
            w = np.where(np.abs(r) > trim_sigma * s, 0.0, w)

    if not np.isfinite(sol).all():
        return a_flat, b_flat, resid0
    r = A_data @ sol - y
    resid = float(np.sqrt(np.average(r**2, weights=w))) if w.sum() > 0 else float("inf")
    return sol[:n_nodes], sol[n_nodes:], resid


def _eval_grid(nodes: np.ndarray, W: int, H: int, nx: int = GRID_NX, ny: int = GRID_NY):
    """Render a node field as a full HxW bilinear image."""
    grid = np.asarray(nodes, dtype=np.float64).reshape(ny + 1, nx + 1)
    u = (np.arange(W) + 0.5) / W * nx
    v = (np.arange(H) + 0.5) / H * ny
    i0 = np.clip(np.floor(u).astype(int), 0, nx - 1)
    j0 = np.clip(np.floor(v).astype(int), 0, ny - 1)
    fs = (u - i0)[None, :]
    ft = (v - j0)[:, None]
    g00 = grid[np.ix_(j0, i0)]
    g10 = grid[np.ix_(j0, i0 + 1)]
    g01 = grid[np.ix_(j0 + 1, i0)]
    g11 = grid[np.ix_(j0 + 1, i0 + 1)]
    return (
        g00 * (1 - fs) * (1 - ft)
        + g10 * fs * (1 - ft)
        + g01 * (1 - fs) * ft
        + g11 * fs * ft
    )


def apply_grid_correction(
    depth: np.ndarray,
    a_nodes: np.ndarray,
    b_nodes: np.ndarray,
    nx: int = GRID_NX,
    ny: int = GRID_NY,
) -> np.ndarray:
    """depth -> 1 / (a(u,v)/depth + b(u,v)); non-positive inverse depth becomes 0."""
    d = np.asarray(depth, dtype=np.float64)
    H, W = d.shape[:2]
    A = _eval_grid(a_nodes, W, H, nx, ny)
    B = _eval_grid(b_nodes, W, H, nx, ny)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = A / d + B
        out = np.where(inv > 1e-9, 1.0 / inv, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def depth_edge_mask(depth: np.ndarray, thresh: float = EDGE_LOG_RANGE) -> np.ndarray:
    """True where the 3x3 *log* depth range exceeds `thresh` (a depth discontinuity).

    Log depth makes the threshold a relative one (5 % by default), so the same number
    works for the near table edge and the far floor.  TSDF fusion must not integrate
    these pixels: bilinear depth across an occlusion boundary invents a sloped surface
    that bridges foreground and background and smears the real one.
    """
    import cv2

    d = np.asarray(depth, dtype=np.float32)
    good = np.isfinite(d) & (d > 1e-6)
    ld = np.zeros_like(d, dtype=np.float32)
    ld[good] = np.log(d[good])
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(ld, k)
    lo = cv2.erode(ld, k)
    rng = hi - lo
    edge = rng > float(thresh)
    # a pixel next to a hole is an edge too
    edge |= cv2.dilate((~good).astype(np.uint8), k).astype(bool) & good
    return edge & good


# --------------------------------------------------------------------------------------
# 2c. (N2) cross-view geometric consistency
# --------------------------------------------------------------------------------------
def _sample_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample of `img` at pixel-centre coordinates (u, v).

    Out-of-bounds and any sample whose 2x2 support contains a hole (depth <= 0) come
    back as 0 -- "no measurement", never an interpolation between a surface and a hole.
    """
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    inside = (u0 >= 0) & (u0 < W - 1) & (v0 >= 0) & (v0 < H - 1)
    u0c = np.clip(u0, 0, W - 2)
    v0c = np.clip(v0, 0, H - 2)
    fu = np.clip(u - u0c, 0.0, 1.0)
    fv = np.clip(v - v0c, 0.0, 1.0)
    a = img[v0c, u0c]
    b = img[v0c, u0c + 1]
    c = img[v0c + 1, u0c]
    d = img[v0c + 1, u0c + 1]
    hole = (a <= 0) | (b <= 0) | (c <= 0) | (d <= 0)
    out = (
        a * (1 - fu) * (1 - fv) + b * fu * (1 - fv)
        + c * (1 - fu) * fv + d * fu * fv
    )
    return np.where(inside & ~hole, out, 0.0)


def geometric_consistency_filter(
    depths: list[np.ndarray],
    Ks: np.ndarray,
    c2ws: np.ndarray,
    k_neighbors: int = GEOM_K,
    rel_tol: float | None = None,
    min_agree: int = GEOM_MIN_AGREE,
    px_tol: float = GEOM_PX_TOL,
    min_kept_frac: float = GEOM_MIN_KEPT,
    max_rel_tol: float = GEOM_MAX_REL_TOL,
) -> tuple[list[np.ndarray], dict]:
    """Drop per-view depth pixels that no neighbouring view confirms.

    Everything upstream of this is *per view*: the confidence gate, the depth-edge mask
    and the grid correction all look at one depth map at a time, so a whole patch of
    plausible-but-wrong depth (repeating fabric texture, clutter a few centimetres off a
    wall) survives them intact and goes straight into the TSDF, where it is only diluted.
    This is the first check that asks the other views whether a pixel is real.

    For every view i and each of its `k_neighbors` nearest views j (by camera-centre
    distance), pixel p of view i is backprojected with its own depth, projected into j at
    `d_ij` / `p_j`, and compared with j's own depth `d_j = depth_j(p_j)` (bilinear):

      * relative depth disagreement  |d_ij - d_j| / d_j <= `rel_tol`, and
      * round-trip reprojection error: backproject `p_j` with `d_j`, project back into i,
        and require it to land within `px_tol` of p.

    A pixel survives when at least `min_agree` neighbours say yes.  `min_agree` is
    clamped to the number of neighbours actually available, so a 2-view scene needs 1
    agreement rather than dropping everything.

    Depth may be in any consistent unit (here: SfM units) -- both tests are scale free.
    Returns (filtered depths, info) with `kept_frac` per view and the global summary that
    the pipeline records as `cloud.recon.geom_consistency`.
    """
    # (N3) `rel_tol=None` (the default) means *per view, from the data*: the
    # disagreement a pixel actually shows against the views that can see it is a
    # direct measure of this capture's relative depth noise, so the gate is set
    # at 1.5 x its median instead of a fixed 2 %. `rel_tol` is scale free, and
    # 1.5 x median(rel) is exactly `1.5 x noise / depth` -- the mm form of the
    # same quantity. A floor of GEOM_REL_TOL keeps clean captures (table_a: 66 %
    # kept at 2 %) bit-identical, and no view may end up keeping less than
    # `min_kept_frac` of its pixels: a view that would be gutted gets its own
    # tolerance relaxed, up to `max_rel_tol`, instead of being thrown away.
    auto = rel_tol is None
    base_rel_tol = float(GEOM_REL_TOL if auto else rel_tol)
    M = len(depths)
    Ks = np.asarray(Ks, dtype=np.float64)
    c2ws = np.asarray(c2ws, dtype=np.float64)
    info = {
        "rel_tol": base_rel_tol,
        "rel_tol_auto": bool(auto),
        "rel_tol_per_view": [base_rel_tol] * M,
        "min_agree": int(min_agree),
        "px_tol": float(px_tol),
        "k_neighbors": int(k_neighbors),
        "kept_frac": [1.0] * M,
        "kept_frac_median": 1.0,
        "n_dropped": 0,
    }
    if M < 2:
        info["skipped"] = "single_view"
        return [np.asarray(d, np.float32) for d in depths], info

    t0 = time.time()
    H, W = depths[0].shape[:2]
    centres = c2ws[:, :3, 3]
    w2cs = np.stack([np.linalg.inv(c) for c in c2ws], axis=0)
    uu, vv = np.meshgrid(
        np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5
    )
    uu, vv = uu.ravel(), vv.ravel()

    out: list[np.ndarray] = []
    n_dropped = 0
    for i in range(M):
        d_i = np.asarray(depths[i], dtype=np.float64)
        sel = np.flatnonzero(d_i.ravel() > 0)
        dist = np.linalg.norm(centres - centres[i], axis=1)
        dist[i] = np.inf
        nbrs = np.argsort(dist)[: max(1, int(k_neighbors))]
        nbrs = [int(j) for j in nbrs if np.isfinite(dist[j])]
        need = int(min(max(1, min_agree), len(nbrs)))
        if sel.size == 0 or not nbrs:
            out.append(np.asarray(depths[i], np.float32))
            continue

        K_i = Ks[i]
        fx_i, fy_i = K_i[0, 0], K_i[1, 1]
        cx_i, cy_i = K_i[0, 2], K_i[1, 2]
        u_i, v_i = uu[sel], vv[sel]
        z_i = d_i.ravel()[sel]
        # camera rays of view i, then world points
        x_c = np.stack([(u_i - cx_i) / fx_i * z_i, (v_i - cy_i) / fy_i * z_i, z_i], axis=1)
        Xw = x_c @ c2ws[i][:3, :3].T + c2ws[i][:3, 3][None, :]

        rel_stack = np.full((len(nbrs), sel.size), np.inf, dtype=np.float32)
        for jj, j in enumerate(nbrs):
            K_j = Ks[j]
            x_j = Xw @ w2cs[j][:3, :3].T + w2cs[j][:3, 3][None, :]
            z_ij = x_j[:, 2]
            vis = z_ij > 1e-9
            zj_safe = np.where(vis, z_ij, 1.0)
            u_j = K_j[0, 0] * x_j[:, 0] / zj_safe + K_j[0, 2]
            v_j = K_j[1, 1] * x_j[:, 1] / zj_safe + K_j[1, 2]
            d_j = _sample_bilinear(np.asarray(depths[j], dtype=np.float64), u_j - 0.5, v_j - 0.5)
            ok = vis & (d_j > 0)
            if not ok.any():
                continue
            rel = np.full(sel.size, np.inf)
            np.divide(np.abs(z_ij - d_j), np.maximum(d_j, 1e-12), out=rel, where=ok)

            # round trip: the neighbour's own surface point, back in view i
            dj_safe = np.where(ok, d_j, 1.0)
            xb = np.stack([
                (u_j - K_j[0, 2]) / K_j[0, 0] * dj_safe,
                (v_j - K_j[1, 2]) / K_j[1, 1] * dj_safe,
                dj_safe,
            ], axis=1)
            Xb = xb @ c2ws[j][:3, :3].T + c2ws[j][:3, 3][None, :]
            xi = Xb @ w2cs[i][:3, :3].T + w2cs[i][:3, 3][None, :]
            zi_b = xi[:, 2]
            ok2 = ok & (zi_b > 1e-9)
            zib_safe = np.where(ok2, zi_b, 1.0)
            u_b = fx_i * xi[:, 0] / zib_safe + cx_i
            v_b = fy_i * xi[:, 1] / zib_safe + cy_i
            px = np.full(sel.size, np.inf)
            np.copyto(px, np.hypot(u_b - u_i, v_b - v_i), where=ok2)

            # the relative disagreement this neighbour reports, +inf where it
            # cannot vouch for the pixel at all (occluded, off-image, bad
            # round trip) -- so the gate below is a pure threshold on `rel`.
            good = ok2 & (px <= px_tol)
            np.copyto(rel_stack[jj], rel.astype(np.float32), where=good)

        # the `need`-th smallest disagreement: the pixel survives at tolerance t
        # exactly when this value is <= t, which makes kept_frac(t) free to
        # evaluate for any t and the per-view relaxation below a lookup.
        if len(nbrs) == 1:
            r_need = rel_stack[0]
        else:
            r_need = np.partition(rel_stack, need - 1, axis=0)[need - 1]
        del rel_stack
        finite = np.isfinite(r_need)
        view_tol = base_rel_tol
        if auto and finite.any():
            view_tol = max(base_rel_tol, 1.5 * float(np.median(r_need[finite])))
        if min_kept_frac > 0 and sel.size:
            if float((r_need <= view_tol).mean()) < min_kept_frac:
                q = float(np.quantile(np.where(finite, r_need, np.inf),
                                      float(min_kept_frac)))
                if np.isfinite(q):
                    view_tol = min(float(max_rel_tol), max(view_tol, q))
        view_tol = min(float(max_rel_tol), float(view_tol))
        info["rel_tol_per_view"][i] = round(float(view_tol), 5)

        keep = r_need <= view_tol
        d_out = d_i.copy()
        flat = d_out.ravel()
        flat[sel[~keep]] = 0.0
        n_dropped += int((~keep).sum())
        info["kept_frac"][i] = float(keep.sum() / max(1, sel.size))
        out.append(d_out.astype(np.float32))

    info["kept_frac_median"] = float(np.median(info["kept_frac"]))
    info["n_dropped"] = int(n_dropped)
    info["seconds"] = round(float(time.time() - t0), 2)
    return out, info


# --------------------------------------------------------------------------------------
# 3. TSDF fusion
# --------------------------------------------------------------------------------------
def tsdf_fuse(
    depths: list[np.ndarray],
    colors: list[np.ndarray],
    Ks: np.ndarray,
    w2cs: np.ndarray,
    unit: float = 1.0,
    voxel_m: float = 0.0025,
    sdf_trunc_m: float = 0.010,
    depth_trunc: float | None = None,
    sor: bool = True,
):
    """Integrate aligned depth maps into one ScalableTSDFVolume and extract cloud + mesh.

    `unit` is the number of SfM units per metre, so the voxel really is `voxel_m` metres
    across whatever arbitrary scale SfM chose.  `depths` are z-depth in SfM units with 0
    meaning "no measurement"; `colors` are HxWx3 uint8.

    Returns (pcd, mesh, info).
    """
    import open3d as o3d

    assert len(depths) == len(colors) == len(Ks) == len(w2cs)
    H, W = depths[0].shape[:2]
    dv = np.concatenate([d[d > 0].ravel() for d in depths]) if depths else np.zeros(0)
    med = float(np.median(dv)) if dv.size else 1.0
    if depth_trunc is None:
        depth_trunc = 3.0 * med

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_m * unit),
        sdf_trunc=float(sdf_trunc_m * unit),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    n_int = 0
    for d, c, K, w2c in zip(depths, colors, Ks, w2cs):
        if not np.isfinite(d).all():
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        depth_img = o3d.geometry.Image(np.ascontiguousarray(d, dtype=np.float32))
        color_img = o3d.geometry.Image(np.ascontiguousarray(c, dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0, depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        vol.integrate(rgbd, intr, np.asarray(w2c, dtype=np.float64))
        n_int += 1

    pcd = vol.extract_point_cloud()
    # (E1b) one statistical-outlier pass on the raw TSDF cloud.  The zero-crossing
    # extraction leaves a thin fringe of voxels wherever only one view saw the surface;
    # those are what thicken the top band and they are exactly what SOR removes.
    n_raw = len(pcd.points)
    n_sor = n_raw
    if sor and n_raw > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        n_sor = len(pcd.points)
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    info = {
        "n_integrated": n_int,
        "n_points_raw": n_raw,
        "n_points_sor": n_sor,
        "unit_per_m": float(unit),
        "voxel": float(voxel_m * unit),
        "sdf_trunc": float(sdf_trunc_m * unit),
        "depth_trunc": float(depth_trunc),
        "median_depth": med,
        "n_points": len(pcd.points),
        "n_tris": len(mesh.triangles),
    }
    return pcd, mesh, info


# --------------------------------------------------------------------------------------
# 4. glue: DA3 depth + SfM poses -> a Recon in SfM units, plus the TSDF cloud
# --------------------------------------------------------------------------------------
def build_sfm_recon(
    recon,
    frames: list,
    sfm: SfmResult,
    geom_filter: bool = True,
    geom_tol: float | None = None,
) -> tuple[object, dict]:
    """Replace `recon`'s poses/intrinsics/pointmap with the SfM ones + aligned depth.

    Only the SfM-registered views survive; the caller must subset `frames` (and the SAM
    masks) by `info["kept"]` so that index i of the returned Recon is still frames[i].

    `scale.py`, `frame.py`, the mount-surface search, `primitives.py` and `export.py`
    therefore keep working byte-for-byte: they only ever read `Recon.pointmap`, `.valid`,
    `.c2w` and `.K`.
    """
    from .recon import backproject
    from .types import Recon

    keep = list(sfm.idx)
    M = len(keep)
    H, W = recon.depth.shape[1:3]

    # ---- (E1c) radial -> pinhole rectification ---------------------------------------
    # One shared camera means one shared map.  Depth is remapped with NEAREST (bilinear
    # across a depth step invents a surface), conf bilinear; the caller rectifies the RGB
    # and the SAM masks with the same map so image-space lookups stay consistent.
    umaps = undistort_maps(sfm.K[0], sfm.k1, W, H) if sfm.k1 else None
    obs_xy_rect = [
        undistort_points(sfm.obs_xy[j], sfm.K[j], sfm.k1) if umaps is not None
        else sfm.obs_xy[j]
        for j in range(M)
    ]

    src_depth = [
        undistort_image(recon.depth[i], umaps, nearest=True) if umaps is not None
        else recon.depth[i]
        for i in keep
    ]
    src_conf = [
        undistort_image(recon.conf[i], umaps) if umaps is not None else recon.conf[i]
        for i in keep
    ]
    src_valid = [
        undistort_image(recon.valid[i], umaps) if umaps is not None else recon.valid[i]
        for i in keep
    ]

    stats = []
    for j, i in enumerate(keep):
        st = align_depth_to_sparse(
            src_depth[j], src_valid[j], sfm, j, xy=obs_xy_rect[j]
        )
        st["view"] = i
        stats.append(st)

    good = [s for s in stats if s["n_anchors"] >= MIN_ANCHORS and np.isfinite(s["a"])]
    if good:
        a_med = float(np.median([s["a"] for s in good]))
        b_med = float(np.median([s["b"] for s in good]))
    else:
        a_med, b_med = 1.0, 0.0

    warnings: list[str] = []
    depth = np.zeros((M, H, W), np.float32)
    conf = np.zeros((M, H, W), np.float32)
    K = np.zeros((M, 3, 3), np.float64)
    c2w = np.zeros((M, 4, 4), np.float64)
    pointmap = np.zeros((M, H, W, 3), np.float32)
    valid = np.zeros((M, H, W), bool)

    n_grid = 0
    for j, i in enumerate(keep):
        st = stats[j]
        if st["n_anchors"] < MIN_ANCHORS or not np.isfinite(st["a"]):
            st["a"], st["b"] = a_med, b_med
            st["fallback"] = True
            st["grid"] = False
            warnings.append(f"sfm_align_few_anchors_view={i}_n={st['n_anchors']}")
        if st.get("grid") and st.get("a_nodes") is not None:
            depth[j] = apply_grid_correction(
                src_depth[j], st["a_nodes"], st["b_nodes"],
                nx=int(st.get("nx", GRID_NX)), ny=int(st.get("ny", GRID_NY)),
            )
            n_grid += 1
        else:
            depth[j] = apply_scale_shift(src_depth[j], st["a"], st["b"])
        conf[j] = src_conf[j]
        K[j] = sfm.K[j]
        c2w[j] = np.linalg.inv(sfm.w2c[j])
        if np.linalg.det(c2w[j, :3, :3]) <= 0:
            c2w[j, :3, :3] = _nearest_rotation(c2w[j, :3, :3])
        pointmap[j], valid[j] = backproject(depth[j], conf[j], K[j], c2w[j])
        valid[j] &= depth[j] > 0

    resids = [s["resid_mm"] for s in stats if np.isfinite(s["resid_mm"])]
    resids_aff = [s["resid_mm_affine"] for s in stats if np.isfinite(s["resid_mm_affine"])]
    anchors = [s["n_anchors"] for s in stats]

    # ---- (E1b) what the TSDF actually integrates -------------------------------------
    # 1. depth-discontinuity pixels are dropped (bilinear depth across an occlusion
    #    boundary fabricates a ramp that the TSDF happily carves into the surface);
    # 2. the worst views by anchor residual are dropped entirely.  Open3D's
    #    ScalableTSDFVolume has no per-view weight, so `1/(1+(resid/2mm)^2)` is realised
    #    as a hard cut of the tail rather than a continuous weight.
    tsdf_depths = []
    n_edge = 0
    for j in range(M):
        d = np.where(valid[j], depth[j], 0.0).astype(np.float32)
        edge = depth_edge_mask(d)
        n_edge += int(edge.sum())
        d[edge] = 0.0
        tsdf_depths.append(d)

    # 3. (N2) cross-view geometric consistency: a pixel no neighbouring view confirms
    #    is dropped.  This is the only check that is not per-view, and it is what
    #    removes the coherent-but-wrong patches (repeating fabric texture, clutter near
    #    a wall) that the conf gate / edge mask / grid fit all accept.
    geom_info = {"enabled": bool(geom_filter)}
    if geom_filter and M >= 2:
        tsdf_depths, gi = geometric_consistency_filter(
            tsdf_depths, K, c2w,
            rel_tol=(float(geom_tol) if geom_tol is not None else None),
        )
        geom_info.update(gi)
        for j in range(M):
            stats[j]["geom_kept_frac"] = round(float(gi["kept_frac"][j]), 4)
        warnings.append(
            f"geom_filter_kept_frac_median={geom_info['kept_frac_median']:.3f}"
        )

    unit = max(float(np.median(depth[valid])) / 0.8, 1e-9) if valid.any() else 1.0
    order = list(range(M))
    resid_metric = [
        (stats[j]["resid_mm"] / unit if np.isfinite(stats[j]["resid_mm"]) else np.inf)
        for j in range(M)
    ]
    for j in range(M):
        stats[j]["resid_mm_metric"] = float(resid_metric[j])
        stats[j]["weight"] = float(1.0 / (1.0 + (resid_metric[j] / 2.0) ** 2))
    n_drop = int(np.floor(0.2 * M)) if M >= 6 else 0
    if n_drop:
        order = sorted(order, key=lambda j: resid_metric[j])[: M - n_drop]
        order.sort()
        dropped = [keep[j] for j in range(M) if j not in set(order)]
        warnings.append(f"tsdf_dropped_views={dropped}")
    out = Recon(depth=depth, conf=conf, K=K, c2w=c2w, pointmap=pointmap, valid=valid)
    out.meta = dict(getattr(recon, "meta", {}) or {})
    out.meta.update({"mode": "sfm_tsdf", "n_views": M})

    dv = depth[valid]
    info = {
        "kept": keep,
        "undistort_maps": umaps,
        "camera_model": str(sfm.camera_model),
        "k1": float(sfm.k1),
        "n_fine_grid_views": int(sum(
            1 for st in stats if st.get("grid") and int(st.get("nx", 0)) == GRID_NX
        )),
        "per_view": stats,
        "tsdf_depths": tsdf_depths,
        "geom_consistency": geom_info,
        "tsdf_views": order,
        "n_grid_views": n_grid,
        "n_edge_px": n_edge,
        "align_resid_mm_affine_median":
            float(np.median(resids_aff)) if resids_aff else float("nan"),
        "align_resid_mm_median": float(np.median(resids)) if resids else float("nan"),
        "anchors_median": float(np.median(anchors)) if anchors else 0.0,
        "anchors_min": int(min(anchors)) if anchors else 0,
        "median_depth_sfm": float(np.median(dv)) if dv.size else 1.0,
        "warnings": warnings,
    }
    return out, info
