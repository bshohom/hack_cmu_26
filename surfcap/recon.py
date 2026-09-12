"""surfcap.recon -- Depth Anything 3 multi-view reconstruction.

DA3 API facts (read from third_party/da3/src/depth_anything_3, 2026-09-11):

  model = DepthAnything3.from_pretrained(ckpt).to(device)          # PyTorchModelHubMixin
  pred  = model.inference(image=[np.ndarray|PIL|str, ...],
                          process_res=504,                          # DA3 default
                          process_res_method="upper_bound_resize",  # long edge -> process_res,
                                                                    # then each dim rounded to a
                                                                    # multiple of PATCH_SIZE=14
                          export_dir=None)                          # -> depth_anything_3.specs.Prediction

  Prediction fields (specs.py):
      depth       (N, H_p, W_p) float32   -- Z-DEPTH, not ray distance
      conf        (N, H_p, W_p) float32   -- NOT in [0,1] (DA3's own default conf_thresh is 1.05)
      extrinsics  (N, 4, 4)               -- WORLD-TO-CAMERA (w2c), OpenCV convention
      intrinsics  (N, 3, 3)               -- in PROCESSED pixel units (H_p, W_p), not input units
      processed_images (N, H_p, W_p, 3) uint8
      is_metric, sky, gaussians, aux, scale_factor

  Depth / extrinsics convention is pinned by DA3's own exporter
  (utils/export/glb.py::_depths_to_world_points_with_colors), which does exactly:

      K_inv = inv(K[i]);  c2w = inv(as_4x4(ext_w2c[i]))
      rays  = K_inv @ [u, v, 1]^T ;  Xc = rays * depth ;  Xw = (c2w @ [Xc,1])[:3]

  i.e. depth is z-depth and `prediction.extrinsics` is w2c (the argument is literally named
  `ext_w2c`).  DA3 uses integer pixel coordinates; we use (u+0.5, v+0.5) per the surfcap plan,
  a half-pixel difference that is harmless and slightly more correct.

Because DA3 resizes internally (process_res, long edge), this module resamples depth/conf back
to each Frame's exact (H, W) and rescales K accordingly, so recon.depth[i] is pixel-for-pixel
aligned with frames[i].rgb.  Segmentation masks of shape (N, H, W) can therefore index the
pointmap directly.

NOTE: run_recon() may subset the view list (n_max / OOM ladder).  When it does it MUTATES the
caller's `frames` list in place (frames[:] = kept) so that index i of the returned Recon always
refers to frames[i].
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from surfcap.types import Frame, Recon

try:  # open3d is only needed by fuse_views
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

DEFAULT_CKPT = "depth-anything/DA3-LARGE-1.1"
SMALL_CKPT = "depth-anything/DA3-SMALL"
DEFAULT_PROCESS_RES = 512  # (F8) LARGE@512 beats BASE@768 on every accuracy metric on a 10 GB
# GPU (E2 A/B: table_a top rms 2.92 vs 3.54 mm; partition_e 2.12 vs 3.23 mm); LARGE @768+ OOMs.
DEFAULT_N_MAX = 10


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(__import__("torch").cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _even_indices(n_total: int, n_keep: int) -> list[int]:
    """Evenly spaced indices covering [0, n_total)."""
    if n_keep >= n_total:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, n_keep).round().astype(int).tolist()))


def _as_4x4(ext: np.ndarray) -> np.ndarray:
    """(3,4) or (4,4) -> (4,4)."""
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext.copy()
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = ext
        return out
    raise ValueError(f"extrinsics must be (3,4) or (4,4), got {ext.shape}")


def _nearest_rotation(R: np.ndarray) -> np.ndarray:
    """Closest proper rotation (det=+1) to R, via SVD."""
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def _resample_to_frame(
    depth_p: np.ndarray, conf_p: np.ndarray, K_p: np.ndarray, H: int, W: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample DA3's processed-resolution outputs onto the frame grid and rescale K.

    depth: nearest (never interpolate across a depth discontinuity).
    conf : linear.
    K    : row 0 (fx, skew, cx) scaled by W/W_p, row 1 (fy, cy) by H/H_p -- the same rule DA3's
           own InputProcessor._resize_ixt uses, and consistent with the (u+0.5) pixel-centre
           convention used in backproject().
    """
    H_p, W_p = depth_p.shape[-2:]
    K = np.asarray(K_p, dtype=np.float64).copy()
    if (H_p, W_p) == (H, W):
        return (
            np.ascontiguousarray(depth_p, dtype=np.float32),
            np.ascontiguousarray(conf_p, dtype=np.float32),
            K,
        )
    depth = cv2.resize(depth_p.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    conf = cv2.resize(conf_p.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    K[0, :] *= W / float(W_p)
    K[1, :] *= H / float(H_p)
    return depth, conf, K


# --------------------------------------------------------------------------------------
# EXIF intrinsics
# --------------------------------------------------------------------------------------
DIAG35_MM = 43.266615305567875  # sqrt(36^2 + 24^2)


def exif_focal_px(path: str, W: int, H: int) -> float | None:
    """Pinhole focal length in pixels for an image rendered at (W, H), from EXIF.

    Uses FocalLengthIn35mmFilm (EXIF tag 41989) when present:
        f_px = f35 * sqrt(W^2 + H^2) / 43.267
    which assumes the crop keeps the full 3:2-equivalent diagonal FOV (true here: the
    iPhone frames are 4:3 both originally and after our long-edge resize).

    Returns None if the tag is missing/unusable.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            ifd = exif.get_ifd(0x8769) or {}
        f35 = ifd.get(41989) or exif.get(41989)
        if f35 is None:
            return None
        f35 = float(f35)
        if not np.isfinite(f35) or f35 <= 0:
            return None
        return f35 * float(np.hypot(W, H)) / DIAG35_MM
    except Exception:  # noqa: BLE001 -- EXIF is best-effort
        return None


def exif_intrinsics(frames: list[Frame]) -> np.ndarray | None:
    """[N,3,3] float64 K for `frames` in each frame's own rgb pixel units, or None.

    All views share one K (same lens, same processed size); a single missing EXIF focal
    disables the whole thing so we never mix EXIF and predicted intrinsics.
    """
    if not frames:
        return None
    Ks = []
    for f in frames:
        H, W = f.rgb.shape[:2]
        fpx = exif_focal_px(f.path, W, H)
        if fpx is None:
            return None
        Ks.append(
            np.array([[fpx, 0.0, W / 2.0], [0.0, fpx, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        )
    return np.stack(Ks, axis=0)


def _scale_K(K: np.ndarray, W_from: int, H_from: int, W_to: int, H_to: int) -> np.ndarray:
    """Rescale K between two renderings of the same view (DA3 InputProcessor._resize_ixt rule)."""
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= W_to / float(W_from)
    K[1, :] *= H_to / float(H_from)
    return K


# --------------------------------------------------------------------------------------
# backprojection
# --------------------------------------------------------------------------------------
def backproject(
    depth: np.ndarray,
    conf: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    conf_floor: float = 0.5,
    pct: float = 40.0,
    edge_thresh: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one view's z-depth map into world coordinates.

    X_cam = K^-1 [u+0.5, v+0.5, 1]^T * D   (D is z-depth, per DA3's own exporter)
    X_world = c2w @ [X_cam, 1]

    Gates applied to `valid`:
      * finite, positive depth
      * conf >= max(conf_floor, percentile(conf, pct)).  DA3's conf is NOT in [0,1] (its own
        default threshold is 1.05), so when max(conf) > 1 the floor is inert and the percentile
        alone decides -- see the `conf_not_unit_range` warning emitted by run_recon.
      * edge kill: 3x3 range of log(depth) > edge_thresh

    Returns (pointmap [H,W,3] float32 world xyz, valid [H,W] bool).
    """
    depth = np.asarray(depth, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    H, W = depth.shape[:2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pix = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (HW,3)

    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    rays = pix @ K_inv.T                                    # (HW,3) unit-z rays
    X_cam = rays * depth.reshape(-1, 1).astype(np.float64)  # z-depth scaling
    c2w = np.asarray(c2w, dtype=np.float64)
    X_w = X_cam @ c2w[:3, :3].T + c2w[:3, 3][None, :]
    pointmap = X_w.reshape(H, W, 3).astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(pointmap).all(axis=-1)

    # ---- confidence gate -------------------------------------------------------------
    if conf is not None and conf.size:
        finite_conf = conf[np.isfinite(conf)]
        thr = float(np.percentile(finite_conf, pct)) if finite_conf.size else -np.inf
        if finite_conf.size and float(finite_conf.max()) <= 1.0 + 1e-6:
            thr = max(float(conf_floor), thr)   # conf really is a probability
        valid &= np.isfinite(conf) & (conf >= thr)

    # ---- edge kill on 3x3 range of log depth -----------------------------------------
    if edge_thresh is not None and edge_thresh > 0:
        pos = depth > 0
        logd = np.empty_like(depth, dtype=np.float32)
        fill = float(np.log(np.median(depth[pos]))) if pos.any() else 0.0
        logd[:] = fill
        np.log(depth, out=logd, where=pos)
        kern = np.ones((3, 3), np.uint8)
        rng = cv2.dilate(logd, kern) - cv2.erode(logd, kern)
        valid &= rng <= float(edge_thresh)

    return pointmap, valid.astype(bool)


# --------------------------------------------------------------------------------------
# recon
# --------------------------------------------------------------------------------------
def _build_recon(
    frames: list[Frame],
    depth_p: np.ndarray,
    conf_p: np.ndarray,
    K_p: np.ndarray,
    ext_w2c: np.ndarray,
    conf_floor: float,
    pct: float,
    edge_thresh: float,
    warnings: list[str],
) -> Recon:
    N = len(frames)
    H, W = frames[0].rgb.shape[:2]
    depth = np.zeros((N, H, W), np.float32)
    conf = np.zeros((N, H, W), np.float32)
    K = np.zeros((N, 3, 3), np.float64)
    c2w = np.zeros((N, 4, 4), np.float64)
    pointmap = np.zeros((N, H, W, 3), np.float32)
    valid = np.zeros((N, H, W), bool)

    conf_max = float(np.nanmax(conf_p)) if conf_p is not None and conf_p.size else 0.0
    if conf_p is None or conf_max > 1.0 + 1e-6:
        warnings.append(f"conf_not_unit_range_max={conf_max:.3f}_percentile_only")

    n_det_bad = 0
    for i, f in enumerate(frames):
        d, c, k = _resample_to_frame(depth_p[i], conf_p[i], K_p[i], H, W)
        w2c = _as_4x4(ext_w2c[i])
        M = np.linalg.inv(w2c)                      # cam-to-world
        R = M[:3, :3]
        det = float(np.linalg.det(R))
        if det <= 0:
            n_det_bad += 1
            warnings.append(f"recon_det_negative_view={i}_det={det:.4f}")
            M = M.copy()
            M[:3, :3] = _nearest_rotation(R)        # project to nearest proper rotation
        depth[i], conf[i], K[i], c2w[i] = d, c, k, M
        pointmap[i], valid[i] = backproject(
            d, c, k, M, conf_floor=conf_floor, pct=pct, edge_thresh=edge_thresh
        )

    # ---- degeneracy checks ------------------------------------------------------------
    centres = c2w[:, :3, 3]
    if len(centres) >= 2:
        dists = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        spread = float(dists.max())
    else:
        spread = 0.0
    dv = depth[valid]
    med_depth = float(np.median(dv)) if dv.size else 0.0
    if n_det_bad > 0 or (med_depth > 0 and spread < 0.01 * med_depth):
        warnings.append(
            f"poses_degenerate (cam_spread={spread:.5f}, median_depth={med_depth:.5f}, "
            f"n_det_negative={n_det_bad})"
        )

    return Recon(depth=depth, conf=conf, K=K, c2w=c2w, pointmap=pointmap, valid=valid)


def run_recon(
    frames: list[Frame],
    n_max: int = DEFAULT_N_MAX,
    ckpt: str = DEFAULT_CKPT,
    device: str = "cuda",
    process_res: int = DEFAULT_PROCESS_RES,
    conf_floor: float = 0.5,
    pct: float = 40.0,
    edge_thresh: float = 0.05,
    use_exif_intrinsics: bool = True,
) -> tuple[Recon, list[str]]:
    """Run DA3 on `frames` and return (Recon, warnings).

    Never raises except on an empty frame list.  On CUDA OOM it walks the ladder
        views n_max -> 12 -> 10   (evenly spaced)
        process_res -> 512
        ckpt -> DA3-SMALL
    and appends `recon_oom_reduced_views=N`.

    MUTATES `frames` in place whenever views are dropped, so that Recon index i == frames[i].
    """
    import torch

    warnings: list[str] = []
    if not frames:
        raise ValueError("zero images")

    # frames of mixed size cannot be stacked into Recon's [N,H,W] arrays
    sizes = [f.size for f in frames]
    if len(set(sizes)) > 1:
        modal = Counter(sizes).most_common(1)[0][0]
        dropped = [i for i, s in enumerate(sizes) if s != modal]
        warnings.append(f"frames_size_mismatch_dropped={len(dropped)}")
        frames[:] = [f for f in frames if f.size == modal]

    if device == "cuda" and not torch.cuda.is_available():
        warnings.append("cuda_unavailable_using_cpu")
        device = "cpu"

    n0 = min(n_max, len(frames))
    ladder: list[tuple[int, int, str]] = [(n0, process_res, ckpt)]
    for n in (12, 10):
        if n < n0:
            ladder.append((n, process_res, ckpt))
    n_last = ladder[-1][0]
    for res in (768, 512):
        if res < process_res:
            ladder.append((n_last, res, ckpt))
    if ckpt != SMALL_CKPT:
        ladder.append((ladder[-1][0], ladder[-1][1], SMALL_CKPT))

    last_exc: BaseException | None = None
    for attempt, (n_views, res, ck) in enumerate(ladder):
        idx = _even_indices(len(frames), n_views)
        sel = [frames[i] for i in idx]
        model = None
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            K_exif = exif_intrinsics(sel) if use_exif_intrinsics else None
            model = DepthAnything3_from_pretrained(ck).to(device)
            model.eval()
            pred = model.inference(
                image=[f.rgb for f in sel],
                intrinsics=K_exif,
                process_res=res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
            depth_p = np.asarray(pred.depth, dtype=np.float32)
            conf_p = (
                np.asarray(pred.conf, dtype=np.float32)
                if pred.conf is not None
                else np.ones_like(depth_p)
            )
            K_p = np.asarray(pred.intrinsics, dtype=np.float64)
            ext_p = np.asarray(pred.extrinsics, dtype=np.float64)   # w2c
            del pred
            if K_exif is not None:
                # DA3 only *consumes* `intrinsics` when `extrinsics` is also given
                # (model/da3.py: `if extrinsics is not None: cam_token = self.cam_enc(...)`)
                # and only echoes them back into prediction.intrinsics in that same case
                # (api.py::_align_to_input_extrinsics_intrinsics returns early without
                # extrinsics).  So substitute our EXIF K for the predicted one here, in the
                # PROCESSED pixel units that prediction.intrinsics uses.
                H_p, W_p = depth_p.shape[-2:]
                K_p = np.stack(
                    [
                        _scale_K(K_exif[i], sel[i].rgb.shape[1], sel[i].rgb.shape[0], W_p, H_p)
                        for i in range(len(sel))
                    ],
                    axis=0,
                )
        except BaseException as exc:  # noqa: BLE001 -- never crash the pipeline
            last_exc = exc
            if _is_oom(exc) and attempt + 1 < len(ladder):
                warnings.append(
                    f"recon_oom (views={n_views}, process_res={res}, ckpt={ck}); retrying"
                )
                continue
            if attempt + 1 < len(ladder):
                warnings.append(f"recon_failed ({type(exc).__name__}: {exc}); retrying")
                continue
            raise
        finally:
            if model is not None:
                del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        if len(sel) != len(frames):
            if attempt > 0:
                warnings.append(f"recon_oom_reduced_views={len(sel)}")
            else:  # deliberate n_max subset, not a failure
                warnings.append(f"recon_subset_views={len(sel)}_of_{len(frames)}")
            frames[:] = sel
        if res != process_res:
            warnings.append(f"recon_reduced_process_res={res}")
        if ck != ckpt:
            # (F8) loud warning: the checkpoint that actually ran differs from the one
            # requested (an OOM-ladder fallback), so callers/target.json must not silently
            # report the requested ckpt as if it had run.
            warnings.append(f"recon_ckpt_fallback={ck}")

        warnings.append(f"intrinsics_from_exif={'true' if K_exif is not None else 'false'}")
        recon = _build_recon(
            frames, depth_p, conf_p, K_p, ext_p, conf_floor, pct, edge_thresh, warnings
        )
        recon.meta = {"ckpt": ck, "process_res": res, "n_views": len(sel)}
        return recon, warnings

    raise RuntimeError(f"DA3 inference failed on every rung of the ladder: {last_exc}")


def DepthAnything3_from_pretrained(ckpt: str):
    """Deferred import so that `import surfcap.recon` works without DA3 installed."""
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(ckpt)


# --------------------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------------------
def fuse_views(
    recon: Recon,
    frames: list[Frame],
    mask: np.ndarray | None = None,
    max_per_view: int = 60000,
    max_total: int = 800000,
    seed: int = 0,
):
    """Fuse per-view pointmaps into one coloured o3d cloud.

    Points are those already gated by backproject (finite depth + conf percentile + edge kill),
    optionally restricted further by a per-view boolean `mask` of shape [N,H,W].
    Subsampling is seeded and applied per view (max_per_view) then globally (max_total).
    No voxel downsample / outlier removal here -- that happens after metric scaling.

    Returns (o3d.geometry.PointCloud, view_idx int32 [P]).
    """
    if o3d is None:  # pragma: no cover
        raise ImportError("open3d is required for fuse_views")
    rng = np.random.default_rng(seed)
    N = recon.pointmap.shape[0]
    pts_all, col_all, vid_all = [], [], []

    for i in range(min(N, len(frames))):
        sel = recon.valid[i].copy()
        if mask is not None:
            sel &= np.asarray(mask[i], dtype=bool)
        flat = np.flatnonzero(sel.reshape(-1))
        if flat.size == 0:
            continue
        if flat.size > max_per_view:
            flat = rng.choice(flat, size=max_per_view, replace=False)
        pts_all.append(recon.pointmap[i].reshape(-1, 3)[flat])
        col_all.append(frames[i].rgb.reshape(-1, 3)[flat])
        vid_all.append(np.full(flat.size, i, dtype=np.int32))

    pcd = o3d.geometry.PointCloud()
    if not pts_all:
        return pcd, np.zeros((0,), np.int32)

    pts = np.concatenate(pts_all, 0).astype(np.float64)
    cols = np.concatenate(col_all, 0)
    vid = np.concatenate(vid_all, 0)
    if pts.shape[0] > max_total:
        keep = rng.choice(pts.shape[0], size=max_total, replace=False)
        keep.sort()
        pts, cols, vid = pts[keep], cols[keep], vid[keep]

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    return pcd, vid



def _debug_cloud_report(pcd, outdir: Path, seed: int = 0) -> None:
    """CPU-only: dominant-plane fit + top/side scatter renders of the fused cloud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors)
    if P.shape[0] < 100:
        print("cloud too small for plane fit")
        return

    lo, hi = P.min(0), P.max(0)
    diag = float(np.linalg.norm(hi - lo))
    print(f"\ncloud bbox min={np.round(lo,3)} max={np.round(hi,3)} diag={diag:.4f}")

    dist = 0.01 * diag
    plane, inl = pcd.segment_plane(
        distance_threshold=dist, ransac_n=3, num_iterations=2000
    )
    a, b, c, d = plane
    inl = np.asarray(inl, dtype=np.int64)
    frac = inl.size / P.shape[0]
    res = np.abs(P[inl] @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])
    rms = float(np.sqrt((res**2).mean()))
    print(
        f"dominant plane n=[{a:.4f},{b:.4f},{c:.4f}] d={d:.4f}  "
        f"dist_thresh={dist:.5f} ({1.0:.0f}% of diag)"
    )
    print(
        f"plane inliers {inl.size}/{P.shape[0]} = {100*frac:.1f}%   "
        f"RMS={rms:.5f} units = {rms/diag:.5f} x bbox diag"
    )

    rng = np.random.default_rng(seed)
    k = min(20000, P.shape[0])
    sub = rng.choice(P.shape[0], size=k, replace=False)
    Ps, Cs = P[sub], np.clip(C[sub], 0, 1)

    for name, (i0, i1), lab in (
        ("fused_top.png", (0, 1), ("X", "Y")),
        ("fused_side.png", (0, 2), ("X", "Z")),
    ):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        ax.scatter(Ps[:, i0], Ps[:, i1], c=Cs, s=0.6, linewidths=0, marker=".")
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
        ax.set_aspect("equal")
        ax.set_title(f"fused cloud {lab[0]}{lab[1]}  ({k} of {P.shape[0]} pts)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / name)
        plt.close(fig)
        print(f"wrote {outdir/name}")



# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------
def _main() -> int:
    import torch

    from surfcap.io_images import load_folder

    ap = argparse.ArgumentParser(prog="python -m surfcap.recon")
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--res", type=int, default=DEFAULT_PROCESS_RES)
    ap.add_argument("--out", default="out/debug")
    args = ap.parse_args()

    t0 = time.time()
    frames, warns = load_folder(args.folder, min_keep=min(args.n, 8), max_images=None)
    print(f"loaded {len(frames)} frames in {time.time()-t0:.1f}s; io warnings={warns}")
    print(f"frame size (W,H)={frames[0].size}  rgb={frames[0].rgb.shape} {frames[0].rgb.dtype}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    recon, rwarns = run_recon(frames, n_max=args.n, ckpt=args.ckpt, process_res=args.res)
    t_recon = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0

    N, H, W = recon.depth.shape
    print(f"\n--- recon ({t_recon:.1f}s, peak VRAM {peak:.0f} MiB) ---")
    print(f"depth    {recon.depth.shape} {recon.depth.dtype}")
    print(f"conf     {recon.conf.shape}  range [{recon.conf.min():.3f}, {recon.conf.max():.3f}]")
    print(f"K        {recon.K.shape}\n{np.round(recon.K[0], 2)}")
    print(f"c2w      {recon.c2w.shape}")
    print(f"pointmap {recon.pointmap.shape}  valid {recon.valid.shape}")
    assert (H, W) == (frames[0].rgb.shape[0], frames[0].rgb.shape[1]), "recon/frame misaligned!"
    print(f"ALIGNMENT OK: recon HxW {(H, W)} == frame.rgb HxW {frames[0].rgb.shape[:2]}")

    dets = np.array([np.linalg.det(recon.c2w[i, :3, :3]) for i in range(N)])
    print(f"det(R): min={dets.min():.6f} max={dets.max():.6f}  all>0={bool((dets > 0).all())}")

    dv = recon.depth[recon.valid]
    print(
        f"depth valid: n={dv.size} ({100*dv.size/recon.valid.size:.1f}%)  "
        f"min={dv.min():.4f} med={np.median(dv):.4f} max={dv.max():.4f}"
    )
    print(f"valid per view: {recon.valid.reshape(N, -1).sum(1).tolist()}")

    C = recon.c2w[:, :3, 3]
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    iu = np.triu_indices(N, 1)
    print(
        f"cam-centre pairwise: min={D[iu].min():.5f} med={np.median(D[iu]):.5f} "
        f"max={D[iu].max():.5f}  spread/median_depth={D[iu].max()/max(np.median(dv),1e-9):.3f}"
    )

    t2 = time.time()
    pcd, vid = fuse_views(recon, frames)
    print(f"fused {len(pcd.points)} points from {len(np.unique(vid))} views in {time.time()-t2:.1f}s")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(outdir / "fused.ply"), pcd)
    with open(outdir / "cams.json", "w") as fh:
        json.dump(
            {
                "paths": [f.path for f in frames],
                "c2w": recon.c2w.tolist(),
                "K": recon.K.tolist(),
                "size_wh": [list(f.size) for f in frames],
            },
            fh,
            indent=2,
        )
    print(f"wrote {outdir/'fused.ply'} and {outdir/'cams.json'}")

    # ---- GPU-free sanity: dominant plane + scatter renders --------------------------
    try:
        _debug_cloud_report(pcd, outdir, seed=0)
    except Exception as exc:  # noqa: BLE001 -- debug only, never fail the run
        print(f"cloud report failed: {type(exc).__name__}: {exc}")

    print(f"warnings: {rwarns}")
    print(f"TOTAL {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
