"""End-to-end surfcap pipeline: images -> metric, Z-up, segmented table target.

Stage order is fixed by the GPU budget: SAM 3 loads, runs, and is fully freed
*before* DA3 is ever constructed.  Every stage is wrapped so a failure degrades
(per the plan's failure table) instead of crashing; failures are recorded in
``warnings`` as ``stage_failed=<name>:<Exc>``.
"""
from __future__ import annotations

import gc
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .types import CARD_MM, Target

__all__ = ["run", "gt_check", "render_views"]

# (F2-3) XY dilation of the mount-surface hull used to grow the object body downward.
# 3 cm inflated table_a's top to 0.430 m wide against a 0.35 m truth; 1.5 cm still
# swallows a rounded edge/apron without eating the neighbourhood.
_HULL_DILATE_M = 0.015

# (F2-1) Generic *appearance* prompts for the reference card.  They rescue the ledge
# (SAM scores the white Costco card 0.98 "sticker" / 0.97 "label" but only 0.19 "card")
# but they also latch onto any other pale rectangle, which cost the cabinet its top.
# So they form a second tier: the whole 0.5/0.35/0.2 threshold ladder is walked with the
# card-word prompts first, and these are only reached when that finds nothing.
_CARD_PROMPTS_APPEARANCE = ("white card", "sticker", "label", "white rectangle")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _vram_reset():
    t = _cuda()
    if t is not None:
        t.cuda.reset_peak_memory_stats()


def _vram_peak_gb() -> float:
    t = _cuda()
    if t is None:
        return 0.0
    return float(t.cuda.max_memory_reserved()) / 1024**3


def _free_gpu():
    gc.collect()
    t = _cuda()
    if t is not None:
        t.cuda.empty_cache()
        t.cuda.ipc_collect()
    gc.collect()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)


@contextmanager
def _stage(name: str, timings: dict, warnings: list):
    t0 = time.time()
    print(f"[surfcap] --- {name} ---", flush=True)
    try:
        yield
    except Exception as e:  # noqa: BLE001 - never crash
        warnings.append(f"stage_failed={name}:{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        dt = time.time() - t0
        timings[name] = round(dt, 3)
        print(f"[surfcap] {name}: {dt:.2f}s", flush=True)


def _apron_points(
    table_xyz: np.ndarray,
    full_xyz: np.ndarray,
    full_rgb: np.ndarray,
    expand_m: float = 0.025,
    z_lo: float = -0.08,
    z_hi: float = -0.004,
    max_pts: int = 40000,
    seed: int = 0,
):
    """Points of the apron/front rail hanging just below the table top.

    The SAM "table" mask covers only the top face, so the table cloud has no vertical
    surface at all and thickness is unmeasurable.  Recover it geometrically: take the
    convex hull of the top face in XY, dilate it by `expand_m` (exact outward edge
    offset, not a scale about the centroid), and keep full-cloud points inside it whose
    z sits in the thin band just under z=0.

    Returns (xyz [M,3], rgb [M,3] float in 0..1) -- possibly empty.
    """
    import cv2

    empty = (np.zeros((0, 3)), np.zeros((0, 3)))
    if len(table_xyz) < 50 or len(full_xyz) < 50:
        return empty
    band = (full_xyz[:, 2] >= z_lo) & (full_xyz[:, 2] <= z_hi)
    if band.sum() < 50:
        return empty
    cand = full_xyz[band]
    cand_rgb = full_rgb[band] if len(full_rgb) == len(full_xyz) else np.zeros((band.sum(), 3))

    hull = cv2.convexHull(table_xyz[:, :2].astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return empty
    c = hull.mean(axis=0)
    v0, v1 = hull, np.roll(hull, -1, axis=0)
    e = v1 - v0
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)              # edge normals
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ok = ln[:, 0] > 1e-9
    nrm, v0 = nrm[ok] / ln[ok], v0[ok]
    flip = (np.einsum("ij,ij->i", nrm, v0 - c) < 0)          # force outward
    nrm[flip] *= -1.0
    # inside the dilated convex hull <=> every outward edge offset <= expand_m
    sd = (cand[:, None, :2] - v0[None, :, :]) * nrm[None, :, :]
    inside = (sd.sum(axis=2) <= expand_m).all(axis=1)
    xyz, rgb = cand[inside], cand_rgb[inside]
    if len(xyz) > max_pts:
        keep = np.random.default_rng(seed).choice(len(xyz), size=max_pts, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


def _inside_dilated_hull(xy_query: np.ndarray, hull_src_xy: np.ndarray, expand_m: float):
    """Boolean mask: which of `xy_query` [N,2] lie inside convexHull(hull_src_xy)
    offset outward by `expand_m` (exact edge offset, not a scale about the centroid)."""
    import cv2

    xy_query = np.asarray(xy_query, dtype=np.float64)[:, :2]
    if len(xy_query) == 0:
        return np.zeros(0, dtype=bool)
    src = np.asarray(hull_src_xy, dtype=np.float32)[:, :2]
    if len(src) < 3:
        return np.zeros(len(xy_query), dtype=bool)
    hull = cv2.convexHull(src).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return np.zeros(len(xy_query), dtype=bool)
    c = hull.mean(axis=0)
    v0, v1 = hull, np.roll(hull, -1, axis=0)
    e = v1 - v0
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ok = ln[:, 0] > 1e-9
    nrm, v0 = nrm[ok] / ln[ok], v0[ok]
    flip = np.einsum("ij,ij->i", nrm, v0 - c) < 0
    nrm[flip] *= -1.0
    out = np.ones(len(xy_query), dtype=bool)
    step = 200000
    for a in range(0, len(xy_query), step):
        q = xy_query[a:a + step]
        sd = (q[:, None, :] - v0[None, :, :]) * nrm[None, :, :]
        out[a:a + step] = (sd.sum(axis=2) <= expand_m).all(axis=1)
    return out


def _drop_card_geometric(pcd, card_xy, expand_m: float = 0.006, z_abs: float = 0.008):
    """(F1-A) Delete every point whose XY lies inside the card polygon (dilated by
    `expand_m`) and whose |z| < `z_abs`.

    The SAM card mask only covers the views where SAM actually found the card
    (7/11 on table_a); the remaining views fuse the card's own pixels back into the
    object through the table mask.  In the world frame the card is a known polygon
    at z ~= 0, so removing it geometrically is view-count independent.
    Returns (new_pcd, n_dropped).
    """
    import open3d as o3d

    xyz = np.asarray(pcd.points)
    if len(xyz) == 0 or card_xy is None or len(card_xy) < 3:
        return pcd, 0
    hit = _inside_dilated_hull(xyz, card_xy, expand_m) & (np.abs(xyz[:, 2]) < z_abs)
    n = int(hit.sum())
    if n == 0:
        return pcd, 0
    keep = np.flatnonzero(~hit)
    out = pcd.select_by_index(keep.tolist())
    return out, n


def _mount_object_cloud(
    pcd_full_w,
    pcd_mask_w=None,
    n_mask_views: int = 0,
    band_z: float = 0.006,
    max_pts: int = 400000,
    seed: int = 0,
    unit: float = 1.0,
    have_card: bool = True,
    tol_m: float = 0.003,
):
    """(F1-B) Define the object cloud geometrically: the mount surface is *the plane
    the card lies on*, i.e. the z ~= 0 cluster that contains the world origin.

    Text prompts segmented the wrong thing on 3 of 4 evaluation scenes (empty on the
    ledge, the drawer front instead of the pedestal top on the cabinet), so the
    prompt must be optional.  The frame stage already puts the mount plane at z = 0
    with the card centre at the origin, which makes the surface trivially findable:

      1. full cloud (card already removed), |z| < 6 mm -> DBSCAN(eps=15 mm, 20 pts)
         -> the cluster around the origin  ==  the mount surface.
      2. object body = that surface, plus full-cloud points inside its XY hull
         dilated by 3 cm with z in [-0.6, +0.006] m that are DBSCAN-connected to it.
      3. if the SAM object mask was non-empty in >= 5 views, union in its fused
         points that lie within 2 cm of the body -- an addition, never a dependency.

    Returns (pcd_object | None, info dict).
    """
    import open3d as o3d

    info: dict = {"mount_surface_source": None, "mount_surface_pts": 0, "unit": round(unit, 4)}
    xyz = np.asarray(pcd_full_w.points)
    if len(xyz) < 500:
        info["mount_fail"] = "full_cloud_too_small"
        return None, info

    # (F2-2) `unit` is the length of one "metre" in the cloud's units; it is 1.0 for
    # metric scenes and a scene-diagonal estimate when no card gave us a scale, so the
    # same band / eps / hull constants work in `units="relative"` mode.
    # (N3) the mount-surface band and its DBSCAN radius follow the measured
    # noise: at 3.5 mm a 6 mm band is 1.7 sigma, the surface breaks into islands
    # and the origin cluster covers only part of it -- which is what shrank the
    # desk's top to 0.36 x 0.30 m on a 0.6 x 0.9 m desk. Floors are the old
    # fixed values, so a low-noise capture is bit-identical.
    band_z = max(band_z, 1.5 * float(tol_m)) * unit
    eps_band, eps_body = max(0.015, 5.0 * float(tol_m)) * unit, 0.02 * unit
    hull_dilate = _HULL_DILATE_M * unit
    z_floor, z_ceil = -0.6 * unit, max(0.006, 1.5 * float(tol_m)) * unit

    band_idx = np.flatnonzero(np.abs(xyz[:, 2]) < band_z)
    if len(band_idx) < 100:
        info["mount_fail"] = f"z_band_pts={len(band_idx)}"
        return None, info
    p_band = pcd_full_w.select_by_index(band_idx.tolist())
    lab = np.asarray(p_band.cluster_dbscan(eps=eps_band, min_points=20, print_progress=False))
    if lab.size == 0 or lab.max() < 0:
        info["mount_fail"] = "no_band_cluster"
        return None, info

    b_xy = np.asarray(p_band.points)[:, :2]
    near = np.linalg.norm(b_xy, axis=1) < 0.12 * unit   # around the card centre
    best_lab, best_score = -1, (-1, -1)
    for L in range(int(lab.max()) + 1):
        m = lab == L
        if not m.any():
            continue
        # prefer the cluster with the most points around the card centre; the card
        # itself has been cut out, so the origin sits in a hole -- size breaks ties.
        # (F2-2) with no card the origin is only the plane centroid, so rank on size.
        score = ((int(np.count_nonzero(m & near)), int(m.sum())) if have_card
                 else (int(m.sum()), int(m.sum())))
        if score > best_score:
            best_lab, best_score = L, score
    if best_lab < 0:
        info["mount_fail"] = "no_origin_cluster"
        return None, info
    surf_idx = band_idx[lab == best_lab]
    if len(surf_idx) < 200:
        info["mount_fail"] = f"mount_surface_pts={len(surf_idx)}"
        return None, info
    surf_xy = xyz[surf_idx][:, :2]
    info["mount_surface_pts"] = int(len(surf_idx))
    info["mount_surface_source"] = "geometry" if have_card else "geometry_nocard"

    # 2. body: inside the surface hull + 1.5 cm, hanging down to 0.6 m, connected
    in_hull = _inside_dilated_hull(xyz, surf_xy, hull_dilate)
    zsel = (xyz[:, 2] >= z_floor) & (xyz[:, 2] <= z_ceil)
    cand_idx = np.flatnonzero(in_hull & zsel)
    if len(cand_idx) > max_pts:
        rng = np.random.default_rng(seed)
        cand_idx = np.sort(rng.choice(cand_idx, size=max_pts, replace=False))
    surf_set = set(surf_idx.tolist())
    p_cand = pcd_full_w.select_by_index(cand_idx.tolist())
    lab2 = np.asarray(p_cand.cluster_dbscan(eps=eps_body, min_points=10, print_progress=False))
    if lab2.size and lab2.max() >= 0:
        is_surf = np.array([int(i) in surf_set for i in cand_idx])
        touching = {int(L) for L in np.unique(lab2[is_surf]) if L >= 0}
        # (F2-5) a cluster that merely sits *under* the surface (a shoe, a pedestal,
        # the floor) is not part of the object: keep only clusters that actually hang
        # from the surface, i.e. whose own TOP (z > -3 cm) comes within 2 cm in XY of
        # the surface hull.  Clusters that already contain surface points are exempt.
        cand_xyz = xyz[cand_idx]
        dropped = 0
        for L in {int(v) for v in np.unique(lab2) if v >= 0} - touching:
            sel = lab2 == L
            if not sel.any():
                continue
            cz = cand_xyz[sel]
            tops = cz[cz[:, 2] > -0.03 * unit]
            if len(tops) >= 5 and _inside_dilated_hull(
                    tops, surf_xy, 0.02 * unit).mean() > 0.25:
                touching.add(L)
            else:
                dropped += int(sel.sum())
        info["growth_pts_dropped_detached"] = dropped
        keep = np.isin(lab2, list(touching)) if touching else (lab2 >= 0)
        obj_idx = cand_idx[keep]
    else:
        obj_idx = cand_idx
    obj_idx = np.union1d(obj_idx, surf_idx)
    if len(obj_idx) > max_pts:
        rng = np.random.default_rng(seed + 1)
        obj_idx = np.sort(rng.choice(obj_idx, size=max_pts, replace=False))
    obj = pcd_full_w.select_by_index(obj_idx.tolist())
    info["object_pts_geometry"] = int(len(obj.points))

    # 3. optional union with the SAM object mask's fused points
    if pcd_mask_w is not None and n_mask_views >= 5 and len(np.asarray(pcd_mask_w.points)) >= 200:
        try:
            d = np.asarray(pcd_mask_w.compute_point_cloud_distance(obj))
            add = np.flatnonzero(d < 0.02 * unit)
            if len(add) >= 200:
                extra = pcd_mask_w.select_by_index(add.tolist())
                merged = o3d.geometry.PointCloud()
                merged.points = o3d.utility.Vector3dVector(
                    np.vstack([np.asarray(obj.points), np.asarray(extra.points)])
                )
                oc = (np.asarray(obj.colors) if obj.has_colors()
                      else np.zeros((len(obj.points), 3)))
                ec = (np.asarray(extra.colors) if extra.has_colors()
                      else np.zeros((len(extra.points), 3)))
                merged.colors = o3d.utility.Vector3dVector(np.vstack([oc, ec]))
                if len(merged.points) > max_pts:
                    merged = merged.random_down_sample(max_pts / len(merged.points))
                obj = merged
                info["mount_surface_source"] += "+mask"
                info["mask_pts_added"] = int(len(add))
        except Exception as e:  # noqa: BLE001
            info["mask_union_failed"] = f"{type(e).__name__}: {e}"
    info["object_pts"] = int(len(obj.points))
    return obj, info


def _largest_horizontal_plane(xyz, c2w, seed: int = 0, max_planes: int = 6):
    """(F2-2) No-card fallback: the largest *near-horizontal* plane of the full cloud.

    With no card there is no metric scale and no origin, so the frame has to come from
    geometry alone.  "Horizontal" is defined by the cameras: phones are held upright, so
    world-up ~= the mean of the cameras' -Y axes (OpenCV +Y is image-down).  We peel off
    planes with RANSAC (sequentially, largest first) and return the biggest whose normal
    is within 35 deg of that up vector -- i.e. a floor / table / sill rather than a wall
    or a window screen.  Returns (n, d, inlier_idx) with n pointing up, or None.
    """
    import open3d as o3d

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 500:
        return None
    c2w = np.asarray(c2w, dtype=np.float64).reshape(-1, 4, 4)
    up = -c2w[:, :3, 1].mean(axis=0) if len(c2w) else np.array([0.0, 0.0, 1.0])
    nu = np.linalg.norm(up)
    up = up / nu if nu > 1e-9 else np.array([0.0, 0.0, 1.0])

    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    thresh = 0.004 * diag / 2.5                       # ~4 mm at a 2.5 m scene diagonal
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    alive = np.arange(len(pts))
    best = None
    for _ in range(max_planes):
        if len(alive) < 300:
            break
        sub = p.select_by_index(alive.tolist())
        try:
            model, inl = sub.segment_plane(thresh, ransac_n=3, num_iterations=800)
        except Exception:  # noqa: BLE001
            break
        inl = np.asarray(inl, dtype=int)
        if len(inl) < 200:
            break
        a, b, c, d = [float(v) for v in model]
        n = np.array([a, b, c], dtype=np.float64)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            break
        n, d = n / ln, d / ln
        if float(n @ up) < 0:
            n, d = -n, -d
        idx = alive[inl]
        if float(n @ up) >= np.cos(np.radians(35.0)):
            if best is None or len(idx) > len(best[2]):
                best = (n, d, idx)
        alive = np.delete(alive, inl)
    return best


def _dilate_masks(mask_stack: np.ndarray, ksize: int = 13) -> np.ndarray:
    """Per-view binary dilation of a [N,H,W] boolean mask stack (13x13 ellipse ~= 6 px)."""
    import cv2

    if mask_stack is None or mask_stack.size == 0:
        return mask_stack
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = np.zeros_like(mask_stack, dtype=bool)
    for i in range(mask_stack.shape[0]):
        out[i] = cv2.dilate(mask_stack[i].astype(np.uint8), kernel).astype(bool)
    return out


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------
def run(
    folder,
    out_dir,
    card_prompts=(
        "card", "credit card", "membership card", "plastic card",
        # (F2-1) ledge diagnosis: SAM 3 scores the same white Costco card 0.13-0.19 for
        # "card"/"credit card" but 0.98 for "sticker", 0.97 for "label", 0.64-0.90 for
        # "white card" / 0.88-0.92 for "white rectangle".  These generic appearance
        # prompts are only reached when the card-word prompts fail (>= 3-view break).
        "white card", "sticker", "label", "white rectangle",
    ),
    table_prompt: str = "table",
    n_views: int | None = None,
    process_res: int = 512,  # (F8) DA3-LARGE-1.1 default; see surfcap/recon.py DEFAULT_PROCESS_RES
    n_max: int = 10,  # (F8) recon view cap, evenly spaced; see surfcap/recon.py DEFAULT_N_MAX
    seed: int = 0,
    debug: bool = True,
    use_exif_intrinsics: bool = True,
    thickness_m: float | None = None,
    exclude_card: bool = False,
    mesh_mode: str = "planar",
    ckpt: str | None = None,
    recon_mode: str = "sfm_tsdf",
    geom_filter: bool = True,
    geom_tol: float | None = None,
    no_detail_meshes: bool = False,
) -> Target:
    import open3d as o3d

    from . import frame as frame_mod
    from . import recon as recon_mod
    from . import scale as scale_mod
    from . import segment as segment_mod
    from .export import export_all
    from .io_images import load_folder
    from . import primitives as primitives_mod
    from .primitives import extract_surfaces

    t_start = time.time()
    out_dir = Path(out_dir)
    dbg = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    timings: dict = {}
    vram: dict = {}

    # ---------------------------------------------------------------- a. load frames
    frames: list = []
    with _stage("load", timings, warnings):
        frames, w = load_folder(folder, long_edge=1024)
        warnings += w
        if n_views is not None and 0 < n_views < len(frames):
            idx = np.linspace(0, len(frames) - 1, n_views).round().astype(int)
            frames = [frames[i] for i in sorted(set(idx.tolist()))]
        print(f"[surfcap] {len(frames)} frames, size={frames[0].size if frames else None}")
    if not frames:
        raise ValueError("zero images")

    # ---------------------------------------------------------------- b. SAM 3
    masks = None
    card_prompt_used = None
    if isinstance(card_prompts, str):
        card_prompts = (card_prompts,)
    with _stage("segment", timings, warnings):
        _vram_reset()
        model, processor = segment_mod.load_sam3()
        best = None  # (k_views, prompt, masks, warns)
        # (F1-D1) threshold ladder: SAM's 0.5 default found the card in 0/11 views on
        # the grey ledge scene.  Walk 0.5 -> 0.35 -> 0.2 and accept the first rung
        # that sees the card in >= 3 views (a lower threshold is only used when the
        # higher one failed, so table_a/desk behave exactly as before).
        try:
            tier1 = [p for p in card_prompts if p not in _CARD_PROMPTS_APPEARANCE]
            tier2 = [p for p in card_prompts if p in _CARD_PROMPTS_APPEARANCE]
            done = False
            for tier in (tier1, tier2):
                if done or not tier or (best is not None and best[0] >= 3):
                    break
                for thr in (0.5, 0.35, 0.2):
                    for cp in tier:
                        m_i, w_i = segment_mod.segment_images(
                            frames,
                            prompts={"table": table_prompt, "card": cp},
                            threshold=thr,
                            model=model,
                            processor=processor,
                        )
                        k = int(sum(bool(m_i.card[j].any())
                                    for j in range(m_i.card.shape[0])))
                        print(f"[surfcap] card prompt {cp!r} @thr={thr}: "
                              f"{k}/{len(frames)} views")
                        if best is None or k > best[0]:
                            best = (k, cp, m_i, w_i, thr)
                        if k >= 4:
                            done = True
                            break
                    if done or (best is not None and best[0] >= 3):
                        break
        finally:
            vram["segment_peak_gb"] = round(_vram_peak_gb(), 2)
            segment_mod.free_gpu(model, processor)
            del model, processor
            _free_gpu()
        masks, card_prompt_used = best[2], best[1]
        warnings += best[3]
        warnings.append(f"card_prompt_used={card_prompt_used}")
        warnings.append(f"card_sam_threshold={best[4]}")
        warnings.append(f"card_views={best[0]}")
        card_detector = "sam"
        # (F1-D3) SAM still blind -> prompt-free geometric quad detector
        # (F2-1) fire at < 3 views, not < 2: 1-2 card views give no usable scale RMS.
        if best[0] < 3:
            n_quad = 0
            quad_masks = np.zeros_like(masks.card, dtype=bool)
            for j, fr in enumerate(frames):
                qm, qinfo = segment_mod.detect_card_quad(fr.gray)
                if qm is not None and qm.shape == quad_masks[j].shape:
                    quad_masks[j] = qm
                    n_quad += 1
            print(f"[surfcap] quad card detector: {n_quad}/{len(frames)} views")
            if n_quad > best[0]:
                masks.card = quad_masks
                card_detector = "quad"
                warnings.append(f"card_views_quad={n_quad}")
        warnings.append(f"card_detector={card_detector}")
        if debug:
            from PIL import Image

            (dbg / "masks").mkdir(parents=True, exist_ok=True)
            for j, fr in enumerate(frames):
                ov = segment_mod._overlay(fr.rgb, masks.table[j], masks.card[j])
                Image.fromarray(ov).save(
                    dbg / "masks" / f"{Path(fr.path).stem}_overlay.jpg", quality=85
                )
    if masks is None:
        n, h, w_ = len(frames), frames[0].rgb.shape[0], frames[0].rgb.shape[1]
        from .types import Masks

        masks = Masks(
            np.zeros((n, h, w_), bool), np.zeros((n, h, w_), bool), [0.0] * n, [0.0] * n
        )
        warnings.append("table_mask_empty")

    # ---------------------------------------------------------------- c. DA3 recon
    recon = None
    paths_before = [f.path for f in frames]
    with _stage("recon", timings, warnings):
        _vram_reset()
        try:
            recon, w = recon_mod.run_recon(
                frames,
                n_max=n_max,
                device="cuda",
                process_res=process_res,
                use_exif_intrinsics=use_exif_intrinsics,
                **({"ckpt": ckpt} if ckpt else {}),
            )
            warnings += w
        finally:
            vram["recon_peak_gb"] = round(_vram_peak_gb(), 2)
            _free_gpu()
        print(
            f"[surfcap] recon: depth{tuple(recon.depth.shape)} "
            f"valid={int(recon.valid.sum())} px"
        )
        if debug:
            _dump(
                dbg / "cams.json",
                {
                    "paths": [Path(f.path).name for f in frames],
                    "K": recon.K,
                    "c2w": recon.c2w,
                    "process_res": process_res,
                },
            )
    if recon is None:
        raise RuntimeError("recon failed and there is no degrade path without depth")

    # run_recon may drop views; re-index the masks to match frames
    paths_after = [f.path for f in frames]
    if paths_after != paths_before:
        keep = [paths_before.index(p) for p in paths_after]
        masks.table = masks.table[keep]
        masks.card = masks.card[keep]
        warnings.append(f"masks_reindexed_after_recon={len(keep)}")

    # ------------------------------------------------- c2. SfM + TSDF (recon_mode)
    # `da3` (default) leaves everything below untouched.  `sfm_tsdf` swaps DA3's poses
    # and intrinsics for pycolmap's, re-scales every depth map onto the SfM sparse
    # points, and replaces the fused *union* cloud with a TSDF-averaged one.  Only
    # `Recon.pointmap/.valid/.c2w/.K` and the full cloud change, so scale/frame/mount/
    # primitives/export run byte-for-byte unmodified.
    tsdf_pcd = None
    tsdf_mesh = None
    recon_extra: dict = {"mode": "da3"}
    if recon_mode == "sfm_tsdf":
        with _stage("sfm", timings, warnings):
            from . import sfm as sfm_mod

            f_px = recon_mod.exif_focal_px(
                frames[0].path, frames[0].rgb.shape[1], frames[0].rgb.shape[0]
            )
            sfm_res = sfm_mod.run_sfm(
                frames, exif_f_px=f_px, work_dir=dbg / "sfm", seed=seed
            )
            if sfm_res is None:
                warnings.append("sfm_failed")
                print("[surfcap] SfM failed -> falling back to recon_mode=da3")
            else:
                warnings += sfm_res.warnings
                recon_sfm, ainfo = sfm_mod.build_sfm_recon(
                    recon, frames, sfm_res,
                    geom_filter=geom_filter, geom_tol=geom_tol,
                )
                warnings += ainfo["warnings"]
                keep = ainfo["kept"]
                if len(keep) != len(frames):
                    frames[:] = [frames[i] for i in keep]
                    masks.table = masks.table[keep]
                    masks.card = masks.card[keep]
                    masks.table_scores = [masks.table_scores[i] for i in keep]
                    masks.card_scores = [masks.card_scores[i] for i in keep]
                    warnings.append(f"sfm_registered_subset={len(keep)}")
                # (E1c) if the SfM camera carried a radial term, build_sfm_recon
                # rectified the depth/conf; put the RGB and the SAM masks through the
                # identical map so mask lookups index the same warp as the cloud.
                umaps = ainfo.get("undistort_maps")
                if umaps is not None:
                    for fr in frames:
                        fr.rgb = sfm_mod.undistort_image(fr.rgb, umaps)
                    masks.table = np.stack(
                        [sfm_mod.undistort_image(m, umaps) for m in masks.table]
                    ) if len(masks.table) else masks.table
                    masks.card = np.stack(
                        [sfm_mod.undistort_image(m, umaps) for m in masks.card]
                    ) if len(masks.card) else masks.card
                    warnings.append(f"sfm_undistorted_k1={ainfo['k1']:.4f}")
                recon = recon_sfm
                # SfM units per metre, pinned so the scene's median camera-to-surface
                # distance reads as ~0.8 m -- this is only used to size the TSDF voxel.
                unit = max(ainfo["median_depth_sfm"] / 0.8, 1e-9)
                # (E1b) build_sfm_recon already masked the depth discontinuities and
                # picked the view subset by alignment residual; use both here.
                sel = ainfo["tsdf_views"]
                depths = [ainfo["tsdf_depths"][i] for i in sel]
                tsdf_pcd, tsdf_mesh, tinfo = sfm_mod.tsdf_fuse(
                    depths,
                    [frames[i].rgb for i in sel],
                    recon.K[sel],
                    np.stack([np.linalg.inv(recon.c2w[i]) for i in sel]),
                    unit=unit,
                )
                tinfo["n_views_integrated"] = len(sel)
                tinfo["n_grid_views"] = int(ainfo["n_grid_views"])
                tinfo["n_edge_px"] = int(ainfo["n_edge_px"])
                print(
                    f"[surfcap] sfm: {sfm_res.n_registered}/{sfm_res.n_input} reg, "
                    f"reproj {sfm_res.reproj_px:.3f} px, {sfm_res.seconds:.1f}s; "
                    f"align resid affine {ainfo['align_resid_mm_affine_median']:.2f} -> "
                    f"grid {ainfo['align_resid_mm_median']:.2f} (sfm-mm, "
                    f"{ainfo['n_grid_views']}/{len(frames)} views on grid, "
                    f"{ainfo['n_fine_grid_views']} fine; cam {ainfo['camera_model']} "
                    f"k1={ainfo['k1']:.4f}), "
                    f"anchors med {ainfo['anchors_median']:.0f}; "
                    f"tsdf {len(sel)} views -> {tinfo['n_points']} pts "
                    f"(raw {tinfo['n_points_raw']}) / {tinfo['n_tris']} tris"
                )
                recon_extra = {
                    "mode": "sfm_tsdf",
                    "n_registered": int(sfm_res.n_registered),
                    "n_input": int(sfm_res.n_input),
                    "reproj_px": round(float(sfm_res.reproj_px), 4),
                    "sfm_s": round(float(sfm_res.seconds), 2),
                    "align_resid_mm_median": round(
                        float(ainfo["align_resid_mm_median"]), 4
                    ),
                    "align_resid_mm_affine_median": round(
                        float(ainfo["align_resid_mm_affine_median"]), 4
                    ),
                    "n_grid_views": int(ainfo["n_grid_views"]),
                    "n_fine_grid_views": int(ainfo["n_fine_grid_views"]),
                    "camera_model": ainfo["camera_model"],
                    "k1": round(float(ainfo["k1"]), 6),
                    "anchors_median": ainfo["anchors_median"],
                    "geom_consistency": {
                        k: (round(float(v), 4) if isinstance(v, float) else v)
                        for k, v in ainfo["geom_consistency"].items()
                        if k != "kept_frac"
                    },
                    "geom_kept_frac": [
                        round(float(x), 4)
                        for x in ainfo["geom_consistency"].get("kept_frac", [])
                    ],
                    "tsdf": tinfo,
                }
                if debug:
                    _dump(dbg / "sfm.json", {
                        "sfm": {k: v for k, v in recon_extra.items() if k != "tsdf"},
                        "tsdf": tinfo,
                        "per_view": [
                            {k: v for k, v in s_.items()
                             if k not in ("a_nodes", "b_nodes")}
                            for s_ in ainfo["per_view"]
                        ],
                    })
                    try:
                        o3d.io.write_triangle_mesh(str(dbg / "tsdf_mesh.ply"), tsdf_mesh)
                        from .postprocess import render_mesh_png as _rmp

                        _rmp(tsdf_mesh, dbg / "tsdf_mesh_iso.png")
                        print(f"[surfcap] wrote {dbg / 'tsdf_mesh.ply'} + tsdf_mesh_iso.png")
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"tsdf_debug_failed={exc}")
                # primitives.py re-estimates normals only when the cloud has none, and
                # the frame stage rotates the points without rotating stored normals --
                # so hand the pipeline a normal-free cloud.
                tsdf_pcd.normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                if len(tsdf_pcd.points) > 800000:
                    tsdf_pcd = tsdf_pcd.random_down_sample(800000 / len(tsdf_pcd.points))
                    warnings.append(f"tsdf_downsampled={len(tsdf_pcd.points)}")

    # ---------------------------------------------------------------- d. scale
    sres = None
    units = "relative"
    s = 1.0
    with _stage("scale", timings, warnings):
        grays = [f.gray for f in frames]
        cam_centres_r = recon.c2w[:, :3, 3]
        sres, w = scale_mod.estimate_scale(
            masks.card,
            recon.pointmap,
            recon.valid,
            grays,
            card_mm=CARD_MM,
            cam_centres=cam_centres_r,
        )
        warnings += w
        print(
            f"[surfcap] scale={sres.scale:.6g} rms_mm={sres.rms_mm:.3f} "
            f"n_views={sres.n_views} reliable={sres.reliable}"
        )
        for k, v in sorted(sres.per_view.items()):
            print(f"[surfcap]   view {k}: {v if not isinstance(v, dict) else {kk: (round(vv,4) if isinstance(vv,float) else vv) for kk, vv in v.items() if kk in ('scale','resid_mm','area_px','rejected','kept','weight','why')}}")
        _dump(dbg / "scale.json", {
            "scale": sres.scale, "rms_mm": sres.rms_mm, "n_views": sres.n_views,
            "reliable": sres.reliable, "per_view": sres.per_view,
            "card_corners_recon": sres.card_corners_w,
        })
        if sres.n_views >= 1 and np.isfinite(sres.scale) and sres.scale > 0:
            s = float(sres.scale)
            units = "m"
        if sres.n_views < 3:
            warnings.append(f"scale_unreliable_n_views={sres.n_views}")
            if sres.n_views == 0:
                s, units = 1.0, "relative"
        if np.isfinite(sres.rms_mm) and sres.rms_mm > 3.0 and not any(
            w.startswith("scale_rms_mm") for w in warnings
        ):
            warnings.append(f"scale_rms_mm={sres.rms_mm:.2f}")
    if sres is None:
        from .types import ScaleResult

        sres = ScaleResult(1.0, float("inf"), 0, False, None, {})
        units, s = "relative", 1.0

    # ---------------------------------------------------------------- e. fuse
    pcd_table_r = None
    pcd_full_r = None
    pcd_full_excl_r = None
    card_dilated = None
    with _stage("fuse", timings, warnings):
        if exclude_card:
            card_dilated = _dilate_masks(masks.card, ksize=13)
            mask_table_excl = masks.table & ~card_dilated
            n_card_excluded_px = int(np.sum(masks.table & card_dilated & recon.valid))
        else:
            card_dilated = np.zeros_like(masks.card, dtype=bool)
            mask_table_excl = masks.table
            n_card_excluded_px = 0
        pcd_table_r, _ = recon_mod.fuse_views(recon, frames, mask=mask_table_excl, seed=seed)
        if tsdf_pcd is not None:
            # the TSDF volume already *averaged* every view; concatenating the point maps
            # on top of it would just re-add the per-view noise we paid SfM to remove.
            pcd_full_r = tsdf_pcd
            warnings.append(f"full_cloud_from_tsdf={len(pcd_full_r.points)}")
        else:
            pcd_full_r, _ = recon_mod.fuse_views(recon, frames, mask=None, seed=seed)
        if exclude_card and tsdf_pcd is None:
            pcd_full_excl_r, _ = recon_mod.fuse_views(recon, frames, mask=~card_dilated, seed=seed)
        else:
            # in sfm_tsdf mode the card is removed geometrically in the world frame
            # (stage f1b), which is view-count independent anyway
            pcd_full_excl_r = pcd_full_r
        warnings.append(f"card_excluded_from_object={exclude_card}")
        if exclude_card:
            warnings.append(f"card_points_excluded={n_card_excluded_px}")
        print(
            f"[surfcap] table cloud {len(pcd_table_r.points)} pts "
            f"(card_points_excluded={n_card_excluded_px}), "
            f"full cloud {len(pcd_full_r.points)} pts"
        )
        if len(pcd_table_r.points) < 500:
            warnings.append("table_mask_empty")
            # degrade: biggest plane of the full cloud is the table
            pcd_table_r = pcd_full_r
    cam_centres_r = recon.c2w[:, :3, 3]

    # ---------------------------------------------------------------- f. frame
    T = np.eye(4)
    pcd_table_w = None
    pcd_full_w = None
    pcd_full_excl_w = None
    c2w_w = recon.c2w.copy()
    card_w = None
    with _stage("frame", timings, warnings):
        table_xyz = np.asarray(pcd_table_r.points)
        n_pl = d_pl = inl = None
        if sres.card_corners_w is None:
            # (F2-2) scale-free fallback: frame from the largest near-horizontal plane
            # of the FULL cloud, not from whatever the text prompt happened to segment.
            hp = _largest_horizontal_plane(
                np.asarray(pcd_full_r.points), recon.c2w, seed=seed
            )
            if hp is not None:
                table_xyz = np.asarray(pcd_full_r.points)
                n_pl, d_pl, inl = hp
                warnings.append(f"frame_from_horizontal_plane_pts={len(inl)}")
                print(f"[surfcap] no card -> horizontal plane frame, {len(inl)} inliers")
            else:
                warnings.append("frame_horizontal_plane_failed")
        if n_pl is None:
            n_pl, d_pl, inl, w = frame_mod.fit_table_plane(
                table_xyz, sres.card_corners_w, cam_centres_r, s, seed=seed
            )
            warnings += w
        T, w = frame_mod.build_world_transform(
            n_pl, d_pl, sres.card_corners_w, cam_centres_r, s,
            table_xyz=table_xyz[inl] if inl is not None and len(inl) else table_xyz,
        )
        warnings += w
        if units == "relative":
            warnings.append("units_relative_scale_unreliable")
        pcd_table_w = o3d.geometry.PointCloud(
            pcd_table_r if len(table_xyz) == len(pcd_table_r.points) else pcd_full_r
        )
        pcd_table_w.points = o3d.utility.Vector3dVector(frame_mod.apply_sim3(T, table_xyz))
        pcd_full_w = o3d.geometry.PointCloud(pcd_full_r)
        pcd_full_w.points = o3d.utility.Vector3dVector(
            frame_mod.apply_sim3(T, np.asarray(pcd_full_r.points))
        )
        pcd_full_excl_w = o3d.geometry.PointCloud(pcd_full_excl_r)
        pcd_full_excl_w.points = o3d.utility.Vector3dVector(
            frame_mod.apply_sim3(T, np.asarray(pcd_full_excl_r.points))
        )
        c2w_w = frame_mod.apply_sim3_to_c2w(T, recon.c2w)
        if sres.card_corners_w is not None:
            card_w = frame_mod.apply_sim3(T, sres.card_corners_w)
        _dump(dbg / "frame.json", {
            "T_world_from_recon": T, "units": units, "scale": s,
            "plane_n": n_pl, "plane_d": d_pl, "n_inliers": int(len(inl)),
            "card_corners_world": card_w,
            "cam_centres_world": c2w_w[:, :3, 3],
        })
    if pcd_table_w is None:
        pcd_table_w = pcd_table_r
        pcd_full_w = pcd_full_r
        pcd_full_excl_w = pcd_full_excl_r
    cam_centres_w = np.asarray(c2w_w)[:, :3, 3]

    if debug:
        try:
            o3d.io.write_point_cloud(str(dbg / "table_world.ply"), pcd_table_w)
            ctx = pcd_full_w
            if len(ctx.points) > 300000:
                ctx = ctx.random_down_sample(300000 / len(ctx.points))
            o3d.io.write_point_cloud(str(dbg / "full_world.ply"), ctx)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"stage_failed=debug_ply:{type(e).__name__}: {e}")

    # ------------------------------------- f1a. (N3) measure the capture noise once
    # The mount surface around the card is the one patch of geometry we know is
    # flat, so its residual *is* this capture's fusion noise. Every tolerance
    # downstream (mount band, top seed band, patch detection, snap band, hull
    # exclusion) is derived from it instead of being fixed at 3 mm, which is what
    # made the desk capture (3.5-4 mm noise) explode into 122k "detail" points.
    noise_mm = None
    tol_m = primitives_mod.tol_from_noise_mm(None)
    if units == "m":
        for src in (pcd_table_w, pcd_full_excl_w):
            if src is None or len(src.points) < 500:
                continue
            try:
                noise_mm = primitives_mod.noise_mm_from_mount(np.asarray(src.points))
            except Exception as e:  # noqa: BLE001
                warnings.append(f"noise_measure_failed={type(e).__name__}")
            if noise_mm is not None:
                break
        tol_m = primitives_mod.tol_from_noise_mm(noise_mm)
    if noise_mm is not None:
        warnings.append(f"noise_mm={noise_mm:.2f}")
    warnings.append(f"plane_tol_mm={tol_m * 1000.0:.2f}")
    print(f"[surfcap] noise: {noise_mm} mm -> plane tol {tol_m * 1000.0:.2f} mm")

    # ------------------------------------- f1b. geometric card removal + mount surface
    mount_ok = False
    mount_info: dict = {}
    with _stage("mount", timings, warnings):
        card_xy = np.asarray(card_w)[:, :2] if card_w is not None else None
        # (F2-2) length of one "metre" in the cloud's units: 1.0 when the card gave us
        # a metric scale, else a scene-diagonal estimate so the geometric mount path
        # still works with units="relative" (typical capture diagonal ~ 2.5 m).
        unit = 1.0
        if units != "m":
            _fx = np.asarray(pcd_full_excl_w.points) if pcd_full_excl_w is not None else None
            if _fx is not None and len(_fx) >= 100:
                lo = np.percentile(_fx, 1, axis=0)
                hi = np.percentile(_fx, 99, axis=0)
                unit = max(float(np.linalg.norm(hi - lo)) / 2.5, 1e-6)
            warnings.append(f"mount_unit_relative={unit:.4f}")
        # (A) delete the card from every cloud, regardless of how many views saw it
        # (F4) ... but only when the caller actually asked for it: removing the card
        # leaves a large hole in the mount surface, and the flatten step puts the
        # card's points on the top plane anyway.
        if card_xy is not None and exclude_card:
            pcd_table_w, n_a = _drop_card_geometric(pcd_table_w, card_xy)
            pcd_full_excl_w, n_b = _drop_card_geometric(pcd_full_excl_w, card_xy)
            warnings.append(f"card_points_dropped_geometric={n_a}+{n_b}")
            print(f"[surfcap] geometric card removal: table -{n_a}, full -{n_b} pts")
        # (B) mount surface = the z~=0 cluster around the origin, prompts optional
        if pcd_full_excl_w is not None:
            src = pcd_full_excl_w
            if len(src.points) > 600000:
                src = src.voxel_down_sample(0.002)
            n_mask_views = int(sum(bool(masks.table[j].any())
                                   for j in range(masks.table.shape[0])))
            obj, mount_info = _mount_object_cloud(
                src,
                pcd_mask_w=pcd_table_w if len(pcd_table_w.points) >= 500 else None,
                n_mask_views=n_mask_views,
                seed=seed,
                unit=unit,
                have_card=card_xy is not None,
                tol_m=tol_m,
            )
            if obj is not None and len(obj.points) >= 500:
                # (F4) only strip the card when asked; kept by default so the mount
                # surface has no card-shaped hole in it.
                if card_xy is not None and exclude_card:
                    obj, n_c = _drop_card_geometric(obj, card_xy)
                else:
                    n_c = 0
                pcd_table_w = obj
                mount_ok = True
                warnings.append(
                    f"mount_surface_source={mount_info.get('mount_surface_source')}"
                )
                warnings.append(f"mount_surface_pts={mount_info.get('mount_surface_pts')}")
                warnings.append(f"mount_object_pts={len(obj.points)}")
            else:
                warnings.append(f"mount_geometry_failed={mount_info.get('mount_fail')}")
        print(f"[surfcap] mount: {mount_info}")
        if debug:
            _dump(dbg / "mount.json", mount_info)



    # ------------------------------------------------------- f2. apron / front rail
    n_apron = 0
    pcd_table_base = None
    with _stage("apron", timings, warnings):
        if mount_ok:
            warnings.append("apron_skipped_mount_geometry")
        elif units == "m" and pcd_full_excl_w is not None and len(pcd_table_w.points) >= 50:
            t_xyz = np.asarray(pcd_table_w.points)
            f_xyz = np.asarray(pcd_full_excl_w.points)
            f_rgb = (
                np.asarray(pcd_full_excl_w.colors)
                if pcd_full_excl_w.has_colors()
                else np.zeros((0, 3))
            )
            a_xyz, a_rgb = _apron_points(t_xyz, f_xyz, f_rgb, seed=seed)
            n_apron = len(a_xyz)
            print(f"[surfcap] apron candidates: {n_apron}")
            if n_apron >= 300:
                t_rgb = (
                    np.asarray(pcd_table_w.colors)
                    if pcd_table_w.has_colors()
                    else np.zeros((len(t_xyz), 3))
                )
                if len(a_rgb) != n_apron:
                    a_rgb = np.zeros((n_apron, 3))
                merged = o3d.geometry.PointCloud()
                merged.points = o3d.utility.Vector3dVector(np.vstack([t_xyz, a_xyz]))
                merged.colors = o3d.utility.Vector3dVector(np.vstack([t_rgb, a_rgb]))
                pcd_table_base = pcd_table_w
                pcd_table_w = merged
                warnings.append(f"apron_points_added={n_apron}")
            elif n_apron:
                warnings.append(f"apron_too_small={n_apron}_skipped")

    # ---------------------------------------------------------------- g. cleanup
    def _cleanup(src):
        pts = np.asarray(src.points)
        if units == "m":
            voxel, eps = 0.003, 0.03
        else:
            diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 1.0
            voxel, eps = 0.004 * diag, 0.04 * diag
        p = src.voxel_down_sample(voxel)
        n0 = len(p.points)
        p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        n1 = len(p.points)
        labels = np.asarray(p.cluster_dbscan(eps=eps, min_points=30, print_progress=False))
        if labels.size and labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            keep = np.flatnonzero(labels == int(np.argmax(counts)))
            if keep.size >= 200:
                p = p.select_by_index(keep.tolist())
        print(f"[surfcap] cleanup {len(src.points)} -> voxel {n0} -> sor {n1} -> dbscan {len(p.points)}")
        return p if len(p.points) >= 200 else src

    pcd_clean = pcd_table_w
    with _stage("cleanup", timings, warnings):
        if units != "m":
            warnings.append("cleanup_relative_voxel")
        pcd_clean = _cleanup(pcd_table_w)
        if pcd_clean is pcd_table_w:
            warnings.append("cleanup_kept_raw_cloud")

    # ---------------------------------------------------------------- h. primitives
    surfaces: list = []
    obb = None
    with _stage("primitives", timings, warnings):
        surfaces, obb, w = extract_surfaces(
            pcd_clean, cam_centres_w, seed=seed, seed_mount=True, tol_m=tol_m
        )
        warnings += w
        if n_apron and pcd_table_base is not None:
            # an apron-born vertical plane that is tiny or non-planar is noise, not a face
            bad = [
                sf for sf in surfaces
                if sf.role in ("front", "side")
                and (sf.n_points < 300 or sf.planarity_rms_mm > 6.0)
            ]
            if bad:
                surfaces = [sf for sf in surfaces if sf not in bad]
                warnings.append(f"dropped_garbage_vertical={len(bad)}")
            # If the apron bought us no usable vertical face it only inflated the top's
            # extent with rail points, so fall back to the table-mask-only cloud.
            if not any(sf.role in ("front", "side", "bottom") for sf in surfaces):
                warnings.append("apron_no_vertical_face_reverted")
                pcd_clean = _cleanup(pcd_table_base)
                surfaces, obb, w2 = extract_surfaces(
                    pcd_clean, cam_centres_w, seed=seed, tol_m=tol_m
                )
                warnings += w2
        for sf in surfaces:
            print(
                f"[surfcap]   {sf.id:<10} {sf.role:<6} n={np.round(sf.normal,3).tolist()} "
                f"ext={np.round(sf.extent_m,4).tolist()} rms={sf.planarity_rms_mm:.2f}mm "
                f"n_pts={sf.n_points}"
            )

    # ---------------------------------------------------------------- i. export
    target = None
    with _stage("export", timings, warnings):
        frame_dict = frame_mod.build_frame_dict(T, units)
        scale_dict = {
            "reference": "ISO ID-1 card 85.60x53.98 mm",
            "factor": float(s),
            "rms_mm": None if not np.isfinite(sres.rms_mm) else float(sres.rms_mm),
            "n_views_with_ref": int(sres.n_views),
            "reliable": bool(sres.reliable and units == "m"),
            "card_prompt_used": card_prompt_used,
        }
        target = export_all(
            out_dir, pcd_clean, surfaces, obb, frame_dict, scale_dict, warnings,
            context={"floor_plane": None},
        )
        # (F8) record the checkpoint/resolution/view-count that actually ran (may differ
        # from the request if the OOM ladder fell back -- see recon_ckpt_fallback warning).
        recon_meta = getattr(recon, "meta", None) or {}
        target.cloud["noise_mm"] = (round(float(noise_mm), 3)
                                    if noise_mm is not None else None)
        target.cloud["plane_tol_mm"] = round(tol_m * 1000.0, 3)
        target.cloud["recon"] = {
            "ckpt": recon_meta.get("ckpt", ckpt or recon_mod.DEFAULT_CKPT),
            "process_res": recon_meta.get("process_res", process_res),
            "n_views": recon_meta.get("n_views", len(frames)),
            **recon_extra,
        }

    if target is not None:
        with _stage("postprocess", timings, warnings):
            from .postprocess import postprocess as _postprocess

            from .postprocess import render_mesh_png as _render_mesh_png

            pp = _postprocess(
                pcd_clean, surfaces, out_dir, obb=obb, thickness_m=thickness_m,
                mesh_mode=mesh_mode,
            )
            mesh_obj = pp.pop("_mesh", None)
            target.cloud["clean_ply"] = pp["clean_ply"]
            target.cloud["mesh_ply"] = pp["mesh_ply"]
            target.cloud["mesh_glb"] = pp["mesh_glb"]
            if pp.get("patches_glb"):
                target.cloud["mesh_patches_glb"] = pp["patches_glb"]
            target.cloud["postprocess"] = pp
            # (F4) the iso render is part of every run, not just --gt-check
            try:
                if mesh_obj is None:
                    import open3d as _o3d

                    mesh_obj = _o3d.io.read_triangle_mesh(str(out_dir / pp["mesh_ply"]))
                _render_mesh_png(mesh_obj, dbg / "mesh_iso.png")
                print(f"[surfcap] wrote {dbg / 'mesh_iso.png'}")
            except Exception as exc:
                warnings.append(f"mesh_render_failed={exc}")

        # (F10) target.json must be on disk *before* detail_mesh/hybrid_mesh run --
        # both read obb/frame back off it -- then re-saved once their stats land.
        target.warnings = warnings
        target.save(Path(out_dir) / "target.json")

        tsdf_mesh_ply = dbg / "tsdf_mesh.ply"
        if (
            not no_detail_meshes
            and recon_mode == "sfm_tsdf"
            and tsdf_mesh is not None
            and tsdf_mesh_ply.exists()
        ):
            # (F10) debug/tsdf_mesh.ply is written in the *recon* frame, in the
            # "sfm" stage above, before the sim3 world transform `T` is even
            # computed -- detail_mesh.py/hybrid_mesh.py's `_tsdf_mesh_to_world`
            # then has to *guess* the frame by checking which of {identity, T}
            # puts more vertices in the OBB. `T` is known here (the "frame"
            # stage has since run), so transform
            # the in-memory tsdf_mesh to world space ourselves and hand both
            # modules an already-world-frame file -- no auto-detect needed.
            tsdf_mesh_world = o3d.geometry.TriangleMesh(tsdf_mesh)
            v_recon = np.asarray(tsdf_mesh.vertices, dtype=np.float64)
            tsdf_mesh_world.vertices = o3d.utility.Vector3dVector(
                v_recon @ T[:3, :3].T + T[:3, 3]
            )
            tsdf_mesh_world_ply = dbg / "tsdf_mesh_world.ply"
            o3d.io.write_triangle_mesh(str(tsdf_mesh_world_ply), tsdf_mesh_world)
            tsdf_mesh_ply = tsdf_mesh_world_ply

            planar_solid_path = Path(out_dir) / pp["mesh_ply"]
            # cabinet-sized TSDF clouds push hybrid_mesh past ~120 s; hand it a
            # coarser voxel so its cropped/downsampled cloud stays near 150k pts.
            n_tsdf_pts = len(tsdf_pcd.points) if tsdf_pcd is not None else 0
            hybrid_voxel_m = 0.002 if n_tsdf_pts <= 300_000 else 0.003

            with _stage("detail_mesh", timings, warnings):
                from . import detail_mesh as detail_mesh_mod

                dstats = detail_mesh_mod.detail_mesh(
                    tsdf_mesh_ply, Path(out_dir) / "target.json", out_dir,
                )
                rms_psr = dstats["psr"]["rms_vs_planar_mm"]
                if rms_psr is None:
                    # detail_mesh.py guesses the planar-solid path from the tsdf
                    # mesh's grandparent dir name (its own `out/e1/sfm_<scene>`
                    # layout); our `out/<scene>/debug/` layout guesses wrong, so
                    # recompute against the planar solid we already know about.
                    try:
                        psr_reloaded = o3d.io.read_triangle_mesh(dstats["psr"]["out_ply"])
                        rms_psr = detail_mesh_mod._rms_to_planar_solid(
                            psr_reloaded, planar_solid_path
                        )
                    except Exception:
                        pass
                target.cloud["detail_mesh"] = {
                    "tris_out": dstats["psr"]["tris_out"],
                    "watertight": dstats["psr"]["watertight"],
                    "rms_vs_planar_mm": rms_psr,
                    "runtime_s": dstats["runtime_s"],
                    "out_ply": dstats["psr"]["out_ply"],
                    "out_glb": dstats["psr"]["out_glb"],
                }

            with _stage("hybrid_mesh", timings, warnings):
                from . import hybrid_mesh as hybrid_mesh_mod

                hstats = hybrid_mesh_mod.hybrid_mesh(
                    tsdf_mesh_ply, Path(out_dir) / "target.json", planar_solid_path,
                    out_dir, voxel_m=hybrid_voxel_m, tol_m=tol_m, noise_mm=noise_mm,
                )
                target.cloud["hybrid_mesh"] = {
                    "tris_out": hstats["tris_out"],
                    "closed": hstats["closed_no_boundary"],
                    "n_patches": hstats["n_patches"],
                    "n_hull_patches": hstats["n_hull_patches"],
                    "n_snapped": hstats.get("n_snapped"),
                    "n_detail": hstats.get("n_detail"),
                    "tol_mm": hstats.get("tol_mm"),
                    "detail_band_before": hstats["detail_band_before"],
                    "detail_band_after": hstats["detail_band_after"],
                    "rms_vs_planar_solid_observed_mm": hstats["rms_vs_planar_solid_observed_mm"],
                    "runtime_s": hstats["runtime_s"],
                    "out_ply": hstats["out_ply"],
                    "out_glb": hstats["out_glb"],
                }

            target.warnings = warnings
            target.save(Path(out_dir) / "target.json")

    timings["total"] = round(time.time() - t_start, 3)
    _dump(dbg / "timings.json", {"timings_s": timings, "vram_gb": vram})
    print(f"[surfcap] TOTAL {timings['total']:.2f}s  vram={vram}")
    if target is None:
        target = Target(
            frame=frame_mod.build_frame_dict(T, units),
            scale={"factor": s, "reliable": False},
            cloud={"ply": "target.ply", "n_points": len(pcd_clean.points)},
            surfaces=surfaces, obb=obb, context={"floor_plane": None}, warnings=warnings,
        )
    return target


# --------------------------------------------------------------------------------------
# ground-truth check + debug renders (GPU-free, run after `run`)
# --------------------------------------------------------------------------------------
_GT_DEFAULT = {
    "card_long_mm": CARD_MM[0],
    "top_extent_m": (0.60, 0.35),
    "thickness_mm": (15.0, 20.0),
    "card_to_front_mm": 13.0 + CARD_MM[1] / 2.0,
    "table_height_m": 0.75,
}


def _read_notes(scene_dir) -> dict:
    """(F1-E) Parse data/<scene>/notes.txt into {key: float} (first number per line).

    Only table_a was ever measured with a ruler, so every other scene must print
    measured values without inventing a ground truth.
    """
    out: dict = {}
    if scene_dir is None:
        return out
    path = Path(scene_dir) / "notes.txt"
    if not path.exists():
        return out
    import re

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        m = re.search(r"[-+]?\d*\.?\d+", val)
        if m:
            try:
                out[key.strip().lower()] = float(m.group(0))
            except ValueError:
                pass
    return out


def gt_check(out_dir, write: bool = True, scene_dir=None) -> str:
    """Compare target.json against ruler ground truth from `data/<scene>/notes.txt`.

    With no notes.txt the measured values are still printed (verdict ``n/a``); the
    card-edge check against the ISO ID-1 reference is always available.
    """
    import open3d as o3d

    out_dir = Path(out_dir)
    dbg = out_dir / "debug"
    tgt = Target.load(out_dir / "target.json")
    with open(dbg / "frame.json") as f:
        fr = json.load(f)

    notes = _read_notes(scene_dir)
    GT = dict(_GT_DEFAULT)
    if notes:
        if "top_length_mm" in notes and "top_width_mm" in notes:
            GT["top_extent_m"] = (notes["top_length_mm"] / 1000.0,
                                  notes["top_width_mm"] / 1000.0)
        if "top_thickness_mm" in notes:
            t = notes["top_thickness_mm"]
            GT["thickness_mm"] = (t - 2.5, t + 2.5)
        if "table_height_mm" in notes:
            GT["table_height_m"] = notes["table_height_mm"] / 1000.0
    else:
        GT["top_extent_m"] = None
        GT["thickness_mm"] = None
        GT["card_to_front_mm"] = None
        GT["table_height_m"] = None

    rows = []

    def row(name, measured, expect, ok):
        if ok is None or expect is None:
            rows.append((name, measured, "(no notes.txt)", "n/a"))
        else:
            rows.append((name, measured, expect, "PASS" if ok else "FAIL"))

    # (i) card long edge
    cc = fr.get("card_corners_world")
    if cc:
        c = np.asarray(cc, float)
        longs = [np.linalg.norm(c[0] - c[1]) * 1000, np.linalg.norm(c[2] - c[3]) * 1000]
        shorts = [np.linalg.norm(c[1] - c[2]) * 1000, np.linalg.norm(c[3] - c[0]) * 1000]
        ml = float(np.mean(longs))
        row("card long edge", f"{ml:.1f} mm (edges {longs[0]:.1f}/{longs[1]:.1f})",
            f"{GT['card_long_mm']:.2f} +/- 1.5 mm", abs(ml - GT["card_long_mm"]) <= 1.5)
        ms = float(np.mean(shorts))
        row("card short edge", f"{ms:.1f} mm", f"{CARD_MM[1]:.2f} +/- 1.5 mm",
            abs(ms - CARD_MM[1]) <= 1.5)
    else:
        row("card long edge", "no card corners", f"{GT['card_long_mm']} mm", False)

    tops = [s for s in tgt.surfaces if s.role == "top"]
    fronts = [s for s in tgt.surfaces if s.role == "front"]
    bottoms = [s for s in tgt.surfaces if s.role == "bottom"]
    top = max(tops, key=lambda s: s.n_points) if tops else None

    # (ii) top normal vs +Z
    if top is not None:
        n = np.asarray(top.normal, float)
        ang = float(np.degrees(np.arccos(np.clip(abs(n[2]) / np.linalg.norm(n), -1, 1))))
        row("top normal vs +Z", f"{ang:.2f} deg", "< 2 deg", ang < 2.0)
        # (iii) extents
        e = sorted(top.extent_m, reverse=True)
        te = GT["top_extent_m"]
        row("top extent", f"{e[0]:.3f} x {e[1]:.3f} m",
            None if te is None else f"{te[0]:.3f} x {te[1]:.3f} m",
            None if te is None else (abs(e[0] - te[0]) < 0.05 and abs(e[1] - te[1]) < 0.05))
    else:
        row("top normal vs +Z", "no top plane", "< 2 deg", False)
        te = GT["top_extent_m"]
        row("top extent", "no top plane",
            None if te is None else f"{te[0]:.3f} x {te[1]:.3f} m",
            None if te is None else False)

    # (iv) thickness
    pcd = o3d.io.read_point_cloud(str(out_dir / "target.ply"))
    pts = np.asarray(pcd.points)
    if top is not None and bottoms:
        bot = max(bottoms, key=lambda s: s.n_points)
        th = abs(float(top.centroid[2]) - float(bot.centroid[2])) * 1000
        src = "top-bottom centroid dz"
    elif fronts:
        f0 = max(fronts, key=lambda s: s.n_points)
        p3 = np.asarray(f0.polygon_3d, float)
        th = float(p3[:, 2].max() - p3[:, 2].min()) * 1000 if len(p3) else float("nan")
        src = "front plane z-range"
    else:
        th, src = float("nan"), "no bottom/front plane in mask"
    tk = GT["thickness_mm"]
    row(
        f"top thickness ({src})",
        "not measurable" if not np.isfinite(th) else f"{th:.1f} mm",
        None if tk is None else f"{tk[0]:.0f}-{tk[1]:.0f} mm",
        None if tk is None else (np.isfinite(th) and tk[0] <= th <= tk[1]),
    )

    # (v) origin (card centre) to front plane
    if fronts:
        f0 = max(fronts, key=lambda s: s.n_points)
        n = np.asarray(f0.normal, float)
        n = n / np.linalg.norm(n)
        dist = abs(float(np.dot(np.asarray(f0.centroid, float), n))) * 1000
        cf = GT["card_to_front_mm"]
        row("card centre -> front plane", f"{dist:.1f} mm",
            None if cf is None else f"~{cf:.1f} mm",
            None if cf is None else abs(dist - cf) < 10)
    elif len(pts):
        # no vertical face was captured: fall back to the cloud's own front
        # boundary (the table edge nearer the card, i.e. the smaller |y| side)
        lo = abs(float(np.percentile(pts[:, 1], 0.5)))
        hi = abs(float(np.percentile(pts[:, 1], 99.5)))
        dist = min(lo, hi) * 1000
        cf = GT["card_to_front_mm"]
        row("card centre -> front edge (cloud)", f"{dist:.1f} mm",
            None if cf is None else f"~{cf:.1f} mm",
            None if cf is None else abs(dist - cf) < 10)
    else:
        cf = GT["card_to_front_mm"]
        row("card centre -> front plane", "no front plane",
            None if cf is None else f"~{cf:.1f} mm", None if cf is None else False)

    # (vi) floor -> top height from the context cloud
    hz = "n/a"
    ok = True
    th_gt = GT["table_height_m"]
    ctx = dbg / "full_world.ply"
    if ctx.exists():
        fp = np.asarray(o3d.io.read_point_cloud(str(ctx)).points)
        if len(fp):
            zf = np.percentile(fp[:, 2], 0.5)
            h = -float(zf)
            hz = f"{h:.3f} m"
            ok = None if th_gt is None else abs(h - th_gt) < 0.08
    row("floor -> top height (optional)", hz,
        None if th_gt is None else f"~{th_gt:.3f} m",
        None if th_gt is None else ok)

    w = max(len(r[0]) for r in rows) + 2
    lines = [f"{'check':<{w}}{'measured':<38}{'expected':<28}{'verdict'}"]
    lines.append("-" * (w + 38 + 28 + 8))
    for name, m, e, v in rows:
        lines.append(f"{name:<{w}}{str(m):<38}{str(e):<28}{v}")
    txt = "\n".join(lines)
    if write:
        (dbg / "gt_check.txt").write_text(txt + "\n")
    return txt


def render_views(out_dir) -> list:
    """Top-down (XY) and side (XZ) matplotlib scatters of the cleaned world cloud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    out_dir = Path(out_dir)
    dbg = out_dir / "debug"
    tgt = Target.load(out_dir / "target.json")
    pcd = o3d.io.read_point_cloud(str(out_dir / "target.ply"))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    card = None
    fj = dbg / "frame.json"
    if fj.exists():
        with open(fj) as f:
            cc = json.load(f).get("card_corners_world")
        if cc:
            card = np.asarray(cc, float)

    made = []
    for tag, (ax0, ax1, lab) in {
        "top": (0, 1, ("X (m)", "Y (m)")),
        "side": (0, 2, ("X (m)", "Z (m)")),
    }.items():
        fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
        ax.scatter(pts[:, ax0], pts[:, ax1], s=0.6,
                   c=cols if cols is not None and len(cols) == len(pts) else "0.4",
                   linewidths=0)
        if card is not None:
            loop = np.vstack([card, card[:1]])
            ax.plot(loop[:, ax0], loop[:, ax1], "-o", color="red", ms=4, lw=1.6,
                    label="card corners")
        colours = {"top": "green", "front": "blue", "side": "orange",
                   "bottom": "purple", "other": "grey"}
        for s in tgt.surfaces:
            c = np.asarray(s.centroid, float)
            ax.scatter([c[ax0]], [c[ax1]], s=70, marker="x",
                       color=colours.get(s.role, "grey"), label=s.id)
        ax.scatter([0], [0], s=90, marker="+", color="black", label="origin")
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_title(f"surfcap world frame - {tag} ({lab[0][0]}{lab[1][0]})")
        ax.legend(fontsize=6, loc="best", markerscale=0.8)
        p = dbg / f"world_{tag}.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        made.append(p)
    return made
