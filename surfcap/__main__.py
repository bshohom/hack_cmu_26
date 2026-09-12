"""CLI: python -m surfcap <folder> --out <dir> [--views N] [--res 768] [--card-prompt ...]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m surfcap")
    ap.add_argument("folder", help="folder of images (jpg/heic/png)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--views", type=int, default=None, help="cap on number of views")
    ap.add_argument("--res", type=int, default=512,
                    help="DA3 process_res (default 512: DA3-LARGE-1.1 OOMs at 768+ on a "
                         "10 GB GPU with 10 views; see README Known limits)")
    ap.add_argument("--n-max", type=int, default=10,
                    help="recon view cap, evenly spaced subset when more are loaded")
    ap.add_argument(
        "--card-prompt", action="append", default=None,
        help="card text prompt (repeatable; tried in order until >=4 views)",
    )
    ap.add_argument("--table-prompt", default="table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--no-exif-intrinsics", action="store_true",
                    help="let DA3 predict intrinsics instead of deriving them from EXIF")
    ap.add_argument("--gt-check", action="store_true",
                    help="also run the ground-truth check + world renders")
    ap.add_argument("--thickness", type=float, default=None,
                    help="override slab thickness (m); default derives from OBB")
    ap.add_argument("--exclude-card", action="store_true",
                    help="drop the reference card from the object cloud/mesh; off by "
                         "default because removing it leaves large gaps (the card is "
                         "flattened onto the mount plane like any other point)")
    ap.add_argument("--mesh-mode", choices=("planar", "poisson"), default="planar",
                    help="planar (default): mesh = union of the fitted planes closed "
                         "into a solid; poisson: mirror-closed screened Poisson (slabs)")
    ap.add_argument("--recon-mode", choices=("da3", "sfm_tsdf"), default="sfm_tsdf",
                    help="sfm_tsdf (default): pycolmap SfM poses + DA3 depth aligned "
                         "to the sparse points, fused in an Open3D TSDF volume, with an "
                         "automatic fallback to da3 if SfM fails; da3: DA3 poses + depth, "
                         "per-view point maps unioned "
                         "(README/log intent -- restores the default a later merge "
                         "reverted in surfcap/__main__.py)")
    ap.add_argument("--ckpt", default=None,
                    help="DA3 checkpoint id override, e.g. depth-anything/DA3-BASE "
                         "(default: recon.DEFAULT_CKPT, DA3-LARGE-1.1)")
    ap.add_argument("--no-detail-meshes", action="store_true",
                    help="skip target_mesh_detail_psr.*/target_mesh_hybrid.* (F10); "
                         "only takes effect in --recon-mode sfm_tsdf")
    ap.add_argument("--no-geom-filter", action="store_true",
                    help="disable the cross-view geometric consistency filter that "
                         "drops per-view depth pixels no neighbouring view confirms "
                         "(only takes effect in --recon-mode sfm_tsdf)")
    ap.add_argument("--geom-tol", type=float, default=None,
                    help="relative depth disagreement a neighbouring view may still "
                         "confirm; default = per view from the measured noise "
                         "(floor 0.02, cap 0.06)")
    args = ap.parse_args(argv)

    from .pipeline import gt_check, render_views, run

    # (F2-1) default list extended with the appearance prompts that actually fire on a
    # white card on a light surface (ledge: SAM scores 0.98 "sticker" / 0.97 "label"
    # vs 0.19 "card").  The card-word prompts are still tried first, so metric scenes
    # that already worked never reach them.
    prompts = tuple(args.card_prompt) if args.card_prompt else (
        "card", "credit card", "membership card", "plastic card",
        "white card", "sticker", "label", "white rectangle",
    )

    t0 = time.time()
    target = run(
        args.folder,
        args.out,
        card_prompts=prompts,
        table_prompt=args.table_prompt,
        n_views=args.views,
        process_res=args.res,
        n_max=args.n_max,
        seed=args.seed,
        debug=not args.no_debug,
        use_exif_intrinsics=not args.no_exif_intrinsics,
        thickness_m=args.thickness,
        exclude_card=args.exclude_card,
        mesh_mode=args.mesh_mode,
        ckpt=args.ckpt,
        recon_mode=args.recon_mode,
        geom_filter=not args.no_geom_filter,
        geom_tol=args.geom_tol,
        no_detail_meshes=args.no_detail_meshes,
    )
    total = time.time() - t0

    sc = target.scale or {}
    print("\n" + "=" * 78)
    print(f"surfcap summary  ->  {Path(args.out).resolve()}")
    print("=" * 78)
    print(f"units            : {target.frame.get('units')}")
    print(f"scale factor     : {sc.get('factor')}")
    print(f"scale rms_mm     : {sc.get('rms_mm')}")
    print(f"n_views_with_ref : {sc.get('n_views_with_ref')}   reliable={sc.get('reliable')}")
    print(f"card prompt used : {sc.get('card_prompt_used')}")
    print(f"cloud            : {target.cloud.get('n_points')} pts -> {target.cloud.get('ply')}")
    print("-" * 78)
    hdr = f"{'id':<12}{'role':<8}{'normal':<26}{'extent_m':<20}{'rms_mm':>8}{'n_pts':>9}"
    print(hdr)
    print("-" * 78)
    for s in target.surfaces:
        n = np.round(np.asarray(s.normal, float), 3).tolist()
        e = np.round(np.asarray(s.extent_m, float), 4).tolist()
        print(f"{s.id:<12}{s.role:<8}{str(n):<26}{str(e):<20}"
              f"{s.planarity_rms_mm:>8.2f}{s.n_points:>9}")
    if not target.surfaces:
        print("(no surfaces)")
    print("-" * 78)
    if target.obb:
        ext = target.obb.get("extents_m") or target.obb.get("extents")
        print(f"obb extents_m    : {np.round(np.asarray(ext, float), 4).tolist() if ext is not None else target.obb}")
    else:
        print("obb              : none")
    print(f"warnings ({len(target.warnings)}):")
    for w in target.warnings:
        print(f"  - {w}")
    print(f"total time       : {total:.1f}s")

    if args.gt_check:
        print("\n" + gt_check(args.out, scene_dir=args.folder))
        for p in render_views(args.out):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
