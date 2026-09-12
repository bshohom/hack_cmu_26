from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from surfcap.types import Frame

_EXTS = (".jpg", ".jpeg", ".png", ".heic")

_HEIF_REGISTERED = False
_HEIF_WARNED = False


def _try_register_heif() -> str | None:
    """Register pillow_heif opener once. Returns a warning string if unavailable."""
    global _HEIF_REGISTERED, _HEIF_WARNED
    if _HEIF_REGISTERED:
        return None
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True
        return None
    except ImportError:
        if not _HEIF_WARNED:
            _HEIF_WARNED = True
            return "pillow_heif not importable; HEIC images will fail to load"
        return None


def load_image(path: str | Path, long_edge: int = 1024) -> Frame:
    path = Path(path)
    warn = _try_register_heif()
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    scale_factor = 1.0
    if max(orig_w, orig_h) != long_edge:
        scale_factor = long_edge / max(orig_w, orig_h)
        new_w = max(1, round(orig_w * scale_factor))
        new_h = max(1, round(orig_h * scale_factor))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w, new_h = orig_w, orig_h

    rgb = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    frame = Frame(
        path=str(path),
        rgb=rgb,
        gray=gray,
        size=(new_w, new_h),
        orig_size=(orig_w, orig_h),
        scale_factor=scale_factor,
        blur=blur,
    )
    if warn:
        # attach as attribute-less side channel; caller (load_folder) handles global warns
        pass
    return frame


def load_folder(
    folder: str | Path,
    long_edge: int = 1024,
    blur_drop: float = 25.0,
    min_keep: int = 8,
    max_images: int | None = None,
) -> tuple[list[Frame], list[str]]:
    folder = Path(folder)
    warnings: list[str] = []

    heif_warn = _try_register_heif()
    if heif_warn:
        warnings.append(heif_warn)

    paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _EXTS],
        key=lambda p: p.name,
    )
    if max_images is not None:
        paths = paths[:max_images]

    if len(paths) == 0:
        raise ValueError("zero images")

    frames: list[Frame] = []
    for p in paths:
        try:
            frames.append(load_image(p, long_edge=long_edge))
        except Exception as e:
            warnings.append(f"failed to load {p.name}: {e}")

    if len(frames) == 0:
        raise ValueError("zero images")

    # blur filtering, but never drop below min_keep
    frames_sorted_by_blur = sorted(frames, key=lambda f: f.blur, reverse=True)
    keep_set = set()
    dropped = []
    for f in frames:
        if f.blur >= blur_drop:
            keep_set.add(f.path)
        else:
            dropped.append(f)

    if len(keep_set) < min_keep:
        # keep the sharpest frames up to min_keep (or all frames if fewer exist)
        n_to_keep = min(min_keep, len(frames))
        keep_set = {f.path for f in frames_sorted_by_blur[:n_to_keep]}

    kept_frames = [f for f in frames if f.path in keep_set]
    dropped_names = [Path(f.path).name for f in frames if f.path not in keep_set]
    if dropped_names:
        warnings.append(
            f"dropped {len(dropped_names)} blurry frame(s): {', '.join(dropped_names)}"
        )

    kept_frames = sorted(kept_frames, key=lambda f: Path(f.path).name)
    return kept_frames, warnings
