"""
Units metres. World frame Z-up (gravity). Origin = credit-card centre projected onto the table-top plane. X = card long edge direction projected onto the plane. Y = Z × X.
Camera model: OpenCV pinhole, K is 3×3. Extrinsics stored cam-to-world (c2w) 4×4; det(R) must be +1.
Every stage appends strings to `warnings`; nothing raises except "zero images".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CARD_MM = (85.60, 53.98)
QUARTER_MM = 24.26
ROLES = ("top", "front", "side", "bottom", "other")


def _round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively convert numpy arrays/scalars to plain python and round floats."""
    if isinstance(obj, np.ndarray):
        return _round_floats(obj.tolist(), ndigits)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


@dataclass
class Frame:
    path: str
    rgb: np.ndarray  # HxWx3 uint8 (resized)
    gray: np.ndarray  # HxW uint8
    size: tuple[int, int]  # (W,H) after resize
    orig_size: tuple[int, int]
    scale_factor: float  # resized/original
    blur: float


@dataclass
class Recon:
    depth: np.ndarray  # [N,H,W] float32
    conf: np.ndarray  # [N,H,W]
    K: np.ndarray  # [N,3,3]
    c2w: np.ndarray  # [N,4,4]
    pointmap: np.ndarray  # [N,H,W,3] world-frame xyz (before scaling)
    valid: np.ndarray  # [N,H,W] bool
    meta: dict = field(default_factory=dict)  # actual {ckpt, process_res, n_views} used


@dataclass
class Masks:
    table: np.ndarray  # [N,H,W] bool
    card: np.ndarray  # [N,H,W] bool
    table_scores: list[float]
    card_scores: list[float]


@dataclass
class ScaleResult:
    scale: float
    rms_mm: float
    n_views: int
    reliable: bool
    card_corners_w: np.ndarray | None  # (4,3) unscaled world frame; 0-1 & 2-3 are LONG edges
    per_view: dict


@dataclass
class Surface:
    id: str
    role: str  # top|front|side|bottom|other
    normal: list[float]
    centroid: list[float]
    extent_m: list[float]  # [long, short]
    polygon_2d: list[list[float]]  # in plane basis, metres
    polygon_3d: list[list[float]]
    planarity_rms_mm: float
    n_points: int


@dataclass
class Target:
    frame: dict
    scale: dict
    cloud: dict
    surfaces: list[Surface]
    obb: dict | None
    context: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "frame": self.frame,
            "scale": self.scale,
            "cloud": self.cloud,
            "surfaces": [asdict(s) if isinstance(s, Surface) else s for s in self.surfaces],
            "obb": self.obb,
            "context": self.context,
            "warnings": list(self.warnings),
        }
        return _round_floats(d)

    @staticmethod
    def from_dict(d: dict) -> "Target":
        surfaces = [
            s if isinstance(s, Surface) else Surface(**s) for s in d.get("surfaces", [])
        ]
        return Target(
            frame=d.get("frame", {}),
            scale=d.get("scale", {}),
            cloud=d.get("cloud", {}),
            surfaces=surfaces,
            obb=d.get("obb"),
            context=d.get("context", {}),
            warnings=list(d.get("warnings", [])),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "Target":
        with open(path, "r") as f:
            d = json.load(f)
        return Target.from_dict(d)
