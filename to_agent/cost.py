"""Wall-time / memory estimate for a problem before running it.

Model: seconds per linear solve = a * n_dof ** b (per device), calibrated by
scripts/calibrate_cost.py into data/calibration.json. Conservative defaults are used when
no calibration exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contracts import TOProblem
from .device import device_memory_bytes, pick_device
from .meshing.voxel_backend import grid_shape
from .regions import resolve_domain

CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.json"

# Deliberately pessimistic defaults (uncalibrated machine).
_DEFAULT = {
    "cpu": {"a": 5e-6, "b": 1.15, "bytes_per_dof": 6000.0},
    "cuda": {"a": 2e-6, "b": 1.10, "bytes_per_dof": 6000.0},
}

OK_SECONDS = 300.0  # < 5 min: ok
SLOW_SECONDS = 1800.0  # < 30 min: slow; above: too_big
MEMORY_FRACTION = 0.8


@dataclass
class CostEstimate:
    device: str
    n_elem: int
    n_nodes: int
    n_dof: int
    n_cases: int
    iters: int
    sec_per_solve: float
    sec_per_iter: float
    total_sec: float
    peak_bytes: float
    device_bytes: float
    level: str  # ok | slow | too_big
    message: str
    calibrated: bool

    def to_dict(self) -> dict:
        return asdict(self)


def load_calibration(path: Path = CALIBRATION_PATH) -> tuple[dict, bool]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return {**_DEFAULT, **{k: v for k, v in data.items() if k in ("cpu", "cuda")}}, True
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT), False


def estimate_counts(problem: TOProblem, element_size: float | None = None) -> tuple[int, int]:
    """(n_elem, n_nodes) for the grid the problem would use."""
    domain = resolve_domain(problem)
    n, _ = grid_shape(domain, element_size or problem.target_element_size)
    return int(np.prod(n)), int(np.prod(n + 1))


def estimate_cost(
    problem: TOProblem,
    device: str = "auto",
    element_size: float | None = None,
    iters: int | None = None,
    calibration: Path = CALIBRATION_PATH,
) -> CostEstimate:
    dev = pick_device(device)
    calib, calibrated = load_calibration(calibration)
    coeff = calib[dev.type]
    n_elem, n_nodes = estimate_counts(problem, element_size)
    n_dof = 3 * n_nodes
    n_cases = len(problem.load_cases)
    iters = iters or problem.max_iters

    sec_per_solve = coeff["a"] * n_dof ** coeff["b"]
    sec_per_iter = n_cases * sec_per_solve * 1.15  # + filter / update overhead
    total = iters * sec_per_iter + 5.0  # + setup
    peak = coeff["bytes_per_dof"] * n_dof
    dev_bytes = float(device_memory_bytes(dev))

    if dev_bytes and peak > MEMORY_FRACTION * dev_bytes:
        level = "too_big"
        message = (
            f"estimated peak memory {peak / 1e9:.1f} GB exceeds {MEMORY_FRACTION:.0%} of "
            f"{dev.type} memory ({dev_bytes / 1e9:.1f} GB); increase target_element_size"
        )
    elif total > SLOW_SECONDS:
        level = "too_big"
        message = (
            f"estimated {total / 60:.0f} min for {n_elem} elements x {iters} iterations; "
            "increase target_element_size or reduce max_iters (or pass --force)"
        )
    elif total > OK_SECONDS:
        level = "slow"
        message = f"estimated {total / 60:.1f} min; acceptable but consider a coarser mesh"
    else:
        level = "ok"
        message = f"estimated {total:.0f} s for {n_elem} elements x {iters} iterations"
    if not calibrated:
        message += " (uncalibrated estimate; run scripts/calibrate_cost.py)"

    return CostEstimate(
        device=str(dev),
        n_elem=n_elem,
        n_nodes=n_nodes,
        n_dof=n_dof,
        n_cases=n_cases,
        iters=iters,
        sec_per_solve=sec_per_solve,
        sec_per_iter=sec_per_iter,
        total_sec=total,
        peak_bytes=peak,
        device_bytes=dev_bytes,
        level=level,
        message=message,
        calibrated=calibrated,
    )


def fit_power_law(n_dofs: list[int], seconds: list[float]) -> tuple[float, float]:
    """Least-squares fit of t = a * n^b in log space. Returns (a, b)."""
    x = np.log(np.asarray(n_dofs, float))
    y = np.log(np.asarray(seconds, float))
    b, log_a = np.polyfit(x, y, 1)
    return float(np.exp(log_a)), float(b)
