"""Run a TOProblem end to end and write all artifacts. Shared by the CLI and the adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import device as _device  # noqa: F401
from ..contracts import TOProblem, save_problem
from ..cost import CostEstimate, estimate_cost
from ..device import describe_device, pick_device
from ..meshing.masks import Masks, build_masks
from ..meshing.voxel_backend import HexMesh, build_hex_grid
from ..postprocess.export import save_stl, save_vti
from ..postprocess.viz import save_history_png, save_render_png
from ..regions import resolve_domain
from ..solver.simp import SIMPResult, optimize


class CostTooHigh(RuntimeError):
    def __init__(self, estimate: CostEstimate):
        super().__init__(estimate.message)
        self.estimate = estimate


@dataclass
class RunOutcome:
    summary: dict
    result: SIMPResult
    mesh: HexMesh
    masks: Masks
    estimate: CostEstimate
    out_dir: Path


def prepare(problem: TOProblem) -> tuple[HexMesh, Masks]:
    mesh = build_hex_grid(resolve_domain(problem), problem.target_element_size)
    return mesh, build_masks(problem, mesh)


def run_problem(
    problem: TOProblem,
    out_dir: str | Path,
    device: str = "auto",
    threshold: float = 0.5,
    force: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> RunOutcome:
    out_dir = Path(out_dir)
    dev = pick_device(device)
    est = estimate_cost(problem, device=device)
    if log:
        log(f"device: {describe_device(dev)}")
        log(f"estimate [{est.level}]: {est.message}")
    if est.level == "too_big" and not force:
        raise CostTooHigh(est)

    mesh, masks = prepare(problem)
    if log:
        log("masks: " + json.dumps(masks.report))
    result = optimize(problem, mesh, masks, dev, log=log)

    out_dir.mkdir(parents=True, exist_ok=True)
    u0 = result.u[0] if result.u else None
    save_vti(mesh, result.rho, out_dir / "rho.vti", u=u0)
    stl_info = save_stl(mesh, result.rho, out_dir / "design.stl", threshold)
    save_history_png(result.compliance, result.volume, out_dir / "history.png")
    _, render_mode = save_render_png(mesh, result.rho, out_dir / "render.png", threshold)
    save_problem(problem, out_dir / "problem.yaml", header="Resolved problem as run (paths absolute).")

    summary = {
        "out_dir": str(out_dir),
        "device": result.device,
        "device_desc": describe_device(dev),
        "solver_mode": result.solver_mode,
        "iters": result.iters,
        "converged": result.converged,
        "wall_time_s": round(result.wall_time, 2),
        "estimate": est.to_dict(),
        "compliance": result.compliance,
        "volume_fraction": result.volume,
        "change": result.change,
        "final_volume_fraction": result.volume[-1] if result.volume else None,
        "masks": masks.report,
        "stl": stl_info,
        "render": render_mode,
        "artifacts": {
            "design_stl": str(out_dir / "design.stl"),
            "rho_vti": str(out_dir / "rho.vti"),
            "history_png": str(out_dir / "history.png"),
            "render_png": str(out_dir / "render.png"),
            "problem_yaml": str(out_dir / "problem.yaml"),
            "result_json": str(out_dir / "result.json"),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return RunOutcome(summary, result, mesh, masks, est, out_dir)
