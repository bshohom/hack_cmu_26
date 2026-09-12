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
from ..meshing.masks import Masks, ProblemSetupError, build_masks
from ..meshing.voxel_backend import HexMesh, build_hex_grid
from ..postprocess.connectivity import carries_boundary_conditions, keep_largest_component
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
    masks = build_masks(problem, mesh)
    from .bc_validation import validate_required_bcs

    bc = validate_required_bcs(problem, mesh, masks.void)
    masks.report["bc_validation"] = bc
    if bc["hard_infeasible"]:
        dead = [
            f"{r['kind']} {r['id']}"
            for r in bc["regions"]
            if r["status"] == "hard_infeasible"
        ]
        raise ProblemSetupError(
            "hard infeasible: required "
            + ", ".join(dead)
            + " has no nodes incident to a non-void element"
        )
    return mesh, masks


def run_problem(
    problem: TOProblem,
    out_dir: str | Path,
    device: str = "auto",
    threshold: float = 0.5,
    force: bool = False,
    log: Optional[Callable[[str], None]] = None,
    progress: Optional[Callable[[int, int, float], None]] = None,
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
    result = optimize(problem, mesh, masks, dev, log=log, progress=progress)

    out_dir.mkdir(parents=True, exist_ok=True)
    # One body before anything is exported or checked, so the STL and the FE model below
    # describe the same object. Trimming the mesh instead would leave the post-check
    # solving a body that is not the one shipped.
    result.rho, trim = keep_largest_component(mesh, result.rho, threshold)
    if trim["trimmed"] and log:
        log(
            f"connectivity: kept the largest of {trim['components_before']} bodies "
            f"({trim['removed_elements']} elements removed)"
        )
    attached = carries_boundary_conditions(
        mesh, result.rho, masks, [lc.id for lc in problem.load_cases], threshold
    )

    u0 = result.u[0] if result.u else None
    save_vti(mesh, result.rho, out_dir / "rho.vti", u=u0)
    stl_info = save_stl(mesh, result.rho, out_dir / "design.stl", threshold)
    save_history_png(result.compliance, result.volume, out_dir / "history.png")
    _, render_mode = save_render_png(mesh, result.rho, out_dir / "render.png", threshold)
    save_problem(problem, out_dir / "problem.yaml", header="Resolved problem as run (paths absolute).")
    try:
        from .postcheck import post_check

        check = post_check(problem, mesh, masks, result.rho, dev, threshold)
        if log:
            w = check["worst_case"]
            fos = f"{w['factor_of_safety']:.2f}" if w["factor_of_safety"] else "n/a"
            log(
                f"post-check ({w['load_case_id']}, nominal load): max disp {w['max_displacement_mm']:.3f} mm, "
                f"max von Mises {w['max_von_mises_MPa']:.2f} MPa, FoS {fos}"
            )
    except Exception as exc:  # noqa: BLE001 — the check must never break a finished run
        check = {"error": f"{type(exc).__name__}: {exc}"}

    summary = {
        "post_check": check,
        "connectivity": {**trim, **attached},
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
