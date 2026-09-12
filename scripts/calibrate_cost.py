"""Time single linear solves at several grid sizes and fit t = a * n_dof^b per device.

Writes data/calibration.json, which to_agent.cost uses for its estimates.
Usage: python scripts/calibrate_cost.py [--sizes 12,20,30,40]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from to_agent import device as _device  # noqa: F401
import numpy as np
import torch

from to_agent.contracts import BoxRegion, LoadCase, Support, TOProblem
from to_agent.cost import CALIBRATION_PATH, fit_power_law
from to_agent.meshing.masks import build_masks
from to_agent.meshing.voxel_backend import build_hex_grid
from to_agent.solver.simp import make_model, solve


def cantilever(n: int) -> tuple[TOProblem, float]:
    L = float(n)
    problem = TOProblem(
        design_domain=BoxRegion(min=(0, 0, 0), max=(L, L * 0.6, L * 0.6)),
        supports=[Support(id="wall", region=BoxRegion(min=(-0.1, -1, -1), max=(0.1, L, L)))],
        load_cases=[LoadCase(id="tip", region=BoxRegion(min=(L - 0.1, -1, -0.1), max=(L + 0.1, L, 0.1)), force_N=(0, 0, -1.0))],
        target_element_size=1.0,
    )
    return problem, 1.0


def time_solve(problem: TOProblem, device: torch.device) -> tuple[int, float, float]:
    mesh = build_hex_grid(problem.design_domain, problem.target_element_size)
    masks = build_masks(problem, mesh)
    model, tdev, _ = make_model(mesh, masks, problem, device)
    model.forces = masks.forces[0].to(tdev)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    solve(model, device, tdev)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    solve(model, device, tdev)
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem = float(torch.cuda.max_memory_allocated())
    else:
        mem = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return mesh.n_dof, time.time() - t0, mem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="12,20,30,40")
    ap.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    calib: dict = {"meta": {"host": platform.node(), "torch": torch.__version__, "sizes": sizes}}
    for name in devices:
        dev = torch.device(name)
        dofs, secs, mems = [], [], []
        for n in sizes:
            problem, _ = cantilever(n)
            n_dof, sec, mem = time_solve(problem, dev)
            dofs.append(n_dof)
            secs.append(sec)
            mems.append(mem)
            print(f"{name}: n={n:3d} dof={n_dof:7d} solve={sec:7.3f}s mem={mem / 1e9:.2f}GB")
        a, b = fit_power_law(dofs, secs)
        # memory per dof from the largest run (peak allocations dominate there); pad by 1.5x
        bytes_per_dof = 1.5 * (mems[-1] - (mems[0] if name == "cpu" else 0.0)) / max(dofs[-1] - (dofs[0] if name == "cpu" else 0), 1)
        bytes_per_dof = float(max(bytes_per_dof, 500.0))
        calib[name] = {"a": a, "b": b, "bytes_per_dof": bytes_per_dof, "dofs": dofs, "seconds": secs}
        print(f"{name}: t = {a:.3e} * dof^{b:.3f}, {bytes_per_dof:.0f} B/dof")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calib, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
