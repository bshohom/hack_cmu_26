"""Command-line entry points: validate / estimate / run / make-cupholder-problem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import device as _device  # noqa: F401  (must precede torch usage)
from .contracts import load_problem, save_problem
from .cost import estimate_cost
from .device import describe_device, pick_device
from .meshing.masks import build_masks
from .meshing.voxel_backend import build_hex_grid
from .regions import resolve_domain

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Agent-facing topology optimization on torch-fem.")


def _prepare(problem_path: Path, elem: Optional[float], iters: Optional[int], volfrac: Optional[float] = None):
    problem = load_problem(problem_path)
    if elem:
        problem.target_element_size = elem
    if iters:
        problem.max_iters = iters
    if volfrac:
        problem.volume_fraction = volfrac
    mesh = build_hex_grid(resolve_domain(problem), problem.target_element_size)
    masks = build_masks(problem, mesh)
    return problem, mesh, masks


@app.command()
def validate(
    problem: Path,
    elem: Optional[float] = typer.Option(None, help="Override target_element_size"),
):
    """Build the grid and masks; report element/node counts per region."""
    _, mesh, masks = _prepare(problem, elem, None)
    typer.echo(json.dumps(masks.report, indent=2))


@app.command()
def estimate(
    problem: Path,
    device: str = typer.Option("auto", help="auto | cuda | cpu"),
    elem: Optional[float] = typer.Option(None, help="Override target_element_size"),
    iters: Optional[int] = typer.Option(None, help="Override max_iters"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Estimate wall time and memory before running (level: ok | slow | too_big)."""
    prob = load_problem(problem)
    est = estimate_cost(prob, device=device, element_size=elem, iters=iters)
    if json_out:
        typer.echo(json.dumps(est.to_dict(), indent=2))
    else:
        typer.echo(
            f"[{est.level}] {est.message}\n"
            f"  device={est.device} elems={est.n_elem} dof={est.n_dof} cases={est.n_cases} iters={est.iters}\n"
            f"  ~{est.sec_per_iter:.2f} s/iter, ~{est.total_sec:.0f} s total, ~{est.peak_bytes / 1e9:.2f} GB"
        )
    if est.level == "too_big":
        raise typer.Exit(code=2)


@app.command()
def run(
    problem: Path,
    out: Path = typer.Option(Path("out"), help="Output directory"),
    device: str = typer.Option("auto", help="auto | cuda | cpu"),
    elem: Optional[float] = typer.Option(None, help="Override target_element_size"),
    iters: Optional[int] = typer.Option(None, help="Override max_iters"),
    volfrac: Optional[float] = typer.Option(None, help="Override volume_fraction"),
    force: bool = typer.Option(False, help="Run even if the estimate says too_big"),
    threshold: float = typer.Option(0.5, help="Density iso-level for the STL"),
):
    """Run SIMP and write rho.vti, design.stl, history.png, render.png, result.json."""
    from .postprocess.export import save_stl, save_vti
    from .postprocess.viz import save_history_png, save_render_png
    from .solver.simp import optimize

    dev = pick_device(device)
    prob = load_problem(problem)
    est = estimate_cost(prob, device=device, element_size=elem, iters=iters)
    typer.echo(f"device: {describe_device(dev)}")
    typer.echo(f"estimate [{est.level}]: {est.message}")
    if est.level == "too_big" and not force:
        typer.echo("refusing to run; pass --force to override", err=True)
        raise typer.Exit(code=2)

    prob, mesh, masks = _prepare(problem, elem, iters, volfrac)
    typer.echo("masks: " + json.dumps(masks.report))
    result = optimize(prob, mesh, masks, dev, log=typer.echo)

    out.mkdir(parents=True, exist_ok=True)
    u0 = result.u[0] if result.u else None
    save_vti(mesh, result.rho, out / "rho.vti", u=u0)
    stl_info = save_stl(mesh, result.rho, out / "design.stl", threshold)
    save_history_png(result.compliance, result.volume, out / "history.png")
    _, render_mode = save_render_png(mesh, result.rho, out / "render.png", threshold)
    save_problem(prob, out / "problem.yaml", header="Resolved problem as run (paths absolute).")

    summary = {
        "device": result.device,
        "solver_mode": result.solver_mode,
        "iters": result.iters,
        "converged": result.converged,
        "wall_time_s": round(result.wall_time, 2),
        "estimate_total_s": round(est.total_sec, 1),
        "compliance": result.compliance,
        "volume_fraction": result.volume,
        "change": result.change,
        "final_volume_fraction": result.volume[-1] if result.volume else None,
        "masks": masks.report,
        "stl": stl_info,
        "render": render_mode,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    typer.echo(
        f"done: {result.iters} iters in {result.wall_time:.1f}s (est {est.total_sec:.0f}s), "
        f"C {result.compliance[0]:.4g} -> {result.compliance[-1]:.4g}, vol {result.volume[-1]:.3f}; "
        f"outputs in {out}/ (render: {render_mode})"
    )


@app.command("make-cupholder-problem")
def make_cupholder_problem(
    dims: Path = typer.Option(Path("cupholder_dimensions.txt")),
    points: Path = typer.Option(Path("cupholder_surface_particles.obj")),
    out: Path = typer.Option(Path("data/cupholder_problem.yaml")),
    elem: float = typer.Option(4.0, help="Target element size (mm)"),
    volfrac: float = typer.Option(0.2, help="Fraction of the design region to fill"),
    safety_factor: float = typer.Option(2.5),
):
    """Demo: derive a TOProblem for the desk-clamp cupholder from its dimension file + scan."""
    from .demo.cupholder import build_cupholder_problem

    problem, report = build_cupholder_problem(dims, points, elem, volfrac, safety_factor)
    save_problem(
        problem,
        out,
        header=(
            "Demo cupholder problem generated by `to-agent make-cupholder-problem`.\n"
            "Loads, material and jaw regions are ASSUMED placeholders (see confidence fields)."
        ),
    )
    typer.echo(json.dumps(report, indent=2, default=str))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
