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
    from .integration.run import CostTooHigh, run_problem

    prob = load_problem(problem)
    if elem:
        prob.target_element_size = elem
    if iters:
        prob.max_iters = iters
    if volfrac:
        prob.volume_fraction = volfrac
    try:
        outcome = run_problem(prob, out, device=device, threshold=threshold, force=force, log=typer.echo)
    except CostTooHigh:
        typer.echo("refusing to run; pass --force to override", err=True)
        raise typer.Exit(code=2)
    s = outcome.summary
    typer.echo(
        f"done: {s['iters']} iters in {s['wall_time_s']:.1f}s (est {s['estimate']['total_sec']:.0f}s), "
        f"C {s['compliance'][0]:.4g} -> {s['compliance'][-1]:.4g}, vol {s['final_volume_fraction']:.3f}; "
        f"outputs in {out}/ (render: {s['render']})"
    )


@app.command("run-agentic")
def run_agentic(
    input: Path = typer.Argument(..., help="JSON file with the interface's TopologyInput (model_dump)"),
    out_root: Path = typer.Option(Path("out/agentic"), help="Root for per-run output directories"),
):
    """Adapter entry point: TopologyInput JSON -> live optimization -> TopologyOutput JSON on stdout."""
    from .integration.agentic import AdapterError, run_topology

    data = json.loads(input.read_text())
    try:
        result = run_topology(data, out_root=out_root, log=lambda m: typer.echo(m, err=True))
    except AdapterError as exc:
        typer.echo(json.dumps({"is_mock": True, "notes": str(exc)}))
        raise typer.Exit(code=2)
    typer.echo(json.dumps(result, indent=2))


@app.command("make-problem")
def make_problem(
    task: str = typer.Argument(..., help="cupholder | desk_bag_hook | stapler_shelf"),
    out: Optional[Path] = typer.Option(None, help="Output YAML (default data/<task>_problem.yaml)"),
    elem: Optional[float] = typer.Option(None, help="Target element size (mm); default per task"),
    volfrac: Optional[float] = typer.Option(None, help="Volume fraction; default per task"),
    safety_factor: float = typer.Option(2.5),
):
    """Build a demo TOProblem for a registered task from its dimension file + point cloud."""
    from .demo.registry import DEMO_FILES, get_builder

    builder = get_builder(task)
    dims, pts = DEMO_FILES[task]
    kwargs = {"safety_factor": safety_factor}
    if elem:
        kwargs["element_size"] = elem
    if volfrac:
        kwargs["volume_fraction"] = volfrac
    problem, report = builder(dims, pts, **kwargs)
    out = out or Path("data") / f"{task}_problem.yaml"
    save_problem(problem, out, header=f"Demo problem for {task} generated by `to-agent make-problem`.\nLoads/material are ASSUMED placeholders.")
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "dims"}, indent=2, default=str))
    typer.echo(f"wrote {out}")


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
