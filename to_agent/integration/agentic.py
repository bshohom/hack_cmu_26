"""Adapter for the agentic interface: TopologyInput (dict) -> TOProblem -> run -> TopologyOutput (dict).

Never imports the interface's `schemas` (it is not a package); everything is plain dicts.
The candidate mesh's own frame is the problem frame; TopologyInput contributes scalars only
(load totals, material, mass cap, ids). Generated warm starts (task == "generated") carry a
`<name>_regions.json` in desk_edge_frame from which the problem is built generically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts import (
    BoxRegion,
    InsideMeshRegion,
    IntersectionRegion,
    LoadCase,
    NearPointsRegion,
    Support,
    TOProblem,
)
from ..cost import estimate_cost
from ..demo.registry import get_builder
from .materials import material_from_name
from .run import CostTooHigh, RunOutcome, run_problem

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "out" / "agentic"
MAX_COARSEN = 3
COARSEN_FACTOR = 1.25

Log = Optional[Callable[[str], None]]


class AdapterError(ValueError):
    """Agent-readable reason why the live optimization could not run."""


# ----------------------------------------------------------------------------- helpers
def _mesh_volume(path: str | None) -> Optional[float]:
    if not path or not Path(path).exists():
        return None
    try:
        import trimesh

        m = trimesh.load(path, force="mesh")
        if m.is_empty or len(m.faces) == 0 or not m.is_watertight:
            return None
        return float(abs(m.volume))
    except Exception:  # noqa: BLE001
        return None


MAX_CONTAINS_FACES = 20_000  # trimesh.contains (ray casting) gets slow beyond this


def _warm_start_regions(candidate: dict, h: float) -> list:
    """Warm-start occupancy: particle KDTree shell (fast) when a particle file exists,
    inside_mesh for small watertight meshes, otherwise a surface sample of the mesh."""
    particles = candidate.get("particle_path")
    if particles and Path(particles).exists():
        return [NearPointsRegion(path=str(particles), tol=h)]
    mesh_path = candidate.get("mesh_path")
    if mesh_path and Path(mesh_path).exists():
        try:
            import trimesh

            m = trimesh.load(mesh_path, force="mesh")
            if not m.is_empty and len(m.faces) > 0:
                if m.is_watertight and len(m.faces) <= MAX_CONTAINS_FACES:
                    return [InsideMeshRegion(path=str(mesh_path))]
                sampled = Path(mesh_path).with_suffix("").as_posix() + "_particles.obj"
                pts, _ = trimesh.sample.sample_surface(m, 75_000)
                with open(sampled, "w") as f:
                    f.write("# surface sample of the candidate mesh; units mm\n")
                    for p in pts:
                        f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
                return [NearPointsRegion(path=sampled, tol=h)]
        except Exception:  # noqa: BLE001
            pass
    raise AdapterError("candidate has neither a usable mesh_path nor a particle_path")


def _box(d: dict, pad: float = 0.0) -> BoxRegion:
    lo = [float(v) - pad for v in d["min"]]
    hi = [float(v) + pad for v in d["max"]]
    return BoxRegion(min=tuple(lo), max=tuple(hi))


def _primary_force(loads: list[dict]) -> tuple[str, tuple[float, float, float]]:
    if not loads:
        raise AdapterError("TopologyInput.loads is empty")
    lc = loads[0]
    f = tuple(float(v) for v in lc.get("force_N", (0.0, 0.0, 0.0)))
    return str(lc.get("load_case_id") or lc.get("name") or "static_gravity"), f


# ----------------------------------------------------------------------------- builders
def build_generated_problem(candidate: dict, topology_input: dict, h: float, volfrac: float, safety_factor: float) -> tuple[TOProblem, dict]:
    """Generic problem for a Grok-generated warm start using its regions file (desk_edge_frame)."""
    regions_path = candidate.get("regions_path")
    if not regions_path or not Path(regions_path).exists():
        raise AdapterError("generated candidate has no regions file")
    spec = json.loads(Path(regions_path).read_text())
    regions = spec.get("regions", {})
    params = spec.get("params", {})
    desk_t = topology_input.get("desk_thickness_mm")
    if desk_t is None:
        for key, val in params.items():
            if "desk" in key.lower() and "thick" in key.lower():
                desk_t = float(val)
                break
    lo = [float(v) for v in candidate["bbox_min_mm"]]
    hi = [float(v) for v in candidate["bbox_max_mm"]]
    pad = h + 1.0
    warm = _warm_start_regions(candidate, h)
    mesh_region = warm[0]

    load_id, force = _primary_force(topology_input.get("loads", []))
    load_box = _box(regions["load"], pad=0.5)
    preserve = [IntersectionRegion(regions=[mesh_region, _box(regions["load"], pad=h)])]
    supports = []
    for i, m in enumerate(regions.get("mounts", [])):
        box = _box(m, pad=0.5)
        supports.append(Support(id=str(m.get("name") or f"mount_{i}"), region=box, confidence="generated"))
        preserve.append(IntersectionRegion(regions=[mesh_region, _box(m, pad=h)]))
    if not supports:
        raise AdapterError("generated candidate regions have no mounts")
    void = [_box(k) for k in regions.get("keep_out", [])]
    if desk_t is not None:
        void.append(BoxRegion(min=(lo[0] - pad, lo[1] - pad, 0.0), max=(0.0, hi[1] + pad, float(desk_t))))

    problem = TOProblem(
        design_domain=BoxRegion(min=(lo[0] - pad, lo[1] - pad, lo[2] - pad), max=(hi[0] + pad, hi[1] + pad, hi[2] + pad)),
        warm_start=warm,
        preserve=preserve,
        void=void,
        supports=supports,
        load_cases=[LoadCase(id=load_id, region=load_box, force_N=force, confidence="user")],
        safety_factor=safety_factor,
        volume_fraction=volfrac,
        target_element_size=h,
        filter_radius=1.5 * h,
        max_iters=50,
        notes=f"generated warm start {candidate.get('candidate_name')} (desk_edge_frame); desk thickness {desk_t} mm",
    )
    return problem, {"regions": regions, "desk_thickness_mm": desk_t}


def build_from_registry(candidate: dict, topology_input: dict, h: float, volfrac: Optional[float], safety_factor: float) -> tuple[TOProblem, dict]:
    task = candidate.get("task") or "cupholder"
    builder = get_builder(task)
    dims = candidate.get("dimensions_path")
    pts = candidate.get("particle_path")
    if not dims or not pts:
        raise AdapterError(f"task {task!r} needs dimensions_path and particle_path on the candidate")
    kwargs: dict[str, Any] = {"element_size": h, "safety_factor": safety_factor}
    if volfrac is not None:
        kwargs["volume_fraction"] = volfrac
    problem, report = builder(dims, pts, **kwargs)
    # prefer the watertight candidate mesh as the warm start when present
    try:
        problem.warm_start = _warm_start_regions(candidate, h)
    except AdapterError:
        pass
    # the interface's primary load (payload mass) replaces the template's primary case
    load_id, force = _primary_force(topology_input.get("loads", []))
    if any(abs(v) > 0 for v in force):
        primary = problem.load_cases[0]
        primary.id = load_id
        primary.force_N = force
        primary.confidence = "user"
    return problem, report


MASS_TARGET_RATIO = 1.0  # same material budget as the warm start (TO redistributes it)
VF_BOUNDS = (0.08, 0.5)
MIN_ITERS_GENERATED = 40


def volume_fraction_from_candidate(problem: TOProblem, candidate: dict) -> tuple[float, str]:
    """Volume fraction of the design region such that preserve + design material ≈
    MASS_TARGET_RATIO x the candidate's volume. Falls back to the current value."""
    from .run import prepare

    v_cand = _mesh_volume(candidate.get("mesh_path"))
    if not v_cand:
        return problem.volume_fraction, "candidate volume unknown; kept template volume fraction"
    mesh, masks = prepare(problem)
    v_e = mesh.elem_volume
    n_design = int(masks.design.sum())
    n_preserve = int(masks.preserve.sum())
    vf = (MASS_TARGET_RATIO * v_cand - n_preserve * v_e) / max(n_design * v_e, 1e-9)
    vf_clamped = float(min(max(vf, VF_BOUNDS[0]), VF_BOUNDS[1]))
    return vf_clamped, (
        f"volume fraction {vf_clamped:.3f} set from candidate volume {v_cand:.0f} mm^3 "
        f"(target {MASS_TARGET_RATIO:.0%}, preserve {n_preserve * v_e:.0f} mm^3, design region {n_design * v_e:.0f} mm^3)"
    )


# ----------------------------------------------------------------------------- entry point
def run_topology(topology_input: dict, out_root: str | Path | None = None, log: Log = None) -> dict:
    """Return a dict with the TopologyOutput fields (is_mock=False) or raise AdapterError."""
    candidate = topology_input.get("candidate")
    if not candidate:
        raise AdapterError("no candidate geometry on TopologyInput (imported or generated warm start required)")
    opts = topology_input.get("solver_options") or {}
    h = float(opts.get("element_size_mm") or 4.0)
    volfrac = opts.get("volume_fraction")
    safety_factor = float(opts.get("safety_factor") or 2.5)
    time_budget = float(opts.get("time_budget_s") or 150.0)
    device = str(opts.get("device") or "auto")
    max_iters = opts.get("max_iters")

    task = candidate.get("task") or "cupholder"
    notes: list[str] = []
    if task == "generated":
        problem, report = build_generated_problem(candidate, topology_input, h, float(volfrac or 0.2), safety_factor)
        if volfrac is None:
            problem.volume_fraction, vf_note = volume_fraction_from_candidate(problem, candidate)
            notes.append(vf_note)
    else:
        problem, report = build_from_registry(candidate, topology_input, h, volfrac, safety_factor)
    material, mat_note = material_from_name(topology_input.get("material"))
    problem.material = material
    if max_iters:
        problem.max_iters = int(max_iters)
    if task == "generated":
        problem.max_iters = max(problem.max_iters, MIN_ITERS_GENERATED)

    if mat_note:
        notes.append(mat_note)
    requested_vf = topology_input.get("target_volume_fraction")
    if requested_vf is not None and abs(float(requested_vf) - problem.volume_fraction) > 1e-9:
        notes.append(f"requested volume fraction {requested_vf} replaced by template value {problem.volume_fraction}")

    # auto-coarsen until the estimate fits the time budget
    for _ in range(MAX_COARSEN + 1):
        est = estimate_cost(problem, device=device)
        if est.level != "too_big" and est.total_sec <= time_budget:
            break
        problem.target_element_size *= COARSEN_FACTOR
        problem.filter_radius = 1.5 * problem.target_element_size
        notes.append(f"coarsened element size to {problem.target_element_size:.2f} mm ({est.message})")

    name = candidate.get("candidate_name") or task
    out_dir = Path(out_root or DEFAULT_OUT_ROOT) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        outcome = run_problem(problem, out_dir, device=device, log=log)
    except CostTooHigh as exc:
        raise AdapterError(f"problem too big: {exc.estimate.message}") from exc
    except RuntimeError as exc:
        if device != "cpu" and "cuda" in str(exc).lower():
            notes.append(f"CUDA failed ({str(exc)[:120]}); retried on cpu")
            outcome = run_problem(problem, out_dir, device="cpu", log=log)
        else:
            raise
    return topology_output(outcome, candidate, problem, notes, report)


def topology_output(outcome: RunOutcome, candidate: dict, problem: TOProblem, notes: list[str], report: dict) -> dict:
    s = outcome.summary
    v_candidate = _mesh_volume(candidate.get("mesh_path"))
    v_design = s["stl"].get("volume_mm3")
    if v_design is None:
        v_design = float(outcome.result.rho.sum() * outcome.mesh.elem_volume)
    mass_reduction = 100.0 * (1.0 - v_design / v_candidate) if v_candidate else None
    model = (
        f"to_agent SIMP/OC on torch-fem ({s['solver_mode']}, h={problem.target_element_size:.2f} mm, "
        f"{s['masks']['n_elem']} elems, SF={problem.safety_factor})"
    )
    components = s["stl"].get("components")
    if components is not None and components != 1:
        notes.append(
            f"WARNING: the rho>=0.5 isosurface has {components} disconnected bodies; the load path is "
            "not fully solid at this volume fraction — raise volume_fraction or max_iters and rerun"
        )
    post_check = None
    check = s.get("post_check") or {}
    if "worst_case" in check:
        from .postcheck import to_analysis_output

        post_check = to_analysis_output(check)
        w = check["worst_case"]
        fos = f"{w['factor_of_safety']:.2f}" if w["factor_of_safety"] else "n/a"
        notes.append(
            f"post-check at nominal load ({w['load_case_id']}): max displacement {w['max_displacement_mm']:.3f} mm, "
            f"max von Mises {w['max_von_mises_MPa']:.2f} MPa, factor of safety {fos}"
        )
    return {
        "is_mock": False,
        "compliance": float(s["compliance"][-1]),
        "volume_fraction": float(s["final_volume_fraction"]),
        "mass_reduction_pct": round(mass_reduction, 1) if mass_reduction is not None else 0.0,
        "optimized_geometry_ref": s["artifacts"]["design_stl"],
        "solver_status": "converged" if s["converged"] else "max_iters_reached",
        "model": model,
        "artifacts": {**s["artifacts"], "candidate_mesh": candidate.get("mesh_path") or "", "out_dir": s["out_dir"]},
        "iterations": int(s["iters"]),
        "wall_time_s": float(s["wall_time_s"]),
        "converged": bool(s["converged"]),
        "notes": "; ".join(notes + [
            f"compliance {s['compliance'][0]:.4g} -> {s['compliance'][-1]:.4g}",
            f"design volume {v_design:.0f} mm^3" + (f" vs candidate {v_candidate:.0f} mm^3" if v_candidate else ""),
            f"estimate {s['estimate']['total_sec']:.0f} s, actual {s['wall_time_s']:.0f} s",
        ]),
        "problem_report": {k: v for k, v in report.items() if k != "dims"},
        "post_check": post_check,
    }
