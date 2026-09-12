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
    Assumption,
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
from .agent_regions import apply_agent_regions, load_cases_from_input
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


def _as_box(region: Any) -> Optional[dict]:
    """Accept {"min","max"} or {"name","box":{"min","max"}} (generators emit both)."""
    if not isinstance(region, dict):
        return None
    if "min" in region and "max" in region:
        return region
    inner = region.get("box") or region.get("bounds")
    if isinstance(inner, dict) and "min" in inner and "max" in inner:
        return {**{k: v for k, v in region.items() if k != "box"}, **inner}
    return None


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
    load_raw = _as_box(regions.get("load"))
    if load_raw is None:
        raise AdapterError("generated candidate regions have no usable 'load' box")
    load_box = _box(load_raw, pad=0.5)
    # Agent load cases are placed by their own regions; the candidate's mounts are kept
    # because its preserve regions (and its geometry) are built around them.
    agent_loads, load_assumptions = load_cases_from_input(topology_input, h)
    assumptions: list[Assumption] = list(load_assumptions)
    assumptions.append(
        Assumption(
            field="supports",
            value=", ".join(str(_as_box(m).get("name", "?")) for m in regions.get("mounts", []) if _as_box(m)),
            basis=f"mounts taken from the generated candidate's regions file {Path(regions_path).name}",
        )
    )
    preserve = [IntersectionRegion(regions=[mesh_region, _box(load_raw, pad=h)])]
    supports = []
    for i, raw in enumerate(regions.get("mounts", [])):
        m = _as_box(raw)
        if m is None:
            continue
        box = _box(m, pad=0.5)
        supports.append(Support(id=str(m.get("name") or f"mount_{i}"), region=box, provenance="assumed"))
        preserve.append(IntersectionRegion(regions=[mesh_region, _box(m, pad=h)]))
    if not supports:
        raise AdapterError("generated candidate regions have no mounts")
    void = [_box(b) for b in (_as_box(k) for k in regions.get("keep_out", [])) if b is not None]
    if desk_t is not None:
        void.append(BoxRegion(min=(lo[0] - pad, lo[1] - pad, 0.0), max=(0.0, hi[1] + pad, float(desk_t))))

    problem = TOProblem(
        design_domain=BoxRegion(min=(lo[0] - pad, lo[1] - pad, lo[2] - pad), max=(hi[0] + pad, hi[1] + pad, hi[2] + pad)),
        warm_start=warm,
        preserve=preserve,
        void=void,
        supports=supports,
        load_cases=agent_loads or [LoadCase(id=load_id, region=load_box, force_N=force, provenance="user")],
        safety_factor=safety_factor,
        volume_fraction=volfrac,
        target_element_size=h,
        filter_radius=1.5 * h,
        max_iters=50,
        assumptions=assumptions,
        notes=f"generated warm start {candidate.get('candidate_name')} (desk_edge_frame); desk thickness {desk_t} mm",
    )
    return problem, {
        "regions": regions,
        "desk_thickness_mm": desk_t,
        "loads_from": "agent load_regions" if agent_loads else "candidate regions file",
        "assumptions": [a.model_dump() for a in assumptions],
    }


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
    # The template is a geometry/warm-start source, not the authority on boundary conditions.
    report["region_source"] = apply_agent_regions(problem, topology_input, h)
    if not report["region_source"]:
        # Nothing usable from the agent: fall back to replacing the template's primary case
        # so at least the payload magnitude is the user's. The rest stay template defaults,
        # which apply_agent_regions has already recorded as an assumption.
        load_id, force = _primary_force(topology_input.get("loads", []))
        if any(abs(v) > 0 for v in force):
            primary = problem.load_cases[0]
            primary.id = load_id
            primary.force_N = force
            primary.provenance = "user"
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


def volume_fraction_for_mass_cap(problem: TOProblem, max_mass_kg: float) -> tuple[float, str]:
    """Largest design volume fraction whose finished part still meets the mass cap.

    `max_part_mass_kg` used to be accepted and then never read by anything, so a 1 g cap and
    a 10 kg cap produced identical solver problems. Preserved material counts toward the
    mass too, which is why the cap can be infeasible before the design region is even used.
    """
    from .run import prepare

    density = problem.material.density_kg_m3
    if not density:
        return problem.volume_fraction, f"material {problem.material.name!r} has no density; mass cap not applied"
    mesh, masks = prepare(problem)
    v_e = mesh.elem_volume  # mm^3
    n_design = int(masks.design.sum())
    n_preserve = int(masks.preserve.sum())
    budget_mm3 = float(max_mass_kg) / (float(density) * 1e-9)
    preserve_mm3 = n_preserve * v_e
    vf_max = (budget_mm3 - preserve_mm3) / max(n_design * v_e, 1e-9)
    if vf_max < VF_BOUNDS[0]:
        raise AdapterError(
            f"max_part_mass_kg={max_mass_kg} kg is not achievable: the non-optimizable "
            f"(preserved) material alone is {preserve_mm3 * float(density) * 1e-9:.3f} kg, and the "
            f"remaining budget needs a volume fraction of {vf_max:.3f}, below the printable "
            f"minimum {VF_BOUNDS[0]}. Raise the mass limit or relax the envelope."
        )
    if vf_max >= problem.volume_fraction:
        return problem.volume_fraction, ""
    return float(vf_max), (
        f"volume fraction lowered {problem.volume_fraction:.3f} -> {vf_max:.3f} to meet the "
        f"{max_mass_kg} kg part mass cap"
    )


def estimate_topology(topology_input: dict) -> float:
    """Predicted wall time (s) for this problem, without running it."""
    opts = topology_input.get("solver_options") or {}
    h = float(opts.get("element_size_mm") or 4.0)
    candidate = topology_input.get("candidate")
    if candidate:
        task = candidate.get("task") or "cupholder"
        if task == "generated":
            problem, _ = build_generated_problem(candidate, topology_input, h, 0.2, 2.5)
        else:
            problem, _ = build_from_registry(candidate, topology_input, h, None, 2.5)
    else:
        from .from_requirements import build_from_requirements

        problem, _ = build_from_requirements(topology_input, element_size=h)
    if opts.get("max_iters"):
        problem.max_iters = int(opts["max_iters"])
    return float(estimate_cost(problem, device=str(opts.get("device") or "auto")).total_sec)


# ----------------------------------------------------------------------------- entry point
def run_topology(
    topology_input: dict,
    out_root: str | Path | None = None,
    log: Log = None,
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> dict:
    """Return a dict with the TopologyOutput fields (is_mock=False) or raise AdapterError.

    With a candidate mesh the problem is built in that mesh's frame (warm-started); without
    one it is built from the requirements alone (from scratch, uniform start).
    """
    candidate = topology_input.get("candidate")
    opts = topology_input.get("solver_options") or {}
    h = float(opts.get("element_size_mm") or 4.0)
    volfrac = opts.get("volume_fraction")
    safety_factor = float(opts.get("safety_factor") or 2.5)
    time_budget = float(opts.get("time_budget_s") or 150.0)
    device = str(opts.get("device") or "auto")
    max_iters = opts.get("max_iters")

    notes: list[str] = []
    unsupported: list[dict] = []
    requested_vf = topology_input.get("target_volume_fraction")
    if not candidate:
        from .from_requirements import build_from_requirements

        # Nothing else determines the budget on this path, so the requested fraction is used.
        problem, report = build_from_requirements(
            topology_input,
            element_size=h,
            volume_fraction=float(volfrac or requested_vf or 0.22),
            safety_factor=safety_factor,
        )
        notes.append("no candidate mesh: designed from the requirements (from scratch)")
        candidate = {"candidate_name": "from_requirements", "task": "from_requirements"}
        task = "from_requirements"
    else:
        task = candidate.get("task") or "cupholder"
    if task == "from_requirements":
        pass
    elif task == "generated":
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
    if requested_vf is not None and abs(float(requested_vf) - problem.volume_fraction) > 1e-9:
        unsupported.append({
            "requirement": "target_volume_fraction",
            "requested": float(requested_vf),
            "applied": float(problem.volume_fraction),
            "reason": (
                "the volume budget is derived from the candidate's own volume on this path, "
                "so the requested fraction was not applied"
            ),
        })

    # The accepted part-mass cap becomes an actual constraint, or an explicit failure.
    max_mass = topology_input.get("max_part_mass_kg")
    if max_mass:
        problem.volume_fraction, mass_note = volume_fraction_for_mass_cap(problem, float(max_mass))
        if mass_note:
            notes.append(mass_note)

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
        outcome = run_problem(problem, out_dir, device=device, log=log, progress=progress)
    except CostTooHigh as exc:
        raise AdapterError(f"problem too big: {exc.estimate.message}") from exc
    except RuntimeError as exc:
        if device != "cpu" and "cuda" in str(exc).lower():
            notes.append(f"CUDA failed ({str(exc)[:120]}); retried on cpu")
            outcome = run_problem(problem, out_dir, device="cpu", log=log, progress=progress)
        else:
            raise
    return topology_output(outcome, candidate, problem, notes, report, topology_input, unsupported)


def _stl_bounds(path: str) -> Optional[tuple[list[float], list[float]]]:
    try:
        import trimesh

        m = trimesh.load(path, force="mesh")
        if m.is_empty or len(m.faces) == 0:
            return None
        lo, hi = m.bounds
        return [float(v) for v in lo], [float(v) for v in hi]
    except Exception:  # noqa: BLE001
        return None


def acceptance_checks(
    outcome: RunOutcome, problem: TOProblem, topology_input: dict, frame_is_desk_edge: bool
) -> dict:
    """Does the exported mesh meet the requirements that were accepted?

    Each entry is True (checked, passed), False (checked, failed) or None (not checkable).
    None is never treated as a pass by the caller — that conflation is what let a design
    with three unverifiable checks report a clean bill of health.
    """
    s = outcome.summary
    stl = s.get("stl") or {}
    conn = s.get("connectivity") or {}
    checks: dict[str, Any] = {}
    reasons: list[str] = []

    components = stl.get("components")
    checks["single_body"] = None if components is None else bool(components == 1)
    if checks["single_body"] is False:
        reasons.append(f"exported mesh has {components} disconnected bodies")

    checks["watertight"] = stl.get("watertight")
    if checks["watertight"] is False:
        reasons.append("exported mesh is not watertight")

    checks["supports_attached"] = conn.get("supports_attached")
    if checks["supports_attached"] is False:
        reasons.append("no solid material reaches the supports")
    checks["loads_attached"] = conn.get("loads_attached")
    if checks["loads_attached"] is False:
        reasons.append(f"load region(s) {conn.get('detached_loads')} carry no material")

    # Envelope. Only meaningful in the desk-edge frame; a candidate's own frame has no
    # defined relationship to max_protrusion/width/height, so the check is not claimed.
    env = topology_input.get("envelope") or {}
    bounds = _stl_bounds(s["artifacts"]["design_stl"])
    if not env or bounds is None or not frame_is_desk_edge:
        checks["within_envelope"] = None
        if env and not frame_is_desk_edge:
            reasons.append("envelope not checked: result is in the candidate's own frame")
    else:
        lo, hi = bounds
        desk_t = float(topology_input.get("desk_thickness_mm") or 0.0)
        limits = {
            "max_protrusion_mm": (hi[0], env.get("max_protrusion_mm")),
            "max_width_mm": (max(abs(lo[1]), abs(hi[1])) * 2.0, env.get("max_width_mm")),
            "max_height_mm": (max(abs(hi[2] - desk_t), abs(desk_t - lo[2])), env.get("max_height_mm")),
        }
        over = [
            f"{k} {actual:.1f} mm > {limit} mm"
            for k, (actual, limit) in limits.items()
            if limit is not None and actual > float(limit) + 1e-6
        ]
        checks["within_envelope"] = not over
        checks["envelope_measured_mm"] = {k: round(v[0], 1) for k, v in limits.items()}
        reasons += over

    # Part mass against the accepted cap.
    cap = topology_input.get("max_part_mass_kg")
    density = problem.material.density_kg_m3
    volume_mm3 = stl.get("volume_mm3")
    if cap is None or density is None or volume_mm3 is None:
        checks["mass_within_cap"] = None
        if cap is not None and volume_mm3 is None:
            reasons.append("mass not checked: exported mesh is not watertight, so it has no volume")
    else:
        mass_kg = float(volume_mm3) * 1e-9 * float(density)
        checks["mass_within_cap"] = mass_kg <= float(cap) + 1e-9
        checks["part_mass_kg"] = round(mass_kg, 4)
        if not checks["mass_within_cap"]:
            reasons.append(f"part mass {mass_kg:.3f} kg exceeds the {cap} kg cap")

    named = ("single_body", "watertight", "supports_attached", "loads_attached", "within_envelope", "mass_within_cap")
    checks["failed"] = [k for k in named if checks.get(k) is False]
    checks["unknown"] = [k for k in named if checks.get(k) is None]
    checks["accepted"] = not checks["failed"] and not checks["unknown"]
    checks["reasons"] = reasons
    return checks


def topology_output(
    outcome: RunOutcome,
    candidate: dict,
    problem: TOProblem,
    notes: list[str],
    report: dict,
    topology_input: Optional[dict] = None,
    unsupported: Optional[list[dict]] = None,
) -> dict:
    topology_input = topology_input or {}
    unsupported = unsupported or []
    s = outcome.summary
    acceptance = acceptance_checks(
        outcome, problem, topology_input, frame_is_desk_edge=report.get("mode") == "from_requirements"
    )
    v_candidate = _mesh_volume(candidate.get("mesh_path"))
    v_design = s["stl"].get("volume_mm3")
    if v_design is None:
        v_design = float(outcome.result.rho.sum() * outcome.mesh.elem_volume)
    mass_reduction = 100.0 * (1.0 - v_design / v_candidate) if v_candidate else None
    model = (
        f"to_agent SIMP/OC on torch-fem ({s['solver_mode']}, h={problem.target_element_size:.2f} mm, "
        f"{s['masks']['n_elem']} elems, SF={problem.safety_factor})"
    )
    conn = s.get("connectivity") or {}
    if conn.get("trimmed"):
        notes.append(
            f"connectivity: the raw result had {conn['components_before']} disconnected bodies; "
            f"the largest was kept ({conn['removed_elements']} elements, "
            f"{conn['removed_fraction']:.1%} of the solid, removed) and both the STL and the "
            "post-check below describe that single body"
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
        "solver_status": (
            "converged" if s["converged"] else "max_iters_reached"
        ) + ("" if acceptance["accepted"] else "; acceptance checks did not pass"),
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
        "acceptance": acceptance,
        "assumptions": [a.model_dump() for a in problem.assumptions],
        "unsupported_requirements": unsupported,
    }
