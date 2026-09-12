"""User envelope as a hard spatial constraint on design elements.

Limits are in desk_edge_frame:
  protrusion  0 <= x <= max_protrusion   (x < 0 is the clamp, not protrusion)
  width       y in [-max_width/2, +max_width/2]
  height      z in a window of length max_height that covers the desk slab when possible

The complementary region is void. Attachment/support boxes are carved out of that void
so a required mount is not deleted; a conflict is reported instead of silently expanding
the user limits.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts import DifferenceRegion, HalfSpaceRegion, TOProblem, UnionRegion
from ..regions import bounds as region_bounds
from .run import prepare


def _fit_height_window(z_lo: float, z_hi: float, max_h: float, desk_t: float) -> tuple[float, float]:
    """A max_h-tall window that prefers to keep the desk slab [min(0,desk_t), max(0,desk_t)]."""
    if z_hi - z_lo <= max_h + 1e-9:
        return z_lo, z_hi
    prefer_lo, prefer_hi = min(0.0, desk_t), max(0.0, desk_t)
    extra = max_h - (prefer_hi - prefer_lo)
    if extra < 0:
        return prefer_lo, prefer_lo + max_h
    down = min(max(0.0, prefer_lo - z_lo), extra)
    up = extra - down
    return prefer_lo - down, prefer_hi + up


def allowed_envelope_desk_edge(
    topology_input: dict,
    domain_desk: tuple[list[float], list[float]],
) -> dict[str, Optional[float]]:
    env = topology_input.get("envelope") or {}
    desk_t = float(topology_input.get("desk_thickness_mm") or 0.0)
    lo, hi = domain_desk
    allowed: dict[str, Optional[float]] = {
        "x_lo": None,
        "x_hi": None,
        "y_lo": None,
        "y_hi": None,
        "z_lo": None,
        "z_hi": None,
    }
    if env.get("max_protrusion_mm") is not None:
        allowed["x_hi"] = float(env["max_protrusion_mm"])
    if env.get("max_width_mm") is not None:
        half = float(env["max_width_mm"]) / 2.0
        allowed["y_lo"] = -half
        allowed["y_hi"] = half
    if env.get("max_height_mm") is not None:
        z_lo, z_hi = _fit_height_window(float(lo[2]), float(hi[2]), float(env["max_height_mm"]), desk_t)
        allowed["z_lo"] = z_lo
        allowed["z_hi"] = z_hi
    return allowed


def _box_conflicts(lo: list[float], hi: list[float], allowed: dict[str, Optional[float]]) -> list[str]:
    reasons: list[str] = []
    if allowed.get("x_hi") is not None and hi[0] > float(allowed["x_hi"]) + 1e-6:
        reasons.append(f"x_max {hi[0]:.1f} > max_protrusion {allowed['x_hi']}")
    if allowed.get("y_lo") is not None and lo[1] < float(allowed["y_lo"]) - 1e-6:
        reasons.append(f"y_min {lo[1]:.1f} < {allowed['y_lo']}")
    if allowed.get("y_hi") is not None and hi[1] > float(allowed["y_hi"]) + 1e-6:
        reasons.append(f"y_max {hi[1]:.1f} > {allowed['y_hi']}")
    if allowed.get("z_lo") is not None and lo[2] < float(allowed["z_lo"]) - 1e-6:
        reasons.append(f"z_min {lo[2]:.1f} < {allowed['z_lo']}")
    if allowed.get("z_hi") is not None and hi[2] > float(allowed["z_hi"]) + 1e-6:
        reasons.append(f"z_max {hi[2]:.1f} > {allowed['z_hi']}")
    return reasons


def _entirely_outside(lo: list[float], hi: list[float], allowed: dict[str, Optional[float]]) -> bool:
    if allowed.get("x_hi") is not None and lo[0] > float(allowed["x_hi"]) + 1e-6:
        return True
    if allowed.get("y_hi") is not None and lo[1] > float(allowed["y_hi"]) + 1e-6:
        return True
    if allowed.get("y_lo") is not None and hi[1] < float(allowed["y_lo"]) - 1e-6:
        return True
    if allowed.get("z_hi") is not None and lo[2] > float(allowed["z_hi"]) + 1e-6:
        return True
    if allowed.get("z_lo") is not None and hi[2] < float(allowed["z_lo"]) - 1e-6:
        return True
    return False


def _halfspaces_desk_edge(allowed: dict[str, Optional[float]]) -> list[HalfSpaceRegion]:
    planes: list[HalfSpaceRegion] = []
    if allowed.get("x_hi") is not None:
        planes.append(HalfSpaceRegion(point=(float(allowed["x_hi"]), 0.0, 0.0), normal=(1.0, 0.0, 0.0)))
    if allowed.get("y_hi") is not None:
        planes.append(HalfSpaceRegion(point=(0.0, float(allowed["y_hi"]), 0.0), normal=(0.0, 1.0, 0.0)))
    if allowed.get("y_lo") is not None:
        planes.append(HalfSpaceRegion(point=(0.0, float(allowed["y_lo"]), 0.0), normal=(0.0, -1.0, 0.0)))
    if allowed.get("z_hi") is not None:
        planes.append(HalfSpaceRegion(point=(0.0, 0.0, float(allowed["z_hi"])), normal=(0.0, 0.0, 1.0)))
    if allowed.get("z_lo") is not None:
        planes.append(HalfSpaceRegion(point=(0.0, 0.0, float(allowed["z_lo"])), normal=(0.0, 0.0, -1.0)))
    return planes


def _to_mesh_halfspace(region: HalfSpaceRegion, desk_to_mesh: Optional[list]) -> HalfSpaceRegion:
    if desk_to_mesh is None:
        return region
    from .agentic import transform_point, transform_vec

    return HalfSpaceRegion(
        point=tuple(transform_point(desk_to_mesh, region.point)),  # type: ignore[arg-type]
        normal=tuple(transform_vec(desk_to_mesh, region.normal)),  # type: ignore[arg-type]
    )


def _bounds_desk_edge(region, desk_to_mesh: Optional[list]) -> Optional[tuple[list[float], list[float]]]:
    box = region_bounds(region)
    if box is None:
        return None
    lo, hi = [float(v) for v in box[0]], [float(v) for v in box[1]]
    if desk_to_mesh is None:
        return lo, hi
    from .agentic import _transform_bounds, invert_rigid

    return _transform_bounds((lo, hi), invert_rigid(desk_to_mesh))


def envelope_void_region(
    topology_input: dict,
    problem: TOProblem,
    desk_to_mesh: Optional[list],
) -> tuple[Optional[Any], dict]:
    """Forbidden half-spaces in the problem/mesh frame, minus required supports."""
    env = topology_input.get("envelope") or {}
    if not any(env.get(k) is not None for k in ("max_protrusion_mm", "max_width_mm", "max_height_mm")):
        return None, {"applied": False}

    from .agentic import _transform_bounds, invert_rigid

    domain = problem.design_domain
    if domain is None:
        return None, {"applied": False, "reason": "no design_domain"}
    if desk_to_mesh is not None:
        domain_desk = _transform_bounds((list(domain.min), list(domain.max)), invert_rigid(desk_to_mesh))
    else:
        domain_desk = (list(domain.min), list(domain.max))
    allowed = allowed_envelope_desk_edge(topology_input, domain_desk)
    planes = [_to_mesh_halfspace(p, desk_to_mesh) for p in _halfspaces_desk_edge(allowed)]
    if not planes:
        return None, {"applied": False, "allowed_desk_edge_mm": allowed}

    conflicts: list[dict] = []
    hard = False
    for s in problem.supports:
        box = _bounds_desk_edge(s.region, desk_to_mesh)
        if box is None:
            continue
        reasons = _box_conflicts(box[0], box[1], allowed)
        if not reasons:
            continue
        entirely = _entirely_outside(box[0], box[1], allowed)
        conflicts.append({
            "kind": "support",
            "id": s.id,
            "reasons": reasons,
            "entirely_outside": entirely,
        })
        hard = hard or entirely
    for i, region in enumerate(problem.preserve):
        box = _bounds_desk_edge(region, desk_to_mesh)
        if box is None:
            continue
        reasons = _box_conflicts(box[0], box[1], allowed)
        if reasons:
            conflicts.append({
                "kind": "preserve",
                "id": f"preserve_{i}",
                "reasons": reasons,
                "entirely_outside": _entirely_outside(box[0], box[1], allowed),
            })

    forbidden: Any = planes[0] if len(planes) == 1 else UnionRegion(regions=planes)
    if problem.supports:
        protect: Any = (
            problem.supports[0].region
            if len(problem.supports) == 1
            else UnionRegion(regions=[s.region for s in problem.supports])
        )
        void = DifferenceRegion(a=forbidden, b=protect)
    else:
        void = forbidden
    return void, {
        "applied": True,
        "allowed_desk_edge_mm": {k: (round(v, 3) if v is not None else None) for k, v in allowed.items()},
        "conflicts": conflicts,
        "hard_infeasible": hard,
        "formula": {
            "protrusion": "void x > max_protrusion; x < 0 remains allowed",
            "width": "void |y| > max_width/2",
            "height": "void z outside a max_height window covering the desk slab",
        },
    }


def apply_envelope_constraint(
    problem: TOProblem,
    topology_input: dict,
    desk_to_mesh: Optional[list],
) -> dict:
    """Append the envelope void. Does not count elements (grid may still coarsen)."""
    void, info = envelope_void_region(topology_input, problem, desk_to_mesh)
    if void is None:
        return info
    problem.void = list(problem.void) + [void]
    info["void_index"] = len(problem.void) - 1
    return info


def count_envelope_mask(problem: TOProblem, info: dict) -> dict:
    """Design-element counts on the current grid, with and without the envelope void."""
    if not info.get("applied"):
        return info
    idx = info.get("void_index")
    voids = list(problem.void)
    if idx is None or idx >= len(voids):
        return info
    problem.void = voids[:idx] + voids[idx + 1 :]
    _, before = prepare(problem)
    problem.void = voids
    _, after = prepare(problem)
    info["design_elems_before"] = int(before.design.sum())
    info["design_elems_after"] = int(after.design.sum())
    info["preserve_elems_before"] = int(before.preserve.sum())
    info["preserve_elems_after"] = int(after.preserve.sum())
    info["void_elems_before"] = int(before.void.sum())
    info["void_elems_after"] = int(after.void.sum())
    return info
