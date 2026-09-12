"""Build a TOProblem from requirements alone — no warm-start mesh, no candidate STL.

This is the "design from scratch" path: the design domain is the allowed envelope, the desk
(or ground) is a void/support, the payload contact patch is the load, and SIMP decides the
shape. Used whenever no candidate geometry exists or warm-start generation failed, so the
workflow always produces a real printable result instead of a placeholder.

Frame: desk_edge_frame — desk underside z=0, desk top z=desk_thickness, desk front edge at
x=0 with the desk occupying x<0; the part protrudes toward +x; +Z is up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..contracts import Assumption, BoxRegion, CapsuleRegion, LoadCase, Region, Support, TOProblem
from .agent_regions import load_cases_from_input, supports_from_input
from .materials import material_from_name

GRIP_DEPTH_MM = 55.0  # how far the clamp reaches back over/under the desk
DEFAULT_WIDTH_MM = 60.0
CONTACT_MM = 8.0  # thickness of preserved contact skins


@dataclass
class Envelope:
    max_protrusion_mm: float
    max_width_mm: float
    max_height_mm: float


def _envelope(topology_input: dict, payload_size: float) -> Envelope:
    env = topology_input.get("envelope") or {}
    dom = topology_input.get("design_domain") or {}
    prot = env.get("max_protrusion_mm") or dom.get("length_mm") or 120.0
    width = env.get("max_width_mm") or dom.get("width_mm") or max(payload_size * 1.6, DEFAULT_WIDTH_MM)
    height = env.get("max_height_mm") or dom.get("height_mm") or 80.0
    return Envelope(float(prot), float(width), float(height))


def _load_geometry(topology_input: dict, env: Envelope, desk_t: float, payload_size: float) -> tuple[str, float, float, tuple[float, float]]:
    """(kind, load_x_centre, load_z, (patch_x, patch_y)) of the payload contact patch."""
    kind = (topology_input.get("payload_kind") or "cylinder").lower()
    regions = topology_input.get("load_regions") or []
    pos = regions[0].get("position_mm") if regions and isinstance(regions[0], dict) else None
    if kind == "strap":
        # A hook: the strap rests on a horizontal arm hanging below the desk edge.
        x = 0.75 * env.max_protrusion_mm
        z = -max(25.0, 0.35 * env.max_height_mm)
        patch = (max(payload_size, 20.0), max(payload_size, 20.0))
    elif kind == "box":
        # A shelf: the payload sits on a raised flat platform.
        x = 0.45 * env.max_protrusion_mm
        z = desk_t + max(40.0, 0.5 * env.max_height_mm)
        patch = (max(payload_size, 40.0), max(payload_size, 40.0))
    elif kind == "cylinder":
        # A cup/bottle holder: the payload rests on a floor above the desk plane.
        x = 0.8 * env.max_protrusion_mm
        z = desk_t + 10.0
        patch = (max(payload_size, 40.0), max(payload_size, 40.0))
    else:
        # Generic support/load: use the structured load region when present.
        x = 0.55 * env.max_protrusion_mm
        z = desk_t + 0.35 * env.max_height_mm
        patch = (max(payload_size, 24.0), max(payload_size, 24.0))
    if pos and len(pos) == 3 and any(abs(float(v)) > 1e-6 for v in pos):
        x, z = float(pos[0]), float(pos[2])
    return kind, x, z, patch


def structure_warm_start(structure: Optional[dict], min_radius: float) -> tuple[list, str]:
    """Rasterize the coarse structural members as capsules: the warm-start density field.

    The agentic workflow's structure stage already proposes nodes and beams between the
    mounts and the payload (and its review loop tunes their thickness). Seeding SIMP with
    that load path beats a uniform start and makes the structural stage feed the optimizer.
    """
    if not structure:
        return [], ""
    nodes = {n["id"]: n.get("position_mm") for n in (structure.get("nodes") or []) if n.get("id")}
    members = structure.get("members") or []
    params = structure.get("parameters") or {}
    radius = max(float(params.get("support_thickness_mm") or 0.0) / 2.0, min_radius)
    capsules: list[Region] = []
    for m in members:
        a, b = nodes.get(m.get("start_node_id")), nodes.get(m.get("end_node_id"))
        if not a or not b:
            continue
        if all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b)):
            continue
        capsules.append(CapsuleRegion(a=tuple(float(v) for v in a), b=tuple(float(v) for v in b), radius=radius))
    if not capsules:
        return [], ""
    note = (
        f"warm start from {len(capsules)} structural member(s) of the coarse layout "
        f"(radius {radius:.1f} mm from support_thickness_mm)"
    )
    return capsules, note


def build_from_requirements(
    topology_input: dict,
    element_size: float = 4.0,
    volume_fraction: float = 0.22,
    safety_factor: float = 2.5,
) -> tuple[TOProblem, dict]:
    loads = topology_input.get("loads") or []
    if not loads:
        raise ValueError("cannot design from requirements without a load case")
    lc = loads[0]
    force = tuple(float(v) for v in lc.get("force_N", (0.0, 0.0, -1.0)))
    load_id = str(lc.get("load_case_id") or lc.get("name") or "static_gravity")

    desk_t = float(topology_input.get("desk_thickness_mm") or 20.0)
    payload_size = float(topology_input.get("payload_size_mm") or 40.0)
    env = _envelope(topology_input, payload_size)
    kind, load_x, load_z, patch = _load_geometry(topology_input, env, desk_t, payload_size)
    method = (topology_input.get("attachment_method") or "clamp").lower()
    clamped = method in ("clamp", "screws", "bolt", "bolts")
    h = element_size
    half_w = env.max_width_mm / 2.0
    half_px, half_py = patch[0] / 2.0, patch[1] / 2.0
    assumptions: list[Assumption] = []

    # Boundary conditions come from the agent when it supplied them. The archetype layouts
    # below are a fallback for when it did not, and say so in `assumptions`.
    supports, sup_assumptions = supports_from_input(topology_input, h)
    assumptions += sup_assumptions
    agent_supports = bool(supports)
    agent_loads, load_assumptions = load_cases_from_input(topology_input, h)
    assumptions += load_assumptions

    # Design domain: the envelope is a hard bound, not a suggestion. The grip region behind
    # the desk edge (x < 0) is not "protrusion", so max_protrusion only bounds +x.
    x_lo = -GRIP_DEPTH_MM if clamped else -10.0
    x_hi = env.max_protrusion_mm
    z_lo = min(load_z - h, -h) if clamped else 0.0
    z_hi = max(desk_t + h, load_z + h)
    # The archetype's load_z only describes the archetype. When the agent placed the regions
    # itself, the domain has to contain those instead.
    for region in [s.region for s in supports] + [c.region for c in agent_loads]:
        z_lo = min(z_lo, region.min[2] - h)
        z_hi = max(z_hi, region.max[2] + h)
    # max_height_mm read as the part's z-extent measured from the desk top surface.
    z_hi_capped = min(z_hi, desk_t + env.max_height_mm)
    z_lo_capped = max(z_lo, desk_t - env.max_height_mm)
    if z_hi_capped < z_hi - 1e-9 or z_lo_capped > z_lo + 1e-9:
        assumptions.append(
            Assumption(
                field="design_domain.z",
                value=f"[{z_lo_capped:.1f}, {z_hi_capped:.1f}] mm",
                basis=(
                    f"max_height_mm={env.max_height_mm} applied as z-extent about the desk "
                    f"top (z={desk_t} mm); the unbounded span would have been "
                    f"[{z_lo:.1f}, {z_hi:.1f}] mm"
                ),
            )
        )
    z_lo, z_hi = z_lo_capped, z_hi_capped
    if z_hi - z_lo < 2 * h:  # a domain thinner than two elements cannot be meshed
        raise ValueError(
            f"envelope max_height_mm={env.max_height_mm} leaves a {z_hi - z_lo:.1f} mm tall "
            f"domain, below the {2 * h:.1f} mm minimum for element size {h} mm"
        )
    domain = BoxRegion(min=(x_lo, -half_w, z_lo), max=(x_hi, half_w, z_hi))

    void: list = []
    preserve: list = []
    if clamped:
        # The desk itself is forbidden material.
        void.append(BoxRegion(min=(x_lo - 2 * h, -half_w - 2 * h, 0.0), max=(0.0, half_w + 2 * h, desk_t)))
    if agent_supports:
        # Keep a solid skin over each attachment patch so the mount has material to react into.
        for s in supports:
            preserve.append(s.region)
    elif clamped:
        # The jaws grip the desk's top and underside.
        supports.append(Support(id="top_jaw", region=BoxRegion(min=(x_lo + 2.0, -half_w, desk_t), max=(-2.0, half_w, desk_t + h)), provenance="assumed"))
        supports.append(Support(id="bottom_jaw", region=BoxRegion(min=(x_lo + 2.0, -half_w, -h), max=(-2.0, half_w, 0.0)), provenance="assumed"))
        preserve.append(BoxRegion(min=(x_lo, -half_w, desk_t), max=(-1.0, half_w, desk_t + CONTACT_MM)))
        preserve.append(BoxRegion(min=(x_lo, -half_w, -CONTACT_MM), max=(-1.0, half_w, 0.0)))
        assumptions.append(
            Assumption(
                field="supports",
                value=f"top_jaw + bottom_jaw gripping {GRIP_DEPTH_MM} mm behind the desk edge",
                basis=f"no fixed_regions supplied; clamp layout assumed from attachment_method={method!r}",
            )
        )
    else:
        # Free-standing: the part is bonded to the desk top over its footprint.
        supports.append(Support(id="base", region=BoxRegion(min=(x_lo, -half_w, z_lo - h), max=(x_hi * 0.6, half_w, z_lo + h)), provenance="assumed"))
        preserve.append(BoxRegion(min=(x_lo, -half_w, z_lo), max=(x_hi * 0.6, half_w, z_lo + CONTACT_MM)))
        assumptions.append(
            Assumption(
                field="supports",
                value="base bonded to the desk top over 60% of the footprint",
                basis=(
                    f"no fixed_regions supplied; free-standing layout assumed from "
                    f"attachment_method={method!r}. Fully fixed in x/y/z, so tipping and "
                    f"sliding are not modelled"
                ),
            )
        )

    # The payload contact patch is preserved and loaded.
    patch_box = BoxRegion(min=(load_x - half_px, -half_py, load_z - h), max=(load_x + half_px, half_py, load_z + CONTACT_MM))
    preserve.append(patch_box)
    if kind == "cylinder":
        # Keep the payload's own volume clear above its floor.
        void.append(BoxRegion(min=(load_x - half_px, -half_py, load_z + CONTACT_MM + 0.5), max=(load_x + half_px, half_py, z_hi + h)))
    elif kind == "box":
        void.append(BoxRegion(min=(load_x - half_px, -half_py, load_z + CONTACT_MM + 0.5), max=(x_hi + h, half_py, z_hi + h)))

    material, mat_note = material_from_name(topology_input.get("material"))
    # Every agent load case, each at its own named region. Only when the agent supplied no
    # usable load regions does the archetype place a single patch and invent a second case.
    load_cases = agent_loads
    if not load_cases:
        load_cases = [
            LoadCase(id=load_id, region=BoxRegion(min=(load_x - half_px, -half_py, load_z - h), max=(load_x + half_px, half_py, load_z + CONTACT_MM)), force_N=force, provenance="user"),
        ]
        mag = max(abs(v) for v in force) or 1.0
        if kind == "strap":
            # Outward pull keeps the retaining lip structural instead of dead weight.
            load_cases.append(LoadCase(id="outward_pull", region=load_cases[0].region, force_N=(0.4 * mag, 0.0, 0.0), weight=0.6, provenance="assumed"))
        else:
            load_cases.append(LoadCase(id="side_bump", region=load_cases[0].region, force_N=(0.0, 0.3 * mag, 0.0), weight=0.4, provenance="assumed"))
        assumptions.append(
            Assumption(
                field="load_cases",
                value=f"{load_cases[0].id} at (x={load_x:.1f}, z={load_z:.1f}) plus {load_cases[1].id}",
                basis=(
                    f"no usable load_regions supplied; patch placed by the {kind!r} archetype "
                    f"and a secondary case sized at {load_cases[1].force_N} N from the primary "
                    f"magnitude"
                ),
            )
        )

    warm_start, warm_note = structure_warm_start(topology_input.get("structure"), min_radius=1.5 * h)
    problem = TOProblem(
        material=material,
        design_domain=domain,
        warm_start=warm_start,  # coarse members if available, else a uniform start
        preserve=preserve,
        void=void,
        supports=supports,
        load_cases=load_cases,
        safety_factor=safety_factor,
        volume_fraction=volume_fraction,
        target_element_size=h,
        filter_radius=1.5 * h,
        max_iters=60,
        assumptions=assumptions,
        notes=(
            f"Designed from requirements only (no candidate mesh): "
            f"{'clamped' if clamped else 'free-standing'}, desk {desk_t} mm, envelope "
            f"{env.max_protrusion_mm}x{env.max_width_mm}x{env.max_height_mm} mm. {warm_note} {mat_note}".strip()
        ),
    )
    report = {
        "mode": "from_requirements",
        "warm_start": warm_note or "uniform (no structural members supplied)",
        "supports_from": "agent fixed_regions" if agent_supports else f"{method!r} archetype",
        "loads_from": "agent load_regions" if not load_assumptions and load_cases else "archetype",
        "payload_kind": kind,
        "attachment": method,
        "desk_thickness_mm": desk_t,
        "envelope_mm": [env.max_protrusion_mm, env.max_width_mm, env.max_height_mm],
        "load_patch_mm": {"x": load_x, "z": load_z, "size": list(patch)},
        "assumptions": [a.model_dump() for a in assumptions],
    }
    return problem, report
