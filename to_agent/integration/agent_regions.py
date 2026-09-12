"""Turn the agent's structured regions into solver supports and load cases.

The interface already decides where the part is held and where it is loaded
(`StructureOutput.attachment_regions` / `load_regions` / `load_cases`, forwarded on
`TopologyInput` as `fixed_regions` / `load_regions` / `loads`). Until now the solver read
none of it and substituted a `payload_kind` archetype, so a better reasoning layer changed
nothing downstream. This module is the consumer: boundary conditions come from the agent
when it supplies them, and any fallback is recorded as an `Assumption` rather than applied
silently.

Region shapes (interface `schemas.py`):
  AttachmentRegion  {name, position_mm, normal, area_mm2}   - a patch, given as a point
  LoadRegion        {name, position_mm, direction}          - a point
  LoadCase          {load_case_id, name, region_name, force_N} - refers to a LoadRegion by name

Both region kinds are points, so a patch box is synthesized around each one: square of
side sqrt(area_mm2) across the normal, one element thick along it.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts import Assumption, BoxRegion, LoadCase, Support

DEFAULT_PATCH_MM = 20.0  # side of a support/load patch when no area is given
MIN_PATCH_MM = 8.0


def _vec(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        out = tuple(float(v) for v in value)
        return out if len(out) == 3 else fallback
    except (TypeError, ValueError):
        return fallback


def patch_box(
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    area_mm2: Optional[float],
    h: float,
    side_mm: Optional[float] = None,
) -> BoxRegion:
    """A thin square patch centred on `position`, facing `normal`.

    The patch is one element thick along the normal's dominant axis and `side` wide across
    it. Normals in practice are axis-aligned; an oblique normal falls back to its dominant
    axis, which keeps the patch a valid box instead of failing.
    """
    if side_mm is None:
        side_mm = (area_mm2 ** 0.5) if area_mm2 and area_mm2 > 0 else DEFAULT_PATCH_MM
    side = max(float(side_mm), MIN_PATCH_MM)
    axis = max(range(3), key=lambda i: abs(normal[i]))
    half = [side / 2.0, side / 2.0, side / 2.0]
    half[axis] = max(h, 1.0) / 2.0
    lo = tuple(position[i] - half[i] for i in range(3))
    hi = tuple(position[i] + half[i] for i in range(3))
    return BoxRegion(min=lo, max=hi)


def _position(raw: dict) -> Optional[tuple[float, float, float]]:
    """The region's position, or None when it carries no spatial information.

    `position_mm` defaults to (0, 0, 0) in the interface schema, so an absent key is not the
    origin — it is a named placeholder with no geometry. Treating that default as intent puts
    supports and loads at the frame origin, which is inside the desk. Absence is not data.
    """
    if "position_mm" not in raw or raw.get("position_mm") is None:
        return None
    try:
        out = tuple(float(v) for v in raw["position_mm"])
    except (TypeError, ValueError):
        return None
    return out if len(out) == 3 else None


def supports_from_input(topology_input: dict, h: float) -> tuple[list[Support], list[Assumption]]:
    """Supports from `fixed_regions`. Empty list means the caller must fall back.

    All-or-nothing: if any supplied region lacks a position, the whole set is rejected.
    Mounting a part on a mix of real intent and guessed placeholders is worse than either.
    """
    raw_regions = [r for r in (topology_input.get("fixed_regions") or []) if isinstance(r, dict)]
    if not raw_regions:
        return [], []
    if any(_position(r) is None for r in raw_regions):
        named = ", ".join(str(r.get("name") or "?") for r in raw_regions if _position(r) is None)
        return [], [
            Assumption(
                field="supports",
                value="archetype layout",
                basis=f"attachment region(s) {named} carry no position_mm; agent regions unusable",
            )
        ]
    supports: list[Support] = []
    assumptions: list[Assumption] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_regions):
        pos = _position(raw) or (0.0, 0.0, 0.0)
        normal = _vec(raw.get("normal"), (0.0, 0.0, 1.0))
        area = raw.get("area_mm2")
        name = str(raw.get("name") or f"mount_{i}")
        while name in seen:  # TOProblem requires unique support/load ids
            name = f"{name}_{i}"
        seen.add(name)
        if not area:
            assumptions.append(
                Assumption(
                    field=f"supports[{name}].area_mm2",
                    value=f"{DEFAULT_PATCH_MM} mm square patch",
                    basis="attachment region carried no area; patch size assumed",
                )
            )
        supports.append(
            Support(id=name, region=patch_box(pos, normal, area, h), provenance="derived")
        )
    return supports, assumptions


def merge_load_cases(
    agent_cases: list[LoadCase], template_cases: list[LoadCase]
) -> tuple[list[LoadCase], list[LoadCase]]:
    """User/agent primary loads win; template retention/stabilization cases are kept.

    Agent replacement used to drop every template case, including hook `tip_retention`
    that exists to keep a preserved tip connected. side_swing and other unmarked
    template cases are still discarded.
    """
    merged = list(agent_cases)
    seen = {case.id for case in merged}
    kept: list[LoadCase] = []
    for case in template_cases:
        if case.id in seen:
            continue
        if case.role == "retention":
            merged.append(case)
            kept.append(case)
    return merged, kept


def apply_agent_regions(problem, topology_input: dict, h: float) -> list[str]:
    """Override a template/candidate problem's boundary conditions with the agent's.

    Templates are a warm-start and geometry source; they are not the authority on where the
    part is held or loaded. Whatever the template guessed stays only where the agent said
    nothing, and every remaining template guess is recorded as an assumption.
    """
    notes: list[str] = []
    template_loads = [case.model_copy(deep=True) for case in problem.load_cases]
    supports, sup_assumptions = supports_from_input(topology_input, h)
    if supports:
        problem.supports = supports
        notes.append(f"supports from agent fixed_regions ({len(supports)})")
    else:
        problem.assumptions += sup_assumptions or [
            Assumption(
                field="supports",
                value=", ".join(s.id for s in problem.supports),
                basis="no usable fixed_regions supplied; template supports kept",
            )
        ]
    cases, load_assumptions = load_cases_from_input(topology_input, h)
    if cases:
        merged, kept = merge_load_cases(cases, template_loads)
        problem.load_cases = merged
        notes.append(f"load cases from agent load_regions ({len(cases)})")
        if kept:
            notes.append(
                "kept template retention loads (" + ", ".join(c.id for c in kept) + ")"
            )
    else:
        problem.assumptions += load_assumptions or [
            Assumption(
                field="load_cases",
                value=", ".join(c.id for c in problem.load_cases),
                basis="no usable load_regions supplied; template load cases kept",
            )
        ]
    return notes


def load_cases_from_input(topology_input: dict, h: float) -> tuple[list[LoadCase], list[Assumption]]:
    """Every agent load case, each positioned at its own named load region.

    This is what makes a payload change move *all* the loads: each case carries its own
    force from the resolved requirements, instead of only case 0 being overwritten.
    """
    regions = {
        str(r.get("name")): r
        for r in (topology_input.get("load_regions") or [])
        if isinstance(r, dict) and r.get("name")
    }
    cases: list[LoadCase] = []
    assumptions: list[Assumption] = []
    seen: set[str] = set()
    for i, raw in enumerate(topology_input.get("loads") or []):
        if not isinstance(raw, dict):
            continue
        force = _vec(raw.get("force_N"), (0.0, 0.0, 0.0))
        if not any(abs(v) > 0.0 for v in force):
            continue  # a zero load case constrains nothing
        case_id = str(raw.get("load_case_id") or raw.get("name") or f"load_{i}")
        while case_id in seen:
            case_id = f"{case_id}_{i}"
        seen.add(case_id)
        region = regions.get(str(raw.get("region_name")))
        if region is None:
            # No matching load region: the caller must place this case itself.
            return [], [
                Assumption(
                    field="load_cases",
                    value="archetype placement",
                    basis=(
                        f"load case {case_id!r} names region "
                        f"{raw.get('region_name')!r}, which is not in load_regions"
                    ),
                )
            ]
        pos = _position(region)
        if pos is None:
            return [], [
                Assumption(
                    field="load_cases",
                    value="archetype placement",
                    basis=(
                        f"load region {region.get('name')!r} carries no position_mm, so load "
                        f"case {case_id!r} cannot be placed from agent data"
                    ),
                )
            ]
        # A load patch has no area in the schema; the direction is the surface it pushes on.
        normal = _vec(region.get("direction"), (0.0, 0.0, -1.0))
        cases.append(
            LoadCase(
                id=case_id,
                region=patch_box(pos, normal, None, h),
                force_N=force,
                provenance="user",
                role="primary",
            )
        )
    return cases, assumptions
