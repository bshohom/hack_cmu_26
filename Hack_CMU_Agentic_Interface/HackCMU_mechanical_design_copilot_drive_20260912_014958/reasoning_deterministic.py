"""Deterministic fallbacks for every reasoning task.

These are the original rule-based tables. They stay as the PRODUCT-mode safety net and as
the offline/mock implementation, but they are no longer the primary path: they are overfit
to the cup-holder/hook demos by construction, which is exactly why live reasoning exists.
Developer mode refuses to use them so the gaps are visible.
"""

from __future__ import annotations

from typing import Dict, List

from agents.interaction import (
    PAYLOAD_KEYWORDS,
    TASK_BED_HANDLE,
    TASK_DESK_HOOK,
    TASK_DESK_SHELF,
    TASK_WALL_SHELF,
    classify_design_task,
    classify_hazard,
    clarification_specs_for_task,
)
from reasoning_contracts import (
    EnvelopeSemantics,
    FailureMode,
    MeasurementPlan,
    MeasurementRequest,
    MechanicsSpec,
    ReactionSurface,
    RevisionProposal,
    ScopeAssessment,
)

UNITS_BY_SUFFIX = {"_mm": "mm", "_kg": "kg", "_n": "N", "_deg": "deg"}


def _unit_for(field: str) -> str:
    for suffix, unit in UNITS_BY_SUFFIX.items():
        if field.lower().endswith(suffix):
            return unit
    return "none"


def payload_kind_from_text(text: str) -> tuple[str, str]:
    """(description, kind) from the task pack, then the keyword table."""
    lowered = (text or "").lower()
    task = classify_design_task(text)
    if task == TASK_BED_HANDLE:
        return "assist_load", "handle"
    if task == TASK_WALL_SHELF:
        return ("router" if "router" in lowered else "object"), "box"
    if task == TASK_DESK_HOOK:
        return "bag", "strap"
    if task == TASK_DESK_SHELF:
        return "object", "box"
    for keyword, description, kind in PAYLOAD_KEYWORDS:
        if keyword in lowered:
            return description, kind
    return "payload", "cylinder"


# What each fixed question actually determines. Without this the fallback plan labels a
# payload mass as "fit" and fails the same validator the live plan has to satisfy.
FIELD_AFFECTS = {
    "filled_bottle_mass_kg": ("capacity", "sets the load the part must carry"),
    "bottle_diameter_mm": ("fit", "sets the payload interface size"),
    "desk_thickness_mm": ("capacity", "sets the clamp couple arm and the jaw reaction"),
    "attachment_method": ("capacity", "decides how the load is reacted into the mount"),
    "allowed_contact_region": ("envelope", "bounds where the part may touch the mount"),
    "max_protrusion_mm": ("envelope", "bounds how far the part may extend"),
    "manufacturing_method": ("manufacturing", "sets printable feature sizes"),
    "supported_load_kg": ("capacity", "sets the load the part must carry"),
    "payload_size_mm": ("fit", "sets the payload interface size"),
    "required_reach_mm": ("envelope", "bounds how far the part may extend"),
    "wall_clearance_mm": ("envelope", "bounds clearance around the part"),
    "attachment_structure": ("capacity", "decides what the part mounts to"),
    "handle_location": ("fit", "places the grip relative to the user"),
    "mounting_region": ("envelope", "bounds where the part may touch the mount"),
    "drilling_allowed": ("capacity", "decides whether fasteners may penetrate the mount"),
}


def plan_measurements(request_text: str) -> MeasurementPlan:
    description, kind = payload_kind_from_text(request_text)
    task = classify_design_task(request_text)
    requests: List[MeasurementRequest] = []
    for spec in clarification_specs_for_task(task, description):
        field = spec["field"]
        affects, why = FIELD_AFFECTS.get(
            field, ("other", spec.get("reason") or "required for this task")
        )
        requests.append(
            MeasurementRequest(
                field=field,
                question=spec["question"],
                unit=spec.get("unit") or _unit_for(field),
                why_it_matters=why,
                affects=affects,
                priority=spec.get("priority") or "high",
            )
        )
    mount = {
        TASK_BED_HANDLE: "bed",
        TASK_WALL_SHELF: "wall",
        TASK_DESK_HOOK: "desk edge",
        TASK_DESK_SHELF: "desk edge",
    }.get(task, "desk edge")
    return MeasurementPlan(
        payload_description=description,
        payload_kind=kind,
        mount_description=mount,
        requests=requests,
        assumptions=["measurement set taken from the task questionnaire, not reasoned"],
        notes="deterministic fallback plan",
    )


def assess_scope(request_text: str) -> ScopeAssessment:
    """The hazard FLOOR. Only an unambiguous hazard is refused here.

    Ambiguous wording stays in scope at this layer so that live reasoning (or the user) can
    settle it: a keyword must never be the thing that refuses "a laptop stand on my desk".
    Because reasoning may restrict but not widen, deferring here is safe — the model can
    still rule the request out, and an unresolved hazard is reported rather than assumed away.
    """
    signal = classify_hazard(request_text)
    if signal.verdict == "refuse":
        return ScopeAssessment(in_scope=False, hazard_class=signal.hazard_class, reasons=list(signal.reasons))
    if signal.verdict == "ambiguous":
        return ScopeAssessment(
            in_scope=True,
            hazard_class=signal.hazard_class,
            reasons=list(signal.reasons) + [f"unresolved: {signal.question}"],
        )
    return ScopeAssessment(in_scope=True, hazard_class="none", reasons=[])


def analyze_mechanics(attachment_method: str | None, payload_kind: str) -> MechanicsSpec:
    method = (attachment_method or "clamp").lower()
    if method in ("clamp", "screws", "bolt", "bolts"):
        surfaces = [
            ReactionSurface(name="top_jaw", description="upper arm underside on the desk top",
                            reaction_type="bearing", restrained_dofs=["x", "y", "z"]),
            ReactionSurface(name="bottom_jaw", description="lower arm top face on the desk underside",
                            reaction_type="bearing", restrained_dofs=["x", "y", "z"]),
        ]
        modes = [
            FailureMode(name="jaw_couple", mechanism="moment reacted as a force couple across the desk",
                        governing_quantity="jaw reaction force", check_id="jaw_couple"),
            FailureMode(name="bearing", mechanism="contact pressure crushing the desk surface",
                        governing_quantity="bearing pressure", check_id="bearing_pressure"),
            FailureMode(name="material", mechanism="von Mises stress vs printed yield",
                        governing_quantity="max von Mises", check_id="material_yield"),
        ]
        scheme = "edge_clamp"
    else:
        surfaces = [ReactionSurface(name="base", description="footprint bonded to the surface",
                                    reaction_type="bonded", restrained_dofs=["x", "y", "z"])]
        modes = [FailureMode(name="material", mechanism="von Mises stress vs printed yield",
                             governing_quantity="max von Mises", check_id="material_yield")]
        scheme = "free_standing"
    return MechanicsSpec(
        support_scheme=scheme,
        reaction_surfaces=surfaces,
        load_path=["payload", "contact patch", "structure", "mount"],
        failure_modes=modes,
        envelope=EnvelopeSemantics(),
        required_measurements=["desk_thickness_mm", "max_protrusion_mm"],
        notes="deterministic fallback mechanics",
    )


def revise(failed_checks: List[Dict]) -> RevisionProposal:
    names = {str(c.get("name", "")) for c in failed_checks}
    if "envelope_fit" in names:
        return RevisionProposal(
            action="switch_to_generated",
            target="geometry_source",
            rationale="the candidate does not fit the stated envelope; design one to the requirements instead",
            user_question="Raise the allowed protrusion, or design a part to your current limit?",
        )
    return RevisionProposal(
        action="change_parameter",
        target="support_thickness_mm",
        value=2.0,
        rationale="deterministic loop: stiffen the supports",
    )
