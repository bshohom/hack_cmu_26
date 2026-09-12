"""Deterministic fallbacks for every reasoning task.

These are the original rule-based tables. They stay as the PRODUCT-mode safety net and as
the offline/mock implementation, but they are no longer the primary path: they are overfit
to the cup-holder/hook demos by construction, which is exactly why live reasoning exists.
Developer mode refuses to use them so the gaps are visible.
"""

from __future__ import annotations

from typing import Dict, List

from agents.interaction import PAYLOAD_KEYWORDS, REQUIRED_FIELDS, SIZE_QUESTIONS, UNSUPPORTED_PATTERNS
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
    """(description, kind) from the keyword table — the overfit path, kept as fallback."""
    lowered = (text or "").lower()
    for keyword, description, kind in PAYLOAD_KEYWORDS:
        if keyword in lowered:
            return description, kind
    return "payload", "cylinder"


def plan_measurements(request_text: str) -> MeasurementPlan:
    description, kind = payload_kind_from_text(request_text)
    requests: List[MeasurementRequest] = []
    for field, question, priority in REQUIRED_FIELDS:
        if field == "bottle_diameter_mm":
            question = SIZE_QUESTIONS.get(kind, question)
        requests.append(
            MeasurementRequest(
                field=field,
                question=question.format(payload=description),
                unit=_unit_for(field),
                why_it_matters="required by the deterministic questionnaire",
                affects="fit",
                priority=priority,
            )
        )
    return MeasurementPlan(
        payload_description=description,
        payload_kind=kind,
        mount_description="desk edge",
        requests=requests,
        assumptions=["measurement set taken from the fixed questionnaire, not reasoned"],
        notes="deterministic fallback plan",
    )


def assess_scope(request_text: str) -> ScopeAssessment:
    lowered = (request_text or "").lower()
    for pattern, reason in UNSUPPORTED_PATTERNS:
        if pattern in lowered:
            return ScopeAssessment(in_scope=False, hazard_class="other", reasons=[reason])
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
