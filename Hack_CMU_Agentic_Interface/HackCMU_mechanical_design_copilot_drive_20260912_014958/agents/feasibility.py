"""Deterministic requirement / geometry feasibility gate.

Catches obviously impossible combinations before STRUCTURE.
Not an LLM and not a detailed engineering formula.
"""

from __future__ import annotations

from typing import List, Optional

from schemas import (
    ClarificationQuestion,
    DesignFeasibilityResult,
    FeasibilityViolation,
    GeometryOutput,
    UserRequirements,
)


def check_design_feasibility(
    requirements: UserRequirements,
    geometry: GeometryOutput,
) -> DesignFeasibilityResult:
    diameter = _first_number(
        geometry.payload_object.bottle_diameter_mm,
        requirements.object_geometry.bottle_diameter_mm,
    )
    protrusion = _first_number(
        geometry.design_envelope.max_protrusion_mm,
        requirements.design_envelope.max_protrusion_mm,
    )
    desk = _first_number(
        geometry.environment.desk_thickness_mm,
        requirements.environment.desk_thickness_mm,
    )
    mass = requirements.payload.filled_mass_kg
    method = (requirements.attachment.method or "").strip().lower()

    violations: List[FeasibilityViolation] = []

    if diameter is None:
        violations.append(
            FeasibilityViolation(
                code="payload_diameter_unavailable",
                message="Payload diameter is required before structural generation.",
                fields=["bottle_diameter_mm"],
            )
        )
    elif diameter <= 0:
        violations.append(
            FeasibilityViolation(
                code="nonpositive_payload_diameter",
                message=f"Payload diameter must be positive and physically nonzero (got {diameter} mm).",
                fields=["bottle_diameter_mm"],
            )
        )

    if desk is None:
        violations.append(
            FeasibilityViolation(
                code="desk_thickness_unavailable",
                message="Desk / support thickness is required before structural generation.",
                fields=["desk_thickness_mm"],
            )
        )
    elif desk <= 0:
        violations.append(
            FeasibilityViolation(
                code="nonpositive_desk_thickness",
                message=f"Desk thickness must be positive and physically nonzero (got {desk} mm).",
                fields=["desk_thickness_mm"],
            )
        )

    if protrusion is not None and protrusion <= 0:
        violations.append(
            FeasibilityViolation(
                code="nonpositive_protrusion",
                message=(
                    "Maximum outward protrusion must be positive and physically "
                    f"nonzero (got {protrusion} mm)."
                ),
                fields=["max_protrusion_mm"],
            )
        )

    if mass is not None and mass <= 0:
        violations.append(
            FeasibilityViolation(
                code="nonpositive_payload_mass",
                message=f"Payload mass must be positive and physically nonzero (got {mass} kg).",
                fields=["filled_bottle_mass_kg"],
            )
        )

    if method == "clamp" and (desk is None or desk <= 0):
        if not any(v.code == "nonpositive_desk_thickness" for v in violations) and not any(
            v.code == "desk_thickness_unavailable" for v in violations
        ):
            violations.append(
                FeasibilityViolation(
                    code="clamp_requires_positive_desk_thickness",
                    message="Clamp attachment requires a positive desk thickness.",
                    fields=["desk_thickness_mm", "attachment_method"],
                )
            )
        elif desk is not None and desk <= 0:
            violations.append(
                FeasibilityViolation(
                    code="clamp_requires_positive_desk_thickness",
                    message="Clamp attachment requires a positive desk thickness.",
                    fields=["desk_thickness_mm", "attachment_method"],
                )
            )

    if (
        diameter is not None
        and protrusion is not None
        and diameter > 0
        and protrusion > 0
        and diameter > protrusion
    ):
        violations.append(
            FeasibilityViolation(
                code="payload_exceeds_design_envelope",
                message=(
                    f"Payload diameter {diameter:g} mm cannot fit inside the "
                    f"maximum outward protrusion {protrusion:g} mm."
                ),
                fields=["bottle_diameter_mm", "max_protrusion_mm"],
            )
        )

    feasible = not violations
    if feasible:
        return DesignFeasibilityResult(
            feasible=True,
            violations=[],
            required_user_revision=False,
            message="Requirement / geometry combination is feasible for structural generation.",
        )

    lines = [v.message for v in violations if v.message]
    message = "DESIGN REQUIREMENTS INFEASIBLE. " + " ".join(lines)
    return DesignFeasibilityResult(
        feasible=False,
        violations=violations,
        required_user_revision=True,
        message=message,
    )


def feasibility_questions(result: DesignFeasibilityResult) -> List[ClarificationQuestion]:
    questions: List[ClarificationQuestion] = []
    seen = set()
    labels = {
        "bottle_diameter_mm": "Revise the payload / bottle diameter (mm).",
        "max_protrusion_mm": "Revise the maximum outward protrusion (mm) so the payload can fit.",
        "desk_thickness_mm": "Revise the desk / mounting-surface thickness (mm).",
        "filled_bottle_mass_kg": "Revise the filled payload mass (kg).",
        "attachment_method": "Revise the attachment method if clamp geometry is not possible.",
    }
    for violation in result.violations:
        for field in violation.fields:
            if field in seen:
                continue
            seen.add(field)
            questions.append(
                ClarificationQuestion(
                    field=field,
                    question=labels.get(field, f"Revise {field}."),
                    priority="high",
                )
            )
    if not questions:
        questions.append(
            ClarificationQuestion(
                field="max_protrusion_mm",
                question="Revise the design envelope so the payload can fit.",
                priority="high",
            )
        )
    return questions


def _first_number(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None
