"""Deterministic design reviewer for the mock structure/analysis loop.

This is integration logic, not physical safety certification.
"""

from __future__ import annotations

from schemas import (
    AnalysisOutput,
    DesignReviewOutput,
    RequestedChange,
    ReviewDecision,
    ReviewViolation,
    StructureOutput,
)

MAX_DISPLACEMENT_MM = 3.0
MOCK_ALLOWABLE_STRESS_MPA = 20.0


def review_design(structure: StructureOutput, analysis: AnalysisOutput) -> DesignReviewOutput:
    violations: list[ReviewViolation] = []
    changes: list[RequestedChange] = []
    stress_mpa = analysis.max_stress_pa / 1.0e6

    if analysis.max_displacement_mm > MAX_DISPLACEMENT_MM:
        violations.append(
            ReviewViolation(
                metric="max_displacement_mm",
                observed=analysis.max_displacement_mm,
                limit=MAX_DISPLACEMENT_MM,
            )
        )
        changes.append(
            RequestedChange(
                parameter="support_thickness_mm",
                action="increase",
                reason="excessive displacement",
            )
        )
        changes.append(
            RequestedChange(
                parameter="brace_count",
                action="increase",
                reason="excessive displacement",
            )
        )

    if stress_mpa > MOCK_ALLOWABLE_STRESS_MPA:
        violations.append(
            ReviewViolation(
                metric="max_stress_mpa",
                observed=stress_mpa,
                limit=MOCK_ALLOWABLE_STRESS_MPA,
            )
        )
        changes.append(
            RequestedChange(
                parameter="support_thickness_mm",
                action="increase",
                reason="excessive stress",
            )
        )
        if not any(change.parameter == "brace_thickness_mm" for change in changes):
            changes.append(
                RequestedChange(
                    parameter="brace_thickness_mm",
                    action="increase",
                    reason="excessive stress",
                )
            )

    decision = ReviewDecision.PASS if not violations else ReviewDecision.REVISE
    return DesignReviewOutput(
        decision=decision,
        violations=violations,
        requested_changes=changes,
        iteration=structure.iteration,
        notes=(
            "Deterministic mock design review for loop testing. "
            "Not physical safety certification."
        ),
    )
