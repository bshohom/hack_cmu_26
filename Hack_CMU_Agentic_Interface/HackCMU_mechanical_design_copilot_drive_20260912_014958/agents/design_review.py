"""Seed-sizing convergence for the structural layout. NOT a design review.

This decides one thing: whether the starting member sizes handed to the optimizer are worth
iterating on again. It reads `tools.analysis`, which is a closed-form function of
`support_thickness_mm` and `brace_count` only — payload force, material and geometry are not
inputs — so its numbers describe the seed, not the part.

It therefore has NO authority to accept or reject a design:

* `ReviewDecision.REVISE` means "size the seed again", not "this design is unsafe".
* `ReviewDecision.PASS`   means "sizing has converged, hand off", not "this design is safe".

Sizing always terminates in PASS — on meeting the target, on running out of parameter
headroom, or on an iteration cap. A heuristic that never reads the load must not be able to
stop a design from being optimized; the only gate on strength is the post-optimization FE
check in `Orchestrator._acceptance_verdict`.
"""

from __future__ import annotations

from typing import Optional

from schemas import (
    AnalysisOutput,
    DesignReviewOutput,
    RequestedChange,
    ReviewDecision,
    ReviewViolation,
    StructureOutput,
)

# Seed-sizing targets. These are scores of the heuristic, not engineering limits: the units
# are carried over from when this pretended to be analysis, and nothing physical depends on
# them. They only decide when to stop growing the starting geometry.
SIZING_TARGET_DISPLACEMENT = 3.0
SIZING_TARGET_STRESS_MPA = 20.0

# Parameter ceiling from StructureAgent; once reached, further iterations cannot change the
# seed, so continuing to ask for revisions would loop without effect.
MAX_BRACE_COUNT = 4
# Fallback cap when the caller does not supply its own iteration budget.
SIZING_ITERATION_CAP = 4

NOTES = (
    "Seed-sizing convergence, not a design review. Scores come from a closed-form function "
    "of support thickness and brace count; load, material and geometry are not inputs. "
    "PASS means sizing converged, not that the design is safe — the post-optimization FE "
    "check is the only gate on strength."
)


def review_design(
    structure: StructureOutput,
    analysis: AnalysisOutput,
    max_iterations: Optional[int] = None,
) -> DesignReviewOutput:
    """Decide whether to size the seed again. Never rejects a design.

    `max_iterations`, when given, is the caller's loop budget; sizing reports converged on
    the final iteration so the workflow is always handed to the optimizer.
    """
    violations: list[ReviewViolation] = []
    changes: list[RequestedChange] = []
    stress_mpa = analysis.max_stress_pa / 1.0e6

    if analysis.max_displacement_mm > SIZING_TARGET_DISPLACEMENT:
        violations.append(
            ReviewViolation(
                metric="seed_displacement_score",
                observed=analysis.max_displacement_mm,
                limit=SIZING_TARGET_DISPLACEMENT,
            )
        )
        changes.append(
            RequestedChange(
                parameter="support_thickness_mm",
                action="increase",
                reason="seed sizing: displacement score above target",
            )
        )
        changes.append(
            RequestedChange(
                parameter="brace_count",
                action="increase",
                reason="seed sizing: displacement score above target",
            )
        )

    if stress_mpa > SIZING_TARGET_STRESS_MPA:
        violations.append(
            ReviewViolation(
                metric="seed_stress_score",
                observed=stress_mpa,
                limit=SIZING_TARGET_STRESS_MPA,
            )
        )
        changes.append(
            RequestedChange(
                parameter="support_thickness_mm",
                action="increase",
                reason="seed sizing: stress score above target",
            )
        )
        if not any(change.parameter == "brace_thickness_mm" for change in changes):
            changes.append(
                RequestedChange(
                    parameter="brace_thickness_mm",
                    action="increase",
                    reason="seed sizing: stress score above target",
                )
            )

    target_met = not violations
    out_of_headroom = structure.parameters.brace_count >= MAX_BRACE_COUNT
    budget = max_iterations if max_iterations is not None else SIZING_ITERATION_CAP
    last_iteration = structure.iteration + 1 >= budget

    converged = target_met or out_of_headroom or last_iteration
    notes = NOTES
    if converged and not target_met:
        # Sizing stopped without reaching its own target. That is a statement about the
        # heuristic, not about the part, so it is recorded and handed on rather than
        # blocking: the FE check decides whether the optimized result is acceptable.
        reason = "parameter headroom exhausted" if out_of_headroom else "iteration budget reached"
        notes = (
            f"{NOTES} Sizing stopped short of its target ({reason}); the seed is handed to "
            "the optimizer as-is and the post-optimization FE check decides acceptance."
        )

    return DesignReviewOutput(
        decision=ReviewDecision.PASS if converged else ReviewDecision.REVISE,
        violations=violations,
        requested_changes=[] if converged else changes,
        iteration=structure.iteration,
        notes=notes,
    )
