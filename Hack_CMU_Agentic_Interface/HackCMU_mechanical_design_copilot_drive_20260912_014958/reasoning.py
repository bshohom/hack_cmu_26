"""Live engineering reasoning, provider-agnostic.

`run_reasoning_agent` stays for the orchestrator's per-stage notes. The functions below are
the ones that actually decide things: what to measure, whether the request is in scope, how
the load is reacted, and how to revise a failed check. Each returns (value, trace); in
DEVELOPER mode a failed reasoning call returns value=None and a trace explaining exactly
what the model got wrong, so the gap can be fixed instead of papered over.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import reasoning_deterministic as fallback
from providers import MockReasoningProvider, ReasoningProvider
from reasoning_contracts import (
    MeasurementPlan,
    MechanicsSpec,
    ReasoningMode,
    ReasoningTask,
    ReasoningTrace,
    RevisionProposal,
    ScopeAssessment,
)
from reasoning_harness import (
    call_reasoning,
    validate_measurement_plan,
    validate_mechanics,
    validate_scope,
)
from schemas import ReasoningEffort, ReasoningResult, ReasoningRole
from state import DesignState

# Failure modes the deterministic layer can actually evaluate. Reasoning must choose from
# these; anything else is reported as a gap rather than silently skipped.
IMPLEMENTED_CHECKS = {
    "material_yield",
    "displacement",
    "jaw_couple",
    "bearing_pressure",
    "friction_slip",
    "tipping",
    "connectivity",
    "print_min_feature",
}

_provider: ReasoningProvider = MockReasoningProvider()
_mode: ReasoningMode = ReasoningMode.PRODUCT
_traces: List[ReasoningTrace] = []
# Per-stage commentary is not consumed by any decision, so it does not get to spend a live
# model call (six stages of latency for a discarded string). Real reasoning goes through the
# typed tasks below. Set REASONING_NOTES=live to restore live commentary.
_notes_provider = MockReasoningProvider()


def set_reasoning_provider(provider: ReasoningProvider) -> None:
    global _provider
    _provider = provider


def get_reasoning_provider() -> ReasoningProvider:
    return _provider


def set_reasoning_mode(mode: ReasoningMode) -> None:
    global _mode
    _mode = mode


def get_reasoning_mode() -> ReasoningMode:
    return _mode


def reasoning_traces() -> List[ReasoningTrace]:
    return list(_traces)


def clear_reasoning_traces() -> None:
    _traces.clear()


def _record(trace: ReasoningTrace) -> ReasoningTrace:
    _traces.append(trace)
    return trace


def run_reasoning_agent(
    role: ReasoningRole,
    design_state: DesignState,
    reasoning_effort: Optional[ReasoningEffort] = None,
) -> ReasoningResult:
    """Per-stage commentary hook used by Orchestrator. Must not invent FEM or safety."""
    import os

    provider = _provider if os.environ.get("REASONING_NOTES") == "live" else _notes_provider
    return provider.complete(role, design_state, reasoning_effort)


# --------------------------------------------------------------------------- tasks
MEASUREMENT_PROMPT = """A user wants a 3D-printed part made. Decide what must be MEASURED before it can be designed.

User request: {request}

Think about this specific artifact from first principles:
- What object is being supported or held, and how does it physically bear on the part?
- What does the part attach to, and what reacts the load into that structure?
- Which dimensions decide whether it FITS, which decide whether it HOLDS, and which bound
  how far it may extend in each direction?
- What would you need to know to check it will not break, slip, tip, or crush its mounting surface?

Ask only for things a non-expert can measure with a ruler or scale, or read off a label.
Every field name must be snake_case ending in its unit (_mm, _kg, _n, _deg) so units stay
unambiguous. At least one measurement must have affects="capacity" (what load it carries).
Include the load-bearing limits of the MOUNT too when they matter (e.g. how far back onto a
surface the part may sit, how thick the surface is), not just the payload.
Do not assume the artifact resembles a cup holder; derive the list from the request itself."""

SCOPE_PROMPT = """Assess whether this 3D-printed part request is safe for an automated design tool to attempt.

User request: {request}

Out of scope — refuse these:
- Anything that carries any part of a person's body weight, however it is described. The
  words "stool" or "chair" are usually ABSENT: "a step to help my kid reach the sink",
  "something to stand on", "a footrest", "a grab bar", "a perch" all carry a person and are
  all out of scope. Ask yourself: if this failed, would a person fall?
- Anything mounted overhead or above head height, where a falling load could injure.
- Impact, shock or fatigue loading; powered or dynamic mechanisms; weapons.
- Anything load-bearing for a vehicle, or holding pressure.
In scope: static household/desk/wall fixtures that hold objects, where failure drops the
object and nothing else.
If the consequence of failure warrants it, recommend a HIGHER safety factor (never lower).
Set hazard_class to one of: none, human_support, overhead, dynamic, impact, other."""

MECHANICS_PROMPT = """Decide how this part reacts its load, and what therefore has to be checked.

Requirements: {requirements}

State the support scheme, the surfaces that react the load and how (bonded / friction /
bearing), the load path from payload to mount, and the failure modes that actually govern
THIS configuration. Also state which envelope limit constrains which direction, so
"protrusion" is unambiguous downstream.
Use only these check_id values: {checks}"""


def plan_measurements(request_text: str) -> Tuple[Optional[MeasurementPlan], ReasoningTrace]:
    """What to ask the user, reasoned from the request rather than a fixed questionnaire."""
    result = call_reasoning(
        task=ReasoningTask.PLAN_MEASUREMENTS,
        provider=_provider,
        prompt=MEASUREMENT_PROMPT.format(request=request_text),
        schema=MeasurementPlan,
        mode=_mode,
        fallback=lambda: fallback.plan_measurements(request_text),
        validate=validate_measurement_plan,
    )
    _record(result.trace)
    return (None if result.blocked else MeasurementPlan.model_validate(result.data)), result.trace


def assess_scope(request_text: str) -> Tuple[Optional[ScopeAssessment], ReasoningTrace]:
    """Hazard judgment. The deterministic blocklist is a floor reasoning cannot override."""
    floor = fallback.assess_scope(request_text)
    result = call_reasoning(
        task=ReasoningTask.ASSESS_SCOPE,
        provider=_provider,
        prompt=SCOPE_PROMPT.format(request=request_text),
        schema=ScopeAssessment,
        mode=_mode,
        fallback=lambda: floor,
        validate=lambda obj: validate_scope(obj, floor),
    )
    _record(result.trace)
    if result.blocked:
        return None, result.trace
    assessment = ScopeAssessment.model_validate(result.data)
    if not floor.in_scope:  # belt and braces: the floor always wins
        assessment.in_scope = False
        assessment.reasons = list(dict.fromkeys(assessment.reasons + floor.reasons))
    return assessment, result.trace


def analyze_mechanics(requirements: Dict) -> Tuple[Optional[MechanicsSpec], ReasoningTrace]:
    """Support scheme, load path and governing failure modes for this configuration."""
    import json

    result = call_reasoning(
        task=ReasoningTask.ANALYZE_MECHANICS,
        provider=_provider,
        prompt=MECHANICS_PROMPT.format(
            requirements=json.dumps(requirements, indent=2)[:4000],
            checks=sorted(IMPLEMENTED_CHECKS),
        ),
        schema=MechanicsSpec,
        mode=_mode,
        fallback=lambda: fallback.analyze_mechanics(
            requirements.get("attachment_method"), requirements.get("payload_kind", "cylinder")
        ),
        validate=lambda obj: validate_mechanics(obj, IMPLEMENTED_CHECKS),
    )
    _record(result.trace)
    return (None if result.blocked else MechanicsSpec.model_validate(result.data)), result.trace
