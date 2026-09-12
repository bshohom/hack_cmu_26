"""Run one reasoning task: propose -> schema-check -> semantic-check -> retry -> settle.

The model never writes numbers straight into the engineering pipeline. Every response is
parsed, validated against its pydantic contract, then run through deterministic semantic
validators (units, ranges, identifiers, safety floors). A rejected response is sent back
with the exact failure text, up to MAX_ATTEMPTS.

Settling depends on the mode:
  PRODUCT   -> use the deterministic fallback, record why (visible, never silent).
  DEVELOPER -> block, carrying the whole trace so the reasoning gap can be fixed.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from reasoning_contracts import (
    AFFECTS_SYNONYMS,
    ALLOWED_AFFECTS,
    ALLOWED_UNITS,
    MeasurementPlan,
    MechanicsSpec,
    ReasoningAttempt,
    ReasoningCallResult,
    ReasoningFailureKind,
    ReasoningMode,
    ReasoningOutcome,
    ReasoningTask,
    ReasoningTrace,
    SAFETY_FACTOR_FLOOR,
    ScopeAssessment,
)

MAX_ATTEMPTS = 3
RAW_EXCERPT = 1200
_IDENT = re.compile(r"^[a-z][a-z0-9_]{1,60}$")

SYSTEM = (
    "You are the engineering reasoning stage of a mechanical design system. "
    "Reply with ONE JSON object and nothing else — no prose, no markdown fence. "
    "Match the requested schema exactly. Do not invent measured values: ask for them. "
    "Do not claim a design is safe."
)


class SemanticError(ValueError):
    """A response that parsed and matched the schema but is not engineering-usable."""


# ----------------------------------------------------------------- semantic validators
def validate_measurement_plan(plan: MeasurementPlan) -> None:
    if not plan.requests:
        raise SemanticError("requests is empty: list the measurements this design needs")
    seen = set()
    for r in plan.requests:
        if not _IDENT.match(r.field):
            raise SemanticError(
                f"field {r.field!r} is not a snake_case identifier (e.g. grip_depth_allowed_mm)"
            )
        if r.field in seen:
            raise SemanticError(f"duplicate field {r.field!r}")
        seen.add(r.field)
        if r.unit not in ALLOWED_UNITS:
            raise SemanticError(f"unit {r.unit!r} for {r.field!r} must be one of {sorted(ALLOWED_UNITS)}")
        if r.unit != "none" and not r.field.lower().endswith(("_mm", "_kg", "_n", "_deg", "_mm2", "_mpa")):
            raise SemanticError(
                f"field {r.field!r} carries unit {r.unit!r}, so its name must end with the matching "
                "suffix (_mm, _kg, _n, _deg) to keep units unambiguous downstream"
            )
        # Category is metadata: normalise rather than burn a retry on taxonomy wording.
        affects = (r.affects or "").strip().lower().replace(" ", "_")
        r.affects = AFFECTS_SYNONYMS.get(affects, affects if affects in ALLOWED_AFFECTS else "other")
        if r.priority not in {"high", "medium", "low"}:
            raise SemanticError(f"priority {r.priority!r} for {r.field!r} is not high/medium/low")
        lo, hi = r.plausible_min, r.plausible_max
        if lo is not None and hi is not None and lo >= hi:
            raise SemanticError(f"plausible range for {r.field!r} is inverted ({lo} >= {hi})")
        if r.unit in {"mm", "kg", "N"} and lo is not None and lo < 0:
            raise SemanticError(f"plausible_min for {r.field!r} must not be negative")
        if not r.why_it_matters.strip():
            raise SemanticError(f"why_it_matters is empty for {r.field!r}")
    if not any(r.priority == "high" for r in plan.requests):
        raise SemanticError("at least one measurement must be high priority")
    if not any(r.affects == "capacity" for r in plan.requests):
        raise SemanticError(
            "no measurement affects capacity: the load the part must carry has to come from somewhere"
        )


def validate_scope(assessment: ScopeAssessment, deterministic_floor: ScopeAssessment) -> None:
    """Reasoning may restrict scope; it may never widen what the floor rejected."""
    if not deterministic_floor.in_scope and assessment.in_scope:
        raise SemanticError(
            "this request is refused by the hazard floor and reasoning cannot approve it: "
            + "; ".join(deterministic_floor.reasons)
        )
    if not assessment.in_scope and not assessment.reasons:
        raise SemanticError("out-of-scope verdict needs at least one reason")
    sf = assessment.recommended_safety_factor
    if sf is not None and sf < SAFETY_FACTOR_FLOOR:
        raise SemanticError(
            f"recommended_safety_factor {sf} is below the floor {SAFETY_FACTOR_FLOOR}; "
            "reasoning may raise the safety factor, never lower it"
        )


def validate_mechanics(spec: MechanicsSpec, implemented_checks: set[str]) -> None:
    if not spec.reaction_surfaces:
        raise SemanticError("reaction_surfaces is empty: say what reacts the load")
    for s in spec.reaction_surfaces:
        if s.reaction_type not in {"bonded", "friction", "bearing"}:
            raise SemanticError(f"reaction_type {s.reaction_type!r} for {s.name!r} is not known")
        bad = [d for d in s.restrained_dofs if d not in {"x", "y", "z"}]
        if bad:
            raise SemanticError(f"restrained_dofs {bad} for {s.name!r} must be x/y/z")
    if not spec.failure_modes:
        raise SemanticError("failure_modes is empty: name what could make this part fail")
    unknown = [m.check_id for m in spec.failure_modes if m.check_id not in implemented_checks]
    if unknown:
        raise SemanticError(
            f"check_id(s) {unknown} are not implemented. Available checks: {sorted(implemented_checks)}. "
            "Use only these, or describe the mode with the closest available check."
        )


# ----------------------------------------------------------------- the call
def _extract_json(text: str) -> dict:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", stripped, re.S)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def call_reasoning(
    task: ReasoningTask,
    provider,
    prompt: str,
    schema: Type[BaseModel],
    mode: ReasoningMode,
    fallback: Callable[[], BaseModel],
    validate: Optional[Callable[[BaseModel], None]] = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> ReasoningCallResult:
    """Ask `provider` for a `schema` object; validate; retry with feedback; settle per mode."""
    name = getattr(provider, "name", provider.__class__.__name__)
    model = getattr(provider, "model", "")
    trace = ReasoningTrace(
        task=task, mode=mode, provider=name, model=model,
        outcome=ReasoningOutcome.LIVE, prompt_excerpt=prompt[:RAW_EXCERPT],
    )
    started = time.perf_counter()

    def settle(kind: ReasoningFailureKind, reason: str) -> ReasoningCallResult:
        trace.total_latency_s = time.perf_counter() - started
        trace.reason = reason
        if mode is ReasoningMode.DEVELOPER:
            trace.outcome = ReasoningOutcome.BLOCKED
            return ReasoningCallResult(trace=trace, blocked=True, data=None)
        trace.outcome = ReasoningOutcome.FALLBACK
        return ReasoningCallResult(trace=trace, blocked=False, data=fallback().model_dump(mode="json"))

    if not getattr(provider, "configured", False):
        return settle(ReasoningFailureKind.NOT_CONFIGURED,
                      getattr(provider, "not_connected_reason", "provider is not configured"))
    if not getattr(provider, "supports_structured", lambda: False)():
        return settle(ReasoningFailureKind.UNSUPPORTED,
                      f"{name} does not implement structured reasoning calls")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{prompt}\n\nJSON schema:\n{json.dumps(schema.model_json_schema())}"},
    ]
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        record = ReasoningAttempt(attempt=attempt)
        try:
            raw = provider.structured_json(messages)
            record.latency_s = time.perf_counter() - t0
            record.raw_excerpt = (raw or "")[:RAW_EXCERPT]
            payload = _extract_json(raw)
            obj = schema.model_validate(payload)
            if validate is not None:
                validate(obj)
            record.ok = True
            trace.attempts.append(record)
            trace.total_latency_s = time.perf_counter() - started
            return ReasoningCallResult(trace=trace, blocked=False, data=obj.model_dump(mode="json"))
        except json.JSONDecodeError as exc:
            record.failure_kind, record.error = ReasoningFailureKind.MALFORMED_JSON, f"response was not JSON: {exc}"
        except ValidationError as exc:
            record.failure_kind, record.error = ReasoningFailureKind.SCHEMA_INVALID, exc.json(indent=None)[:1500]
        except SemanticError as exc:
            record.failure_kind, record.error = ReasoningFailureKind.SEMANTIC_INVALID, str(exc)
        except Exception as exc:  # noqa: BLE001 — transport/provider failures
            record.failure_kind, record.error = ReasoningFailureKind.TRANSPORT, f"{type(exc).__name__}: {exc}"
        record.latency_s = record.latency_s or (time.perf_counter() - t0)
        trace.attempts.append(record)
        if record.failure_kind is ReasoningFailureKind.TRANSPORT:
            break  # retrying a broken transport just burns the clock
        messages.append({"role": "assistant", "content": record.raw_excerpt or "(no content)"})
        messages.append({
            "role": "user",
            "content": f"That response was rejected: {record.error}\nReturn the corrected complete JSON object.",
        })

    last = trace.attempts[-1] if trace.attempts else None
    return settle(last.failure_kind if last else ReasoningFailureKind.TRANSPORT,
                  f"reasoning failed after {len(trace.attempts)} attempt(s): {last.error if last else 'no response'}")
