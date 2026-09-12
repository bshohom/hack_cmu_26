"""Typed contracts for live engineering reasoning.

The reasoning model proposes structured judgments; deterministic code validates them,
computes from them, and may veto. Nothing here carries free-form numbers into the solver:
every field is schema-checked and then semantically checked (see reasoning_harness).

Two failure policies, chosen by ReasoningMode:
  PRODUCT   — a failed reasoning call falls back to the deterministic tables, visibly.
  DEVELOPER — a failed reasoning call BLOCKS with the full trace, so the gap can be fixed.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Units the deterministic layer knows how to handle.
ALLOWED_UNITS = {"mm", "kg", "N", "deg", "mm2", "MPa", "none"}
SAFETY_FACTOR_FLOOR = 2.0

# What a measurement influences. Kept deliberately broad: a narrow taxonomy makes the model
# argue with the schema instead of doing engineering, and anything unrecognised is
# normalised to "other" rather than rejected (the category is metadata, not a gate).
ALLOWED_AFFECTS = {
    "fit", "capacity", "envelope", "manufacturing", "safety",
    "stability", "stiffness", "durability", "ergonomics", "other",
}
AFFECTS_SYNONYMS = {
    "hold": "capacity", "holding": "capacity", "strength": "capacity", "load": "capacity",
    "tipping": "stability", "balance": "stability", "stability_margin": "stability",
    "clearance": "envelope", "size": "fit", "geometry": "fit", "fitment": "fit",
    "printing": "manufacturing", "print": "manufacturing", "deflection": "stiffness",
}


class ReasoningMode(str, Enum):
    PRODUCT = "product"
    DEVELOPER = "developer"


class ReasoningTask(str, Enum):
    PLAN_MEASUREMENTS = "plan_measurements"
    ASSESS_SCOPE = "assess_scope"
    ANALYZE_MECHANICS = "analyze_mechanics"
    REVISE = "revise"


class ReasoningFailureKind(str, Enum):
    NOT_CONFIGURED = "not_configured"
    UNSUPPORTED = "unsupported"
    TRANSPORT = "transport"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"
    SEMANTIC_INVALID = "semantic_invalid"


class ReasoningOutcome(str, Enum):
    LIVE = "live"
    FALLBACK = "fallback"
    BLOCKED = "blocked"


class ReasoningAttempt(BaseModel):
    """One model call: what came back and exactly why it was not accepted."""

    attempt: int
    ok: bool = False
    failure_kind: Optional[ReasoningFailureKind] = None
    error: str = ""
    raw_excerpt: str = ""
    latency_s: float = 0.0


class ReasoningTrace(BaseModel):
    """Full record of a reasoning call. Surfaced verbatim in developer mode."""

    task: ReasoningTask
    mode: ReasoningMode
    provider: str
    model: str = ""
    outcome: ReasoningOutcome
    attempts: List[ReasoningAttempt] = Field(default_factory=list)
    reason: str = ""  # why it fell back / blocked
    prompt_excerpt: str = ""
    total_latency_s: float = 0.0

    @property
    def failed(self) -> bool:
        return self.outcome is not ReasoningOutcome.LIVE

    def summary(self) -> str:
        tail = self.attempts[-1] if self.attempts else None
        detail = f" last failure: {tail.failure_kind.value if tail and tail.failure_kind else 'n/a'}"
        return (
            f"{self.task.value} [{self.outcome.value}] via {self.provider}/{self.model or 'n/a'} "
            f"after {len(self.attempts)} attempt(s){detail}. {self.reason}".strip()
        )


# --------------------------------------------------------------------------- measurements
class MeasurementRequest(BaseModel):
    """One quantity the design needs, with the reason it is needed."""

    field: str  # snake_case identifier, e.g. grip_depth_allowed_mm
    question: str
    unit: str
    why_it_matters: str
    affects: str  # fit | capacity | envelope | manufacturing | safety
    priority: str = "high"  # high | medium | low
    plausible_min: Optional[float] = None
    plausible_max: Optional[float] = None


class MeasurementPlan(BaseModel):
    """What to ask the user for this particular artifact — not a fixed questionnaire."""

    payload_description: str = ""
    payload_kind: str = "cylinder"  # model's judgment; compiled to a contact shape downstream
    mount_description: str = ""
    requests: List[MeasurementRequest] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- scope
class ScopeAssessment(BaseModel):
    """Hazard judgment. May only RESTRICT scope; the deterministic blocklist is the floor."""

    in_scope: bool = True
    hazard_class: str = "none"  # none | human_support | overhead | dynamic | impact | other
    reasons: List[str] = Field(default_factory=list)
    recommended_safety_factor: Optional[float] = None


# --------------------------------------------------------------------------- mechanics
class ReactionSurface(BaseModel):
    name: str
    description: str
    reaction_type: str = "bonded"  # bonded | friction | bearing
    restrained_dofs: List[str] = Field(default_factory=lambda: ["x", "y", "z"])


class FailureMode(BaseModel):
    name: str
    mechanism: str
    governing_quantity: str
    check_id: str  # must resolve against the implemented-check registry


class EnvelopeSemantics(BaseModel):
    """Which direction each envelope limit constrains, so 'protrusion' means one thing."""

    outward_axis: str = "+x"
    outward_limit_field: str = "max_protrusion_mm"
    grip_axis: str = "-x"
    grip_limit_field: Optional[str] = None
    vertical_limit_field: Optional[str] = None


class MechanicsSpec(BaseModel):
    """How the part reacts its load, and what must therefore be checked."""

    support_scheme: str = "edge_clamp"  # edge_clamp | bolted | free_standing | hung | other
    reaction_surfaces: List[ReactionSurface] = Field(default_factory=list)
    load_path: List[str] = Field(default_factory=list)
    failure_modes: List[FailureMode] = Field(default_factory=list)
    envelope: EnvelopeSemantics = Field(default_factory=EnvelopeSemantics)
    required_measurements: List[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- revision
class RevisionProposal(BaseModel):
    """What to change when a deterministic check fails."""

    action: str  # relax_requirement | change_parameter | switch_to_generated | reject
    target: str = ""
    value: Optional[float] = None
    rationale: str = ""
    user_question: str = ""


class ReasoningCallResult(BaseModel):
    """Harness return: the value (live or fallback) plus the trace that produced it."""

    trace: ReasoningTrace
    blocked: bool = False
    data: Optional[Dict] = None
