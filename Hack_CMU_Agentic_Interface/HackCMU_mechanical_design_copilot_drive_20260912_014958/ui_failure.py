"""Presentation-only failure interpretation. Does not change engineering behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence


STAGE_WARM_START = "warm_start_generation_failed"
STAGE_RECONSTRUCTION = "reconstruction_failed"
STAGE_OPTIMIZATION = "optimization_failed"
STAGE_NONCONVERGED = "optimization_not_converged"
STAGE_VERIFICATION = "verification_failed"
STAGE_SETUP = "problem_setup_failed"

_PROTRUSION_RE = re.compile(
    r"x\s*=\s*(?P<actual>[\d.]+)\s*mm.*?max_protrusion_mm\s*=\s*(?P<limit>[\d.]+)",
    re.IGNORECASE | re.DOTALL,
)
_PROTRUSION_RE_FLIP = re.compile(
    r"max_protrusion_mm\s*=\s*(?P<limit>[\d.]+).*?x\s*=\s*(?P<actual>[\d.]+)",
    re.IGNORECASE | re.DOTALL,
)
_BODIES_RE = re.compile(r"(?P<n>\d+)\s+disconnected\s+bod", re.IGNORECASE)


@dataclass(frozen=True)
class FailureFinding:
    kind: str
    what: str
    action: str
    field: Optional[str] = None
    current: Optional[float] = None
    needed: Optional[float] = None


@dataclass(frozen=True)
class FailureCard:
    stage: str
    headline: str
    findings: List[FailureFinding] = field(default_factory=list)
    primary_label: str = "Try again"
    secondary_label: Optional[str] = None
    log_label: str = "View full log"
    raw_log: str = ""
    constraint_field: Optional[str] = None

    def public_text(self) -> str:
        bits = [self.headline]
        for item in self.findings:
            bits.append(item.what)
            bits.append(item.action)
        return "\n".join(bits)


def _as_text(*chunks: Any) -> str:
    parts: List[str] = []
    for chunk in chunks:
        if chunk in (None, "", [], {}):
            continue
        if isinstance(chunk, (list, tuple)):
            parts.extend(str(item) for item in chunk if item not in (None, ""))
        else:
            parts.append(str(chunk))
    return "\n".join(parts)


def parse_failure_findings(text: str) -> List[FailureFinding]:
    """Turn raw solver/mesh text into user-facing what/next pairs."""
    blob = text or ""
    lowered = blob.lower()
    findings: List[FailureFinding] = []
    seen: set[str] = set()

    def _add(item: FailureFinding) -> None:
        if item.kind in seen:
            return
        seen.add(item.kind)
        findings.append(item)

    bodies = _BODIES_RE.search(blob)
    if bodies:
        count = int(bodies.group("n"))
        _add(
            FailureFinding(
                "disconnected_bodies",
                f"The generated shape split into {count} separate pieces.",
                "Try generating the starting design again.",
            )
        )
    elif "disconnected" in lowered:
        _add(
            FailureFinding(
                "disconnected_bodies",
                "The generated part was split into separate pieces.",
                "Try generating the starting design again.",
            )
        )

    if "not watertight" in lowered or "non-watertight" in lowered or "non watertight" in lowered:
        _add(
            FailureFinding(
                "watertight",
                "The generated shape is not one printable solid.",
                "Try generating it again.",
            )
        )

    match = _PROTRUSION_RE.search(blob) or _PROTRUSION_RE_FLIP.search(blob)
    if match:
        actual = float(match.group("actual"))
        limit = float(match.group("limit"))
        over = actual - limit
        over_txt = f"{over:g}"
        _add(
            FailureFinding(
                "envelope",
                f"The design exceeds your {limit:g} mm size limit by {over_txt} mm.",
                f"Keep the limit and retry, or increase it to at least {actual:g} mm.",
                field="max_protrusion_mm",
                current=limit,
                needed=actual,
            )
        )
    elif "max_protrusion" in lowered or "envelope" in lowered or "protrudes" in lowered:
        _add(
            FailureFinding(
                "envelope",
                "The design exceeded the allowed size.",
                "Retry with the current limit, or increase the size limit.",
                field="max_protrusion_mm",
            )
        )

    if "support region" in lowered or "load region" in lowered:
        _add(
            FailureFinding(
                "support_region",
                "The attachment area could not be identified.",
                "Check or change where the part should attach.",
                field="allowed_contact_region",
            )
        )

    return findings


def compose_raw_log(
    *,
    stage: str,
    warm_start: Optional[dict] = None,
    notes: str = "",
    warm_start_error: str = "",
    registration_error: str = "",
    registration_log: str = "",
    extra: Sequence[str] = (),
) -> str:
    """Unsimplified expert log. Never rewrite the backend text."""
    lines: List[str] = [f"stage: {stage}"]
    if warm_start:
        lines.append(f"attempts: {warm_start.get('attempts')}")
        lines.append(f"model: {warm_start.get('model')}")
        if warm_start.get("error"):
            lines.append(str(warm_start["error"]))
        for item in warm_start.get("problems") or []:
            lines.append(str(item))
        if warm_start.get("out_dir"):
            lines.append(f"out_dir: {warm_start['out_dir']}")
        if warm_start.get("script_path"):
            lines.append(f"script: {warm_start['script_path']}")
    if warm_start_error:
        lines.append(str(warm_start_error))
    if notes:
        lines.append(str(notes))
    if registration_error:
        lines.append(str(registration_error))
    if registration_log:
        lines.append(str(registration_log))
    lines.extend(str(item) for item in extra if item)
    return "\n".join(lines)


def build_failure_card(
    *,
    stage: str,
    raw_text: str = "",
    warm_start: Optional[dict] = None,
    notes: str = "",
    warm_start_error: str = "",
    registration_error: str = "",
    registration_log: str = "",
    extra: Sequence[str] = (),
) -> FailureCard:
    source = _as_text(
        raw_text,
        (warm_start or {}).get("error"),
        (warm_start or {}).get("problems"),
        warm_start_error,
        notes,
        registration_error,
    )
    findings = parse_failure_findings(source)
    constraint = next((item.field for item in findings if item.field), None)
    raw_log = compose_raw_log(
        stage=stage,
        warm_start=warm_start,
        notes=notes,
        warm_start_error=warm_start_error,
        registration_error=registration_error,
        registration_log=registration_log,
        extra=extra,
    )
    if stage == STAGE_RECONSTRUCTION:
        if not findings:
            findings = [
                FailureFinding(
                    "reconstruction",
                    "The scene could not be reconstructed from these photos.",
                    "Add clearer photos from more angles.",
                )
            ]
        return FailureCard(
            stage=stage,
            headline="We couldn't reconstruct the scene from these photos.",
            findings=findings,
            primary_label="Add / replace photos",
            raw_log=raw_log,
        )
    if stage == STAGE_NONCONVERGED:
        if not findings:
            findings = [
                FailureFinding(
                    "nonconvergence",
                    "The optimization ran but did not settle on a stable design.",
                    "Retry optimization or revise the design constraints.",
                )
            ]
        return FailureCard(
            stage=stage,
            headline="The optimization ran but did not settle on a stable design.",
            findings=findings,
            primary_label="Retry optimization",
            secondary_label="Change constraints",
            raw_log=raw_log,
        )
    if stage == STAGE_OPTIMIZATION:
        return FailureCard(
            stage=stage,
            headline="Optimization did not finish.",
            findings=findings
            or [
                FailureFinding(
                    "optimization",
                    "No optimized part was produced.",
                    "Retry optimization or revise the design constraints.",
                )
            ],
            primary_label="Retry optimization",
            raw_log=raw_log,
        )
    if stage == STAGE_VERIFICATION:
        return FailureCard(
            stage=stage,
            headline="Verification did not pass.",
            findings=findings,
            primary_label="Change constraints",
            raw_log=raw_log,
        )
    if stage == STAGE_SETUP:
        return FailureCard(
            stage=stage,
            headline="The design setup needs revision.",
            findings=findings,
            primary_label="Change constraints",
            raw_log=raw_log,
            constraint_field=constraint,
            secondary_label="Change size limit" if constraint == "max_protrusion_mm" else None,
        )
    secondary = "Change size limit" if constraint == "max_protrusion_mm" else (
        "Change constraints" if constraint else None
    )
    return FailureCard(
        stage=stage,
        headline="Generation failed validation",
        findings=findings
        or [
            FailureFinding(
                "warm_start",
                "We couldn’t create a valid starting design.",
                "Try generating the starting design again.",
            )
        ],
        primary_label="Try again",
        secondary_label=secondary,
        raw_log=raw_log,
        constraint_field=constraint,
    )


def detect_ui_failure(
    state: Any,
    *,
    warm_start: Optional[dict] = None,
    registration_error: str = "",
    registration_run: Any = None,
    warm_start_error: str = "",
) -> Optional[FailureCard]:
    """Pick the first user-facing failure. Engineering state is not modified."""
    stage_value = getattr(getattr(state, "stage", None), "value", None) or str(
        getattr(state, "stage", "") or ""
    )
    notes = getattr(state, "notes", "") or ""
    topo = getattr(state, "topology", None)
    run_ok = bool(registration_run is not None and getattr(registration_run, "ok", False))
    if registration_error and not run_ok:
        return build_failure_card(
            stage=STAGE_RECONSTRUCTION,
            registration_error=registration_error,
            registration_log=getattr(registration_run, "log", "") or "",
            notes=notes,
        )
    if warm_start and warm_start.get("ok") is False:
        return build_failure_card(
            stage=STAGE_WARM_START,
            warm_start=warm_start,
            notes=notes,
            warm_start_error=warm_start_error,
        )
    if warm_start_error or "warm-start generation failed" in notes.lower():
        return build_failure_card(
            stage=STAGE_WARM_START,
            notes=notes,
            warm_start_error=warm_start_error,
        )
    if stage_value == "topology_failed":
        return build_failure_card(stage=STAGE_OPTIMIZATION, notes=notes)
    if topo is not None:
        accept = getattr(topo, "acceptance", None) or {}
        if accept.get("acceptance_status") == "unresolved_not_converged" or getattr(
            topo, "converged", None
        ) is False:
            extra = [str(getattr(topo, "notes", "") or "")]
            return build_failure_card(
                stage=STAGE_NONCONVERGED,
                notes=notes,
                extra=extra,
            )
    if stage_value == "verification_failed":
        return build_failure_card(stage=STAGE_VERIFICATION, notes=notes)
    if stage_value == "design_review_failed":
        return build_failure_card(stage=STAGE_SETUP, notes=notes)
    return None


def raw_tokens_for_tests() -> Iterable[str]:
    return (
        "watertight",
        "disconnected bodies",
        "max_protrusion_mm",
        "RuntimeError",
    )
