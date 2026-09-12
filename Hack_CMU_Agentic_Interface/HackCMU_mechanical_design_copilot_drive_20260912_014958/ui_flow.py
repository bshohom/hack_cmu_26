"""Presentation-only mapping from DesignState to the five user-facing phases.

Does not change Orchestrator transitions, schemas, or acceptance semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from schemas import WorkflowStage
from state import DesignState

UI_PHASES = ("Describe", "Capture", "Design", "Verify", "Export")

# Clarification fields a future reconstruction pass would typically fill.
# Used only to pick the highlighted phase; questions still come from the backend.
_SCENE_FIELDS = {
    "desk_thickness_mm",
    "allowed_contact_region",
    "max_protrusion_mm",
}

_NUMERIC_EMPTY = {0, 0.0}


@dataclass(frozen=True)
class ActionSpec:
    """What the central card should say. Display only."""

    phase: str
    title: str
    subtitle: str
    button: Optional[str]
    kind: str  # start | continue | optimize | download | none
    result_status: Optional[str] = None  # preliminary | preview | unverified | ready


def _stage(state: DesignState) -> WorkflowStage:
    return state.stage


def clarification_fields(state: DesignState) -> List[str]:
    return [q.field for q in (state.clarifications or []) if getattr(q, "field", None)]


def _field_is_empty(field: str, value: Any) -> bool:
    if value in (None, ""):
        return True
    if field.endswith("_mm") or field.endswith("_kg"):
        try:
            return float(value) in _NUMERIC_EMPTY
        except (TypeError, ValueError):
            return True
    return False


def missing_detail_count(state: DesignState, answers: Optional[Dict[str, Any]] = None) -> int:
    """How many backend clarification fields still need a value."""
    answers = answers or {}
    return sum(
        1 for field in clarification_fields(state) if _field_is_empty(field, answers.get(field))
    )


def need_details_title(count: int) -> str:
    if count <= 0:
        return "Ready to optimize"
    if count == 1:
        return "Need 1 more detail"
    return f"Need {count} more details"


def answers_look_complete(state: DesignState, answers: Optional[Dict[str, Any]]) -> bool:
    """True when every backend clarification field has a non-empty user value."""
    fields = clarification_fields(state)
    if not fields:
        return bool(state.requirements)
    return missing_detail_count(state, answers) == 0


def ui_phase(state: DesignState, answers: Optional[Dict[str, Any]] = None) -> str:
    """Map backend stage + artifacts onto one of the five UI phases."""
    stage = _stage(state)
    if stage == WorkflowStage.REJECTED:
        return "Describe"
    if stage == WorkflowStage.COMPLETE and state.cad is not None:
        return "Export"
    if stage == WorkflowStage.COMPLETE:
        return "Verify" if state.topology is not None else "Export"

    if stage in (
        WorkflowStage.VERIFICATION,
        WorkflowStage.VERIFICATION_FAILED,
    ) or state.verification is not None:
        return "Verify"
    if stage in (
        WorkflowStage.ANALYSIS,
        WorkflowStage.DESIGN_REVIEW,
        WorkflowStage.DESIGN_REVIEW_FAILED,
    ) and state.topology is not None:
        return "Verify"
    if state.topology is not None:
        return "Verify"

    if stage in (
        WorkflowStage.GEOMETRY,
        WorkflowStage.CANDIDATE_FIT,
        WorkflowStage.FEASIBILITY_CHECK,
        WorkflowStage.STRUCTURE,
        WorkflowStage.TOPOLOGY_OPTIMIZATION,
        WorkflowStage.TOPOLOGY_FAILED,
        WorkflowStage.ANALYSIS,
        WorkflowStage.DESIGN_REVIEW,
        WorkflowStage.DESIGN_REVIEW_FAILED,
    ) or state.geometry is not None or state.structure is not None:
        return "Design"

    if stage == WorkflowStage.REQUEST_INFORMATION:
        if answers_look_complete(state, answers):
            return "Design"
        fields = set(clarification_fields(state))
        if fields & _SCENE_FIELDS:
            return "Capture"
        return "Describe"

    if state.requirements is None:
        return "Describe"
    return "Capture"


def phase_index(phase: str) -> int:
    try:
        return UI_PHASES.index(phase)
    except ValueError:
        return 0


def result_status(state: DesignState) -> Optional[str]:
    """preliminary / preview / unverified / ready — never invents a pass."""
    topo = state.topology
    if topo is None:
        if state.stage == WorkflowStage.COMPLETE:
            return "unverified"
        return None
    acceptance = topo.acceptance or {}
    status = acceptance.get("acceptance_status")
    if status == "unresolved_not_converged" or topo.converged is False:
        return "preview"
    if topo.is_mock:
        return "preliminary"
    if status == "pass" and acceptance.get("accepted"):
        if state.verification and state.verification.safety_validated and not topo.is_mock:
            return "ready"
        return "unverified"
    if status == "fail":
        return "unverified"
    return "preliminary" if topo.is_mock else "unverified"


def action_spec(state: DesignState, answers: Optional[Dict[str, Any]] = None) -> ActionSpec:
    """Single primary action for the current DesignState."""
    stage = _stage(state)
    phase = ui_phase(state, answers)
    status = result_status(state)

    if stage == WorkflowStage.REJECTED:
        return ActionSpec(
            phase,
            "Out of scope",
            state.reject_reason or "This request cannot continue.",
            "Revise inputs",
            "revise",
        )
    if stage == WorkflowStage.DESIGN_REVIEW_FAILED:
        return ActionSpec(
            phase,
            "Design review did not pass",
            "Topology was not started.",
            "Revise inputs",
            "revise",
            "unverified",
        )
    if stage == WorkflowStage.TOPOLOGY_FAILED:
        return ActionSpec(
            phase,
            "Optimization did not finish",
            "No result was produced.",
            "Revise inputs",
            "revise",
            "unverified",
        )
    if stage == WorkflowStage.VERIFICATION_FAILED:
        return ActionSpec(
            phase,
            "Verification did not pass",
            "Checks did not meet acceptance criteria.",
            "Revise inputs",
            "revise",
            "unverified",
        )
    if state.contract_error:
        return ActionSpec(
            phase,
            "Inputs disagree",
            "",
            "Continue",
            "continue",
        )
    if (
        stage == WorkflowStage.REQUEST_INFORMATION
        and state.feasibility is not None
        and not state.feasibility.feasible
    ):
        return ActionSpec(
            phase,
            "Requirements conflict",
            state.feasibility.message or "Revise the values below.",
            "Continue",
            "continue",
        )
    if (
        stage == WorkflowStage.REQUEST_INFORMATION
        and state.candidate_fit is not None
        and not state.candidate_fit.fits
    ):
        return ActionSpec(
            phase,
            "Design does not fit",
            state.candidate_fit.message or "Revise the values below.",
            "Continue",
            "continue",
        )

    if stage == WorkflowStage.COMPLETE:
        has_stl = bool(
            state.cad
            and state.cad.filename
            and not str(state.cad.filename).startswith("mock://")
        )
        title = "Design ready" if status != "preview" else "Optimization preview"
        subtitle = _complete_subtitle(state, status)
        return ActionSpec(
            phase,
            title,
            subtitle,
            "Download STL" if has_stl else None,
            "download" if has_stl else "none",
            status,
        )

    if state.topology is not None:
        return ActionSpec(
            phase,
            "Review the optimized design",
            _complete_subtitle(state, status),
            None,
            "none",
            status,
        )

    if stage == WorkflowStage.REQUIREMENTS and state.requirements is None:
        return ActionSpec(
            phase,
            "What do you need?",
            "",
            "Start design",
            "start",
        )

    if stage == WorkflowStage.REQUEST_INFORMATION:
        if answers_look_complete(state, answers):
            return ActionSpec(
                "Design",
                "Ready to optimize",
                "",
                "Run optimization",
                "optimize",
            )
        missing = missing_detail_count(state, answers)
        title = need_details_title(missing)
        if set(clarification_fields(state)) & _SCENE_FIELDS:
            return ActionSpec(
                "Capture",
                title,
                "",
                "Continue",
                "continue",
            )
        return ActionSpec(
            phase,
            title,
            "",
            "Continue",
            "continue",
        )

    if stage in (
        WorkflowStage.GEOMETRY,
        WorkflowStage.CANDIDATE_FIT,
        WorkflowStage.FEASIBILITY_CHECK,
        WorkflowStage.STRUCTURE,
        WorkflowStage.ANALYSIS,
        WorkflowStage.DESIGN_REVIEW,
        WorkflowStage.TOPOLOGY_OPTIMIZATION,
    ) and state.topology is None:
        return ActionSpec(
            phase,
            "Ready to optimize",
            "",
            "Run optimization",
            "optimize",
        )

    return ActionSpec(
        phase,
        "What do you need?",
        "",
        "Start design",
        "start",
    )


def _complete_subtitle(state: DesignState, status: Optional[str]) -> str:
    if status == "preview":
        return "Not converged — preview only."
    if status == "preliminary":
        return "Preview only."
    if status == "unverified":
        return "Physical safety is not certified."
    if status == "ready":
        return "Ready to export."
    return ""


def design_summary_rows(
    state: DesignState,
    scene_status: str,
    answers: Optional[Dict[str, Any]] = None,
) -> List[tuple[str, str]]:
    """Human-readable summary rows. Empty values are omitted."""
    req = state.requirements
    answers = answers or {}
    rows: List[tuple[str, str]] = []
    if req is None and not answers:
        return rows
    message = ((req.user_message if req else "") or "").strip()
    if message:
        rows.append(("Requested function", message if len(message) < 160 else message[:157] + "…"))

    def _val(req_value: Any, *keys: str) -> Any:
        if req_value not in (None, ""):
            return req_value
        for key in keys:
            if answers.get(key) not in (None, ""):
                return answers[key]
        return None

    mass = _val(req.payload.filled_mass_kg if req else None, "filled_bottle_mass_kg")
    kind = req.object_geometry.kind if req else None
    size = _val(req.object_geometry.bottle_diameter_mm if req else None, "bottle_diameter_mm")
    payload_bits = []
    if mass is not None:
        payload_bits.append(f"{mass:g} kg")
    if kind:
        payload_bits.append(str(kind))
    if size is not None:
        payload_bits.append(f"{size:g} mm")
    if payload_bits:
        rows.append(("Payload", " · ".join(payload_bits)))
    method = _val(req.attachment.method if req else None, "attachment_method")
    region = _val(req.attachment.allowed_contact_region if req else None, "allowed_contact_region")
    attach = [str(v) for v in (method, region) if v]
    if attach:
        rows.append(("Attachment", " · ".join(attach)))
    material = _val(req.manufacturing.material if req else None, "material")
    if material:
        rows.append(("Material", str(material)))
    mfg = _val(req.manufacturing.method if req else None, "manufacturing_method")
    if mfg:
        rows.append(("Manufacturing", str(mfg)))
    env = []
    prot = _val(req.design_envelope.max_protrusion_mm if req else None, "max_protrusion_mm")
    if prot is not None:
        env.append(f"protrusion ≤ {prot:g} mm")
    if req and req.design_envelope.max_width_mm is not None:
        env.append(f"width ≤ {req.design_envelope.max_width_mm:g} mm")
    if req and req.design_envelope.max_height_mm is not None:
        env.append(f"height ≤ {req.design_envelope.max_height_mm:g} mm")
    if env:
        rows.append(("Envelope", " · ".join(env)))
    rows.append(("Geometry / scene", scene_status))
    return rows


def scene_status_text(
    *,
    has_photo: bool,
    registration: Optional[Dict[str, Any]],
    geometry_mode: str,
) -> str:
    bits: List[str] = []
    if has_photo:
        bits.append("photo attached")
    else:
        bits.append("no photo")
    if registration is None:
        bits.append("no registration file")
    elif registration.get("error"):
        bits.append("registration unreadable")
    elif registration.get("prefill"):
        bits.append("registration measurements available")
    else:
        bits.append("registration present, low confidence")
    bits.append(geometry_mode)
    return " · ".join(bits)


def stepper_states(current_phase: str, blocked: bool = False) -> Sequence[tuple[str, str]]:
    """(phase_name, css_state) for the top indicator: done / current / blocked / todo."""
    idx = phase_index(current_phase)
    out = []
    for i, name in enumerate(UI_PHASES):
        if i < idx:
            out.append((name, "done"))
        elif i == idx:
            out.append((name, "blocked" if blocked else "current"))
        else:
            out.append((name, "todo"))
    return out
