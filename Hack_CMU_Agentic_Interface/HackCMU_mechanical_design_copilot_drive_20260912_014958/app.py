"""Mechanical Design Copilot — thin Streamlit view over the existing orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st

from fixtures import load_clarifications, load_user_request
from geometry_sources import (
    FIELD_LABELS,
    GEOM_ADAPTIVE,
    GEOM_GOLDEN,
    GEOM_IMPORTED,
    GEOM_LIVE,
    GEOM_MODE_OPTIONS,
    chat_error_key,
    effective_geometry_mode,
    fixtures_for_geometry_mode,
    format_mismatch_message,
    geometry_provenance_text,
    golden_fixture_assumptions,
    imported_candidate_for_mode,
    requirement_geometry_rows,
)
from cursor_adapter import (
    COMPARE_A_MODEL_ID,
    COMPARE_A_PARAM_HINTS,
    COMPARE_B_MODEL_ID,
    COMPARE_B_PARAM_HINTS,
    DEFAULT_MODEL_ID,
    DEFAULT_PARAM_HINTS,
    find_catalog_model,
    is_cursor_configured,
    list_cursor_models,
    observation_to_requirements_update,
    pick_catalog_model_id,
    resolve_param_values,
)
from orchestrator import Orchestrator
from providers import get_provider
from reasoning import set_reasoning_provider
from schemas import (
    InteractionResult,
    MassProvenance,
    RequirementsUpdate,
    SceneObservation,
    StructureInput,
    StructureOutput,
    WorkflowStage,
)
from state import DesignState
from ui_inspect import (
    PIPELINE,
    design_loop_timeline,
    engineering_evidence_is_mocked,
    highlighted_stage,
    inspect_cards,
    iteration_cards,
    pipeline_status,
)
from ui_viz import (
    geometry_figure,
    imported_candidate_mesh_figure,
    imported_candidate_particle_figure,
    structure_figure,
)

HAPPY_PATH_MESSAGE = load_user_request()
MISSING_INFO_MESSAGE = "I want a cup holder attached to this desk."
REJECTED_MESSAGE = "Design a stool that supports a 100 kg person."

NUMERIC_FIELDS = {
    "filled_bottle_mass_kg",
    "bottle_diameter_mm",
    "bottle_height_mm",
    "desk_thickness_mm",
    "max_protrusion_mm",
    "max_part_mass_kg",
}
SELECT_FIELDS = {
    "attachment_method": ["clamp", "screws", "adhesive"],
    "manufacturing_method": ["3d_print", "fdm", "sla"],
}

BADGE_COLORS = {
    "NOT STARTED": ("#6b7280", "#f3f4f6"),
    "WAITING FOR INPUT": ("#92400e", "#fef3c7"),
    "NEEDS REVISION": ("#92400e", "#fef3c7"),
    "MOCK": ("#1e40af", "#dbeafe"),
    "LIVE": ("#065f46", "#d1fae5"),
    "COMPLETE": ("#065f46", "#d1fae5"),
    "FEASIBLE": ("#065f46", "#d1fae5"),
    "INFEASIBLE": ("#991b1b", "#fee2e2"),
    "PASS": ("#065f46", "#d1fae5"),
    "REVISE": ("#92400e", "#fef3c7"),
    "FAILED": ("#991b1b", "#fee2e2"),
    "BLOCKED": ("#9a3412", "#ffedd5"),
    "REJECTED": ("#991b1b", "#fee2e2"),
    "IMPORTED": ("#1e40af", "#dbeafe"),
    "ITERATION 0": ("#1e40af", "#dbeafe"),
    "ITERATION 1": ("#1e40af", "#dbeafe"),
    "ITERATION 2": ("#1e40af", "#dbeafe"),
    "ITERATION 3": ("#1e40af", "#dbeafe"),
}


def _new_orchestrator() -> Orchestrator:
    mode = effective_geometry_mode(st.session_state.get("mode_geom"))
    return Orchestrator(
        fixtures=fixtures_for_geometry_mode(mode),
        imported_candidate=imported_candidate_for_mode(mode),
    )


def apply_pending_request_prefill(store: Dict[str, Any]) -> None:
    """Copy pending prefill into the widget key before the widget is created.

    `request_text` is owned by the text_area. Call this only at the start of a
    script run, before `st.text_area(..., key="request_text")`.
    """
    pending = store.get("pending_request_prefill")
    if pending is not None:
        store["request_text"] = pending
        store["pending_request_prefill"] = None


REASONING_CHOICES = ["Mock", "K2 Horizon", "Grok", "Cursor"]


def _default_reasoning_choice() -> str:
    return "Cursor" if is_cursor_configured() else "Mock"


def _geometry_mode_label(mode: str) -> str:
    if mode == GEOM_GOLDEN:
        return "Golden Fixture  —  regression / integration test"
    if mode == GEOM_LIVE:
        return "Live Geometry  —  not connected"
    if mode == GEOM_IMPORTED:
        return "Imported Candidate Geometry"
    return "Adaptive Synthetic Mock"


def _init_session() -> None:
    if st.session_state.get("mode_geom") not in GEOM_MODE_OPTIONS:
        st.session_state.mode_geom = GEOM_ADAPTIVE
    if st.session_state.get("mode_reason") not in REASONING_CHOICES:
        st.session_state.mode_reason = _default_reasoning_choice()
    if "orch" not in st.session_state:
        st.session_state.orch = _new_orchestrator()
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "image_name" not in st.session_state:
        st.session_state.image_name = None
    if "compare" not in st.session_state:
        st.session_state.compare = None
    if "pending_request_prefill" not in st.session_state:
        st.session_state.pending_request_prefill = None
    if "submitted_request" not in st.session_state:
        st.session_state.submitted_request = ""
    if "ui_notice" not in st.session_state:
        st.session_state.ui_notice = ""
    if "last_chat_error_key" not in st.session_state:
        st.session_state.last_chat_error_key = None
    if "scene_observation" not in st.session_state:
        st.session_state.scene_observation = None
    if "scene_observation_meta" not in st.session_state:
        st.session_state.scene_observation_meta = {}
    if "cursor_compare" not in st.session_state:
        st.session_state.cursor_compare = None
    if "pending_analyze" not in st.session_state:
        st.session_state.pending_analyze = False
    if "request_text" not in st.session_state and not st.session_state.pending_request_prefill:
        st.session_state.request_text = HAPPY_PATH_MESSAGE


def _apply_pending_prefills() -> None:
    apply_pending_request_prefill(st.session_state)


def _clear_answer_widgets() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("ans_") or key == "no_drill":
            del st.session_state[key]


def _reset(
    ingest_message: Optional[str] = None,
    prefill_happy: bool = False,
    request_prefill: Optional[str] = None,
    clear_image: bool = False,
) -> None:
    """Reset orchestrator/session data. Does not write instantiated widget keys.

    To change the design-request text box, set `request_prefill` and rerun.
    `_apply_pending_prefills` copies it into `request_text` before the widget.
    """
    st.session_state.orch = _new_orchestrator()
    st.session_state.chat = []
    st.session_state.answers = {}
    st.session_state.compare = None
    st.session_state.cursor_compare = None
    st.session_state.scene_observation = None
    st.session_state.scene_observation_meta = {}
    st.session_state.ui_notice = ""
    st.session_state.last_chat_error_key = None
    _clear_answer_widgets()
    if request_prefill is not None:
        st.session_state.pending_request_prefill = request_prefill
    if prefill_happy:
        st.session_state.answers = load_clarifications().model_dump()
    if clear_image:
        st.session_state.image_bytes = None
        st.session_state.image_name = None
    if ingest_message:
        st.session_state.submitted_request = ingest_message
        _ingest(ingest_message)
    elif request_prefill is not None:
        st.session_state.submitted_request = ""


def _append(role: str, text: str, error_key: Optional[tuple] = None) -> None:
    if error_key is not None:
        if st.session_state.get("last_chat_error_key") == error_key:
            return
        st.session_state.last_chat_error_key = error_key
    st.session_state.chat.append({"role": role, "text": text})


def _assistant_after_interaction(state: DesignState) -> str:
    decision = state.interaction_decision.value if state.interaction_decision else "none"
    if state.stage == WorkflowStage.REJECTED:
        reason = state.reject_reason or "Request is out of scope."
        return (
            f"Rejected. This request is outside the prototype scope.\n\n"
            f"Reason: {reason}\n\n"
            "Geometry, FEM, and topology were not executed."
        )
    if state.stage == WorkflowStage.REQUEST_INFORMATION:
        if state.feasibility is not None and not state.feasibility.feasible:
            conflicts = "\n".join(
                f"- {v.code}: {v.message}" for v in state.feasibility.violations
            )
            questions = "\n".join(f"- {q.question}" for q in state.clarifications)
            return (
                "DESIGN REQUIREMENTS INFEASIBLE\n\n"
                f"{state.feasibility.message}\n\n"
                f"Conflicting values:\n{conflicts}\n\n"
                "Please revise the requirements. Analysis and topology were not started.\n\n"
                f"{questions}"
            )
        if state.candidate_fit is not None and not state.candidate_fit.fits:
            checks = "\n".join(
                f"- {c.name}: {c.status.value.upper()} — {c.message}" for c in state.candidate_fit.checks
            )
            questions = "\n".join(f"- {q.question}" for q in state.clarifications)
            return (
                "CANDIDATE GEOMETRY REJECTED\n\n"
                f"{state.candidate_fit.message}\n\n"
                f"{checks}\n\n"
                "STRUCTURE was not started. Revise geometry/design requirements.\n\n"
                f"{questions}"
            )
        questions = "\n".join(f"- {q.question}" for q in state.clarifications)
        return (
            f"Need more information before engineering can start.\n\n"
            f"{questions}"
        )
    return f"Requirements complete. Decision: {decision}. Continuing the design workflow."


def _ingest(message: str) -> None:
    orch: Orchestrator = st.session_state.orch
    orch.ingest_user_request(message)
    _append("user", message)
    _append("assistant", _assistant_after_interaction(orch.state))


def _answers_to_update() -> RequirementsUpdate:
    raw = dict(st.session_state.answers)
    payload: Dict[str, Any] = {}
    for field, value in raw.items():
        if value is None or value == "":
            continue
        if field in NUMERIC_FIELDS:
            try:
                payload[field] = float(value)
            except (TypeError, ValueError):
                continue
        elif field == "manufacturing_method" and str(value).lower() == "fdm":
            payload[field] = "3d_print"
        else:
            payload[field] = value
    return RequirementsUpdate.model_validate(payload)


def _queue_request_text(message: str) -> None:
    """Update the design-request box. Call only from on_click callbacks.

    Callbacks run before widgets are instantiated, so assigning `request_text`
    here is legal. `pending_request_prefill` is applied again at script start
    in case a later widget merge overwrites the callback assignment.
    """
    st.session_state.pending_request_prefill = message
    st.session_state.request_text = message


def _current_request_text() -> str:
    return (st.session_state.get("request_text") or "").strip()


def _on_happy_path() -> None:
    st.session_state.mode_geom = GEOM_GOLDEN
    _queue_request_text(HAPPY_PATH_MESSAGE)
    _reset(ingest_message=HAPPY_PATH_MESSAGE, prefill_happy=True)


def _on_missing_info() -> None:
    _queue_request_text(MISSING_INFO_MESSAGE)
    _reset(ingest_message=MISSING_INFO_MESSAGE, prefill_happy=False)


def _on_rejected() -> None:
    _queue_request_text(REJECTED_MESSAGE)
    _reset(ingest_message=REJECTED_MESSAGE, prefill_happy=False)


def _on_reset_session() -> None:
    st.session_state.mode_geom = GEOM_ADAPTIVE
    st.session_state.mode_reason = _default_reasoning_choice()
    _queue_request_text(HAPPY_PATH_MESSAGE)
    _reset(clear_image=True)


def _on_submit_request() -> None:
    message = _current_request_text()
    st.session_state.submitted_request = message
    if not message:
        st.session_state.ui_notice = "Write a design request first."
        return
    orch: Orchestrator = st.session_state.orch
    if orch.state.stage != WorkflowStage.REQUIREMENTS or orch.state.requirements is not None:
        _reset()
    _ingest(message)


def _on_continue_design() -> None:
    _continue_design()


def _sync_orch_fixtures() -> None:
    orch: Orchestrator = st.session_state.orch
    if orch.state.geometry is not None:
        return
    mode = st.session_state.get("mode_geom")
    orch.fixtures = fixtures_for_geometry_mode(mode)
    orch.imported_candidate = imported_candidate_for_mode(mode)


def _continue_design() -> None:
    orch: Orchestrator = st.session_state.orch
    _sync_orch_fixtures()
    if orch.state.stage == WorkflowStage.REQUIREMENTS and orch.state.requirements is None:
        message = _current_request_text()
        if not message:
            st.session_state.ui_notice = "Write a design request first."
            return
        st.session_state.submitted_request = message
        _ingest(message)
    if orch.state.stage == WorkflowStage.REQUEST_INFORMATION:
        update = _answers_to_update()
        _append("user", _format_answers(update))
        orch.apply_answers(update)
        _append("assistant", _assistant_after_interaction(orch.state))
    if orch.state.contract_error:
        _append(
            "assistant",
            format_mismatch_message(orch.state),
            error_key=chat_error_key(orch.state),
        )
        return
    if orch.state.stage not in (
        WorkflowStage.COMPLETE,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUEST_INFORMATION,
        WorkflowStage.DESIGN_REVIEW_FAILED,
    ):
        orch.run()
        if orch.state.contract_error:
            _append(
                "assistant",
                format_mismatch_message(orch.state),
                error_key=chat_error_key(orch.state),
            )
        else:
            _append("assistant", _run_summary(orch.state))


def _format_answers(update: RequirementsUpdate) -> str:
    data = {k: v for k, v in update.model_dump().items() if v is not None}
    if not data:
        return "(no answers provided)"
    lines = ["Clarification answers:"]
    for key, value in data.items():
        label = FIELD_LABELS.get(key, key)
        lines.append(f"- {label} = {value}")
    return "\n".join(lines)


def _run_summary(state: DesignState) -> str:
    if state.stage == WorkflowStage.REJECTED:
        return _assistant_after_interaction(state)
    if state.contract_error:
        return format_mismatch_message(state)
    if state.stage == WorkflowStage.REQUEST_INFORMATION:
        return _assistant_after_interaction(state)
    if state.stage == WorkflowStage.DESIGN_REVIEW_FAILED:
        n = len(state.design_iterations)
        return (
            f"DESIGN_REVIEW_FAILED after {n} structural iteration(s). "
            "Topology optimization was not started. "
            "Mock review is not physical safety certification."
        )
    lines = [f"Workflow stage: {state.stage.value}."]
    if state.registration is not None:
        lines.append(
            f"Registration: {'MOCK' if state.registration.is_mock else 'LIVE'} — "
            f"frame {state.registration.frame_id}."
        )
    if state.geometry is not None:
        g = state.geometry
        lines.append(
            f"Geometry: {'MOCK' if g.is_mock else 'LIVE'} — "
            f"payload {g.payload_object.kind}, "
            f"{g.payload_object.bottle_diameter_mm} mm, "
            f"desk {g.environment.desk_thickness_mm} mm."
        )
    if state.structure is not None:
        s = state.structure
        load = s.load_cases[0] if s.load_cases else None
        force = abs(load.force_N[2]) if load else None
        lines.append(
            f"Structure: iteration {s.iteration}, "
            f"thickness {s.parameters.support_thickness_mm:g} mm, "
            f"braces {s.parameters.brace_count}, "
            f"{len(s.nodes)} nodes, {len(s.members)} members, "
            f"load {load.load_case_id if load else 'n/a'}"
            + (f", gravity {force:.2f} N." if force is not None else ".")
        )
    if state.design_iterations:
        loop = []
        for item in state.design_iterations:
            loop.append(
                f"iter {item.iteration}: "
                f"t={item.structure.parameters.support_thickness_mm:g} mm, "
                f"braces={item.structure.parameters.brace_count}, "
                f"disp={item.analysis.max_displacement_mm:.2f} mm, "
                f"review={item.review.decision.value}"
            )
        lines.append("Design loop: " + " → ".join(loop))
    if state.analysis is not None:
        a = state.analysis
        lines.append(
            f"Analysis: {'MOCK' if a.is_mock else 'LIVE'} — "
            f"max stress {a.max_stress_pa:.0f} Pa, "
            f"FoS {a.factor_of_safety}, "
            f"safety validation={'YES' if a.is_safety_validation else 'NO'}."
        )
    if state.topology is not None:
        t = state.topology
        lines.append(
            f"Topology: {'MOCK' if t.is_mock else 'LIVE'} — "
            f"volume fraction {t.volume_fraction}."
        )
    if state.verification is not None:
        v = state.verification
        lines.append(
            f"Verification: artifacts complete={'YES' if v.artifacts_complete else 'NO'}, "
            f"safety validated={'YES' if v.safety_validated else 'NO'}, "
            f"status={state.safety_status.value.upper()}."
        )
    if engineering_evidence_is_mocked(state):
        lines.append("PHYSICAL SAFETY: UNVERIFIED — mock engineering evidence is present.")
    return "\n".join(lines)


def _badge_html(label: str) -> str:
    fg, bg = BADGE_COLORS.get(label, ("#111827", "#e5e7eb"))
    return (
        f'<span style="font-size:0.72rem;font-weight:700;letter-spacing:0.04em;'
        f'color:{fg};background:{bg};padding:2px 8px;border-radius:999px;">{label}</span>'
    )


def _structure_input_from_state(state: DesignState) -> Optional[StructureInput]:
    if state.geometry is None or state.requirements is None:
        return None
    if state.resolved_payload_mass is not None:
        mass = state.resolved_payload_mass.mass_kg
        provenance = state.resolved_payload_mass.provenance
    elif state.requirements.payload.filled_mass_kg is not None:
        mass = state.requirements.payload.filled_mass_kg
        provenance = MassProvenance.USER_REQUIREMENTS
    elif state.geometry.payload_object.filled_mass_kg is not None:
        mass = state.geometry.payload_object.filled_mass_kg
        provenance = MassProvenance.GEOMETRY_OUTPUT
    else:
        return None
    return StructureInput(
        requirements=state.requirements,
        geometry=state.geometry,
        payload_mass_kg=mass,
        payload_mass_provenance=provenance,
    )


def _compare_column(title: str, kind: str, result) -> None:
    output, error, latency = result
    st.markdown(f"**{title}**")
    if kind == "missing":
        ok = isinstance(output, InteractionResult)
        st.write("Schema validation:", "PASS" if ok else "FAIL")
        st.caption(f"latency {latency:.2f}s")
        if ok:
            st.write("decision:", output.decision.value)
            st.write("questions:", len(output.questions))
            for question in output.questions[:8]:
                st.write(f"- {question.field}: {question.question}")
            if output.reject_reason:
                st.write("reject:", output.reject_reason)
        else:
            st.error(error or "Did not match InteractionResult.")
        return
    ok = isinstance(output, StructureOutput)
    st.write("Schema validation:", "PASS" if ok else "FAIL")
    st.caption(f"latency {latency:.2f}s")
    if ok:
        load = output.load_cases[0] if output.load_cases else None
        st.write("nodes:", len(output.nodes))
        st.write("members:", len(output.members))
        st.write("load case:", load.load_case_id if load else "n/a")
        if output.load_paths:
            st.write("load path:", output.load_paths[0].description)
        st.write("concept:", output.concept)
    else:
        st.error(error or "Did not match StructureOutput.")


def _run_compare(task: str) -> None:
    orch: Orchestrator = st.session_state.orch
    state = orch.state
    k2 = get_provider("k2_horizon")
    grok = get_provider("grok")
    if task == "missing-information detection":
        message = ""
        if state.requirements and state.requirements.user_message:
            message = state.requirements.user_message
        else:
            message = st.session_state.submitted_request or _current_request_text()
        st.session_state.compare = {
            "task": task,
            "left": k2.detect_missing(message),
            "right": grok.detect_missing(message),
            "kind": "missing",
        }
        return
    inp = _structure_input_from_state(state)
    if inp is None:
        st.session_state.compare = {
            "task": task,
            "error": "Structure compare needs geometry plus a payload mass on DesignState.",
        }
        return
    st.session_state.compare = {
        "task": task,
        "left": k2.generate_structure(inp),
        "right": grok.generate_structure(inp),
        "kind": "structure",
    }


def _sync_reasoning_provider(choice: str) -> None:
    mapping = {
        "Mock": "mock",
        "K2 Horizon": "k2_horizon",
        "Grok": "grok",
        "Cursor": "cursor",
    }
    set_reasoning_provider(get_provider(mapping[choice]))


def _cursor_catalog() -> Tuple[List[Dict[str, Any]], str]:
    cached = st.session_state.get("cursor_catalog")
    cached_err = st.session_state.get("cursor_catalog_error")
    if cached is not None:
        return cached, cached_err or ""
    models, error = list_cursor_models()
    st.session_state.cursor_catalog = models
    st.session_state.cursor_catalog_error = error
    return models, error


def _ensure_select_default(key: str, value: str, options: List[str]) -> None:
    if not options or not value or value not in options:
        return
    if key not in st.session_state:
        st.session_state[key] = value


def _param_option_label(param: Dict[str, Any], value: str) -> str:
    for item in param.get("values") or []:
        if str(item.get("value")) == str(value):
            return str(item.get("display_name") or value)
    return value


def _render_catalog_params(
    model: Optional[Dict[str, Any]],
    key_prefix: str,
    hints: Dict[str, str],
) -> Dict[str, str]:
    if not model:
        return {}
    preferred = resolve_param_values(model, hints)
    selected: Dict[str, str] = {}
    for param in model.get("parameters") or []:
        pid = str(param.get("id") or "").strip()
        options = [
            str(item.get("value"))
            for item in (param.get("values") or [])
            if item.get("value") not in (None, "")
        ]
        if not pid or not options:
            continue
        widget_key = f"{key_prefix}{pid}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = preferred.get(pid, options[0])
        elif st.session_state[widget_key] not in options:
            st.session_state[widget_key] = preferred.get(pid, options[0])
        label = str(param.get("display_name") or pid)
        st.selectbox(
            label,
            options,
            key=widget_key,
            format_func=lambda value, _param=param: _param_option_label(_param, value),
            help=f"Catalog parameter `{pid}`",
        )
        value = st.session_state.get(widget_key)
        if value in options:
            selected[pid] = str(value)
    return selected


def _params_from_widgets(model: Optional[Dict[str, Any]], key_prefix: str) -> Dict[str, str]:
    if not model:
        return {}
    selected: Dict[str, str] = {}
    for param in model.get("parameters") or []:
        pid = str(param.get("id") or "").strip()
        options = [
            str(item.get("value"))
            for item in (param.get("values") or [])
            if item.get("value") not in (None, "")
        ]
        if not pid or not options:
            continue
        value = st.session_state.get(f"{key_prefix}{pid}")
        if value in options:
            selected[pid] = str(value)
    return selected


def _format_scene_observation(observation: SceneObservation) -> str:
    methods = ", ".join(observation.likely_attachment_methods) or "(none)"
    missing = "\n".join(f"- {item}" for item in observation.missing_measurements) or "- (none)"
    uncertain = "\n".join(f"- {item}" for item in observation.uncertainties) or "- (none)"
    return (
        "LIVE SCENE OBSERVATION\n\n"
        "Detected\n"
        f"- payload: {observation.detected_payload_type or '(unknown)'}\n"
        f"- support: {observation.detected_support_type or '(unknown)'}\n"
        f"- attachment options: {methods}\n\n"
        "Need from user\n"
        f"{missing}\n\n"
        "Uncertain\n"
        f"{uncertain}"
    )


def _render_candidate_panel(candidate, fit) -> None:
    st.markdown("**Candidate Geometry**")
    st.caption("external generated concept geometry — not reconstructed scene geometry.")
    mesh_cols = st.columns(4)
    mesh_cols[0].metric("Vertices", candidate.vertex_count or "n/a")
    mesh_cols[1].metric("Faces", candidate.face_count or "n/a")
    mesh_cols[2].metric("Watertight", "yes" if candidate.watertight else "no")
    mesh_cols[3].metric("Components", candidate.connected_components or "n/a")
    st.markdown("**Key dimensions**")
    st.write(
        {
            "inner diameter (mm)": candidate.inner_diameter_mm,
            "outer diameter (mm)": candidate.outer_diameter_mm,
            "holder height (mm)": candidate.holder_height_mm,
            "desk compatibility (mm)": (
                [candidate.compatible_desk_min_mm, candidate.compatible_desk_max_mm]
                if candidate.compatible_desk_min_mm is not None
                else None
            ),
        }
    )
    st.markdown("**Fit against requirements**")
    if fit is None:
        st.caption("Run Continue design after entering requirements to compute fit.")
        return
    by_name = {check.name: check for check in fit.checks}

    def _status(name: str) -> str:
        check = by_name.get(name)
        if check is None:
            return "N/A"
        if check.status.value == "n/a":
            return "N/A"
        return check.status.value.upper()

    st.write(
        {
            "payload fit": _status("payload_fit"),
            "desk fit": _status("desk_fit"),
            "envelope fit": _status("envelope_fit"),
            "overall": "PASS" if fit.fits else "FAIL",
        }
    )


def _render_scene_observation(observation: SceneObservation, meta: Dict[str, Any]) -> None:
    st.markdown("**LIVE SCENE OBSERVATION**")
    if meta:
        model_name = meta.get("model") or "(unknown model)"
        latency = meta.get("latency")
        latency_txt = f"{latency:.2f}s" if isinstance(latency, (int, float)) else "n/a"
        params = meta.get("params") or {}
        param_txt = ", ".join(f"{k}={v}" for k, v in params.items()) or "(catalog defaults)"
        image_flag = meta.get("image_accepted")
        image_txt = "yes" if image_flag else ("no" if image_flag is False else "n/a")
        valid = "PASS" if meta.get("schema_valid") else "FAIL"
        st.caption(
            f"model `{model_name}` · {param_txt} · latency {latency_txt} · "
            f"image accepted {image_txt} · schema {valid} · source `{observation.source}`"
        )
    st.markdown("Detected")
    methods = ", ".join(observation.likely_attachment_methods) or "(none)"
    st.write(f"- payload: {observation.detected_payload_type or '(unknown)'}")
    st.write(f"- support: {observation.detected_support_type or '(unknown)'}")
    st.write(f"- attachment options: {methods}")
    if observation.likely_attachment_regions:
        st.write("- attachment regions: " + ", ".join(observation.likely_attachment_regions))
    st.markdown("Need from user")
    if observation.missing_measurements:
        for item in observation.missing_measurements:
            st.write(f"- {item}")
    else:
        st.write("- (none)")
    st.markdown("Uncertain")
    if observation.uncertainties:
        for item in observation.uncertainties:
            st.write(f"- {item}")
    else:
        st.write("- (none)")
    st.caption(
        "Inferred values are not copied into authoritative requirements. "
        "InteractionAgent still asks for missing engineering measurements."
    )
    with st.expander("Raw structured model output"):
        st.json(observation.model_dump(mode="json"))


def _analyze_image_and_requirements() -> None:
    if st.session_state.get("mode_reason") != "Cursor":
        st.session_state.ui_notice = (
            "Select Reasoning Provider = Cursor to analyze image and requirements."
        )
        return
    image_bytes = st.session_state.get("image_bytes")
    message = _current_request_text()
    if not image_bytes:
        st.session_state.ui_notice = "Upload a desk / bottle image first."
        return
    if not message:
        st.session_state.ui_notice = "Write a design request first."
        return
    provider = get_provider("cursor")
    catalog, _catalog_error = _cursor_catalog()
    model_id = st.session_state.get("cursor_model") or ""
    model_entry = find_catalog_model(catalog, model_id)
    model_params = _params_from_widgets(model_entry, "cursor_param_")
    call = provider.observe_scene(
        image_bytes,
        message,
        model_id=model_id,
        image_name=st.session_state.get("image_name"),
        model_params=model_params,
        catalog=catalog,
    )
    observation = call.observation
    error = call.error
    latency = call.latency_s
    used_model = call.model_id or model_id
    st.session_state.scene_observation_meta = {
        "model": used_model,
        "params": call.params,
        "latency": latency,
        "error": error,
        "image_accepted": call.image_accepted,
        "schema_valid": call.schema_valid,
    }
    if observation is None:
        st.session_state.scene_observation = None
        st.session_state.ui_notice = error or "Cursor live reasoning not connected"
        return
    # Policy: SceneObservation never writes inferred dimensions into DesignState.
    assert observation_to_requirements_update(observation) == {}
    st.session_state.scene_observation = observation
    orch: Orchestrator = st.session_state.orch
    if orch.state.stage != WorkflowStage.REQUIREMENTS or orch.state.requirements is not None:
        _reset()
        st.session_state.scene_observation = observation
        st.session_state.scene_observation_meta = {
            "model": used_model,
            "params": call.params,
            "latency": latency,
            "error": "",
            "image_accepted": call.image_accepted,
            "schema_valid": call.schema_valid,
        }
    st.session_state.submitted_request = message
    orch = st.session_state.orch
    orch.ingest_user_request(message)
    _append("user", message)
    _append("assistant", _format_scene_observation(observation))
    _append("assistant", _assistant_after_interaction(orch.state))


def _run_cursor_compare() -> None:
    image_bytes = st.session_state.get("image_bytes")
    message = st.session_state.submitted_request or _current_request_text()
    catalog, _catalog_error = _cursor_catalog()
    model_a = st.session_state.get("cursor_compare_a") or st.session_state.get("cursor_model") or ""
    model_b = st.session_state.get("cursor_compare_b") or ""
    if not image_bytes:
        st.session_state.cursor_compare = {"error": "Upload an image first."}
        return
    if not message:
        st.session_state.cursor_compare = {"error": "Write a design request first."}
        return
    if not model_a or not model_b:
        st.session_state.cursor_compare = {
            "error": "Select two catalog model IDs before comparing."
        }
        return
    provider = get_provider("cursor")
    entry_a = find_catalog_model(catalog, model_a)
    entry_b = find_catalog_model(catalog, model_b)
    left = provider.observe_scene(
        image_bytes,
        message,
        model_id=model_a,
        image_name=st.session_state.get("image_name"),
        model_params=_params_from_widgets(entry_a, "cursor_compare_a_param_"),
        catalog=catalog,
    )
    right = provider.observe_scene(
        image_bytes,
        message,
        model_id=model_b,
        image_name=st.session_state.get("image_name"),
        model_params=_params_from_widgets(entry_b, "cursor_compare_b_param_"),
        catalog=catalog,
    )
    st.session_state.cursor_compare = {
        "left": left,
        "right": right,
        "model_a": model_a,
        "model_b": model_b,
    }


def _cursor_compare_column(title: str, result) -> None:
    st.markdown(f"**{title}**")
    ok = result.schema_valid and result.observation is not None
    st.write("Schema validation:", "PASS" if ok else "FAIL")
    param_txt = ", ".join(f"{k}={v}" for k, v in (result.params or {}).items()) or "(none)"
    st.caption(
        f"model `{result.model_id or '(n/a)'}` · {param_txt} · "
        f"latency {result.latency_s:.2f}s · image accepted "
        f"{'yes' if result.image_accepted else ('no' if result.image_accepted is False else 'n/a')}"
    )
    observation = result.observation
    if not ok:
        st.error(result.error or "Did not match SceneObservation.")
        return
    st.write("confidence:", observation.confidence)
    st.write("missing measurements:", observation.missing_measurements or "(none)")
    st.write("inferred fields:", observation.inferred_values or {})
    st.write("payload:", observation.detected_payload_type)
    st.write("support:", observation.detected_support_type)


st.set_page_config(page_title="Mechanical Design Copilot", layout="wide")
_init_session()
_apply_pending_prefills()
if st.session_state.pop("pending_analyze", False):
    with st.spinner("Analyzing image and requirements with Cursor..."):
        _analyze_image_and_requirements()

st.markdown(
    """
    <style>
    .stage-card {padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;margin-bottom:8px;background:#fff;}
    .stage-card.current {border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,0.15);}
    .stage-title {font-weight:700;font-size:0.92rem;margin-bottom:4px;}
    .arrow {color:#9ca3af;text-align:center;margin:0 0 8px 0;font-size:0.85rem;}
    .warn {background:#fff7ed;border:1px solid #fdba74;padding:10px 12px;border-radius:8px;color:#9a3412;}
    .danger {background:#fef2f2;border:1px solid #fca5a5;padding:10px 12px;border-radius:8px;color:#991b1b;}
    .loop-card {padding:8px 10px;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:6px;background:#f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo scenarios")
    st.button(
        "Load Happy Path",
        use_container_width=True,
        on_click=_on_happy_path,
        help="Regression / integration test using the golden fixture.",
    )
    st.button(
        "Load Missing Information Case",
        use_container_width=True,
        on_click=_on_missing_info,
    )
    st.button(
        "Load Rejected Case",
        use_container_width=True,
        on_click=_on_rejected,
    )
    st.button(
        "Reset session",
        use_container_width=True,
        on_click=_on_reset_session,
    )

    st.divider()
    st.header("Component Mode")
    st.caption(
        "Default custom workflow uses Cursor live scene reasoning when available and "
        "Adaptive Synthetic Geometry. Registration, Analysis, Topology, and CAD remain mocked. "
        "Golden Fixture is a regression / integration test, not the normal experiment."
    )
    st.radio("Registration", ["Mock Fixture", "Live"], index=0, disabled=True, key="mode_reg")
    st.caption("Live implementation not connected yet.")
    st.radio(
        "Geometry source",
        GEOM_MODE_OPTIONS,
        key="mode_geom",
        format_func=_geometry_mode_label,
        help="Adaptive is the default experiment. Imported Candidate loads the external OBJ. Golden Fixture is regression only. Live is not connected.",
    )
    if st.session_state.mode_geom == GEOM_LIVE:
        st.caption("Live Geometry — not connected.")
    elif st.session_state.mode_geom == GEOM_GOLDEN:
        assumptions = golden_fixture_assumptions()
        st.markdown("**Regression / integration test**")
        st.caption(
            f"Assumes payload {assumptions['payload_mass_kg']} kg, "
            f"diameter {assumptions['bottle_diameter_mm']} mm, "
            f"desk {assumptions['desk_thickness_mm']} mm, "
            f"protrusion {assumptions['max_protrusion_mm']} mm, "
            f"{assumptions['attachment']}, {assumptions['manufacturing']}."
        )
    elif st.session_state.mode_geom == GEOM_IMPORTED:
        st.caption(geometry_provenance_text(GEOM_IMPORTED))
        st.caption("Uses cupholder_dimensions.txt, cupholder_single_piece_PLA.obj, and cupholder_surface_particles.obj.")
    else:
        st.caption(geometry_provenance_text(GEOM_ADAPTIVE))
    st.radio("Analysis", ["Mock Fixture", "Live"], index=0, disabled=True, key="mode_analysis")
    st.caption("Live implementation not connected yet.")
    st.radio("Topology", ["Mock Fixture", "Live"], index=0, disabled=True, key="mode_topo")
    st.caption("Live implementation not connected yet.")
    reasoning_choice = st.radio(
        "Reasoning Provider",
        REASONING_CHOICES,
        key="mode_reason",
        help="Cursor is used only for live image + requirement understanding. Workflow decisions stay in InteractionAgent.",
    )
    _sync_reasoning_provider(reasoning_choice)
    k2 = get_provider("k2_horizon")
    grok = get_provider("grok")
    cursor = get_provider("cursor")
    if not k2.configured:
        st.caption(k2.not_connected_reason)
    if not grok.configured:
        st.caption(grok.not_connected_reason)
    if reasoning_choice == "Cursor":
        if not cursor.configured:
            st.caption(cursor.not_connected_reason or "Cursor live reasoning not connected")
        else:
            catalog, catalog_error = _cursor_catalog()
            model_ids = [item["id"] for item in catalog if item.get("id")]
            if catalog_error:
                st.caption(catalog_error)
            if model_ids:
                default_id = pick_catalog_model_id(catalog, DEFAULT_MODEL_ID)
                _ensure_select_default("cursor_model", default_id, model_ids)
                st.selectbox(
                    "Cursor model",
                    model_ids,
                    key="cursor_model",
                    help="IDs from Cursor.models.list() for this account. Not guessed.",
                )
                selected_id = st.session_state.get("cursor_model") or default_id
                if st.session_state.get("cursor_model_params_for") != selected_id:
                    for key in list(st.session_state.keys()):
                        if key.startswith("cursor_param_"):
                            del st.session_state[key]
                    st.session_state.cursor_model_params_for = selected_id
                selected = find_catalog_model(catalog, selected_id) or catalog[0]
                _render_catalog_params(selected, "cursor_param_", DEFAULT_PARAM_HINTS)
            else:
                st.caption("No Cursor model IDs discovered for this account.")
    elif not cursor.configured:
        st.caption("Cursor live reasoning not connected")

    st.divider()
    st.header("Compare reasoning")
    st.caption(
        "Same DesignState, two providers, same Pydantic schema. "
        "Does not replace the orchestrator."
    )
    task = st.selectbox(
        "Task",
        ["missing-information detection", "structural concept generation"],
    )
    if st.button("Run K2 vs Grok", use_container_width=True):
        _run_compare(task)
    if st.session_state.compare:
        payload = st.session_state.compare
        st.markdown(f"**Task:** {payload.get('task')}")
        if payload.get("error"):
            st.warning(payload["error"])
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                _compare_column("K2 Horizon", payload["kind"], payload["left"])
            with col_b:
                _compare_column("Grok", payload["kind"], payload["right"])

    st.divider()
    st.header("Compare Cursor models")
    st.caption(
        "Same image + prompt, two catalog IDs, SceneObservation schema. "
        "Does not replace InteractionAgent."
    )
    cursor_for_compare = get_provider("cursor")
    if not cursor_for_compare.configured:
        st.caption("Cursor live reasoning not connected")
    else:
        cursor_catalog, catalog_error = _cursor_catalog()
        cursor_ids = [item["id"] for item in cursor_catalog if item.get("id")]
        if catalog_error:
            st.caption(catalog_error)
        if len(cursor_ids) < 2:
            st.caption("Need at least two catalog model IDs to compare.")
        else:
            default_a = pick_catalog_model_id(cursor_catalog, COMPARE_A_MODEL_ID)
            default_b = pick_catalog_model_id(cursor_catalog, COMPARE_B_MODEL_ID)
            if default_b == default_a and len(cursor_ids) > 1:
                default_b = next(
                    (item for item in cursor_ids if item != default_a),
                    cursor_ids[1],
                )
            _ensure_select_default("cursor_compare_a", default_a, cursor_ids)
            _ensure_select_default("cursor_compare_b", default_b, cursor_ids)
            st.selectbox("Cursor model A", cursor_ids, key="cursor_compare_a")
            selected_a = st.session_state.get("cursor_compare_a") or default_a
            if st.session_state.get("cursor_compare_a_params_for") != selected_a:
                for key in list(st.session_state.keys()):
                    if key.startswith("cursor_compare_a_param_"):
                        del st.session_state[key]
                st.session_state.cursor_compare_a_params_for = selected_a
            entry_a = find_catalog_model(cursor_catalog, selected_a)
            _render_catalog_params(entry_a, "cursor_compare_a_param_", COMPARE_A_PARAM_HINTS)
            st.selectbox(
                "Cursor model B",
                cursor_ids,
                key="cursor_compare_b",
            )
            selected_b = st.session_state.get("cursor_compare_b") or default_b
            if st.session_state.get("cursor_compare_b_params_for") != selected_b:
                for key in list(st.session_state.keys()):
                    if key.startswith("cursor_compare_b_param_"):
                        del st.session_state[key]
                st.session_state.cursor_compare_b_params_for = selected_b
            entry_b = find_catalog_model(cursor_catalog, selected_b)
            _render_catalog_params(entry_b, "cursor_compare_b_param_", COMPARE_B_PARAM_HINTS)
            if st.button("Run Cursor A vs B", use_container_width=True):
                _run_cursor_compare()
    if st.session_state.cursor_compare:
        payload = st.session_state.cursor_compare
        if payload.get("error"):
            st.warning(payload["error"])
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                _cursor_compare_column(payload.get("model_a") or "Model A", payload["left"])
            with col_b:
                _cursor_compare_column(payload.get("model_b") or "Model B", payload["right"])

orch: Orchestrator = st.session_state.orch
state = orch.state
status = pipeline_status(state)
current = highlighted_stage(state)

st.title("Mechanical Design Copilot")
st.caption(
    "Prototype UI over the cup-holder orchestrator. "
    "Engineering numbers from fixtures are simulated, not physical validation."
)

if engineering_evidence_is_mocked(state) or (
    state.verification is not None and not state.verification.safety_validated
):
    st.markdown(
        '<div class="warn"><b>SIMULATED / MOCK DATA</b> — '
        "PHYSICAL SAFETY: UNVERIFIED. Mock FEM or topology output does not "
        "validate a physical design.</div>",
        unsafe_allow_html=True,
    )
if state.stage == WorkflowStage.REJECTED:
    st.markdown(
        f'<div class="danger"><b>REJECTED</b> — {state.reject_reason or "out of scope"}'
        "<br>Geometry / FEM / topology were not executed.</div>",
        unsafe_allow_html=True,
    )
if (
    state.feasibility is not None
    and not state.feasibility.feasible
    and state.stage == WorkflowStage.REQUEST_INFORMATION
):
    conflicts = "<br>".join(
        f"• {v.code}: {v.message}" for v in state.feasibility.violations
    )
    st.markdown(
        '<div class="danger"><b>DESIGN REQUIREMENTS INFEASIBLE</b><br>'
        f"{state.feasibility.message}<br>{conflicts}"
        "<br>Revise the requirements. Analysis and Topology remain NOT STARTED.</div>",
        unsafe_allow_html=True,
    )
if (
    state.candidate_fit is not None
    and not state.candidate_fit.fits
    and state.stage == WorkflowStage.REQUEST_INFORMATION
):
    checks = "<br>".join(
        f"• {c.name}: {c.status.value.upper()} — {c.message}" for c in state.candidate_fit.checks
    )
    st.markdown(
        '<div class="danger"><b>CANDIDATE GEOMETRY REJECTED</b><br>'
        f"{state.candidate_fit.message}<br>{checks}"
        "<br>STRUCTURE was not started.</div>",
        unsafe_allow_html=True,
    )
if state.stage == WorkflowStage.DESIGN_REVIEW_FAILED:
    st.markdown(
        '<div class="danger"><b>DESIGN_REVIEW_FAILED</b> — iteration budget exhausted '
        "without PASS. Topology optimization was not started.</div>",
        unsafe_allow_html=True,
    )

left, center, right = st.columns([1.05, 1.35, 1.1])

with left:
    st.subheader("User / Chat")
    uploaded = st.file_uploader("Upload a photo of the desk / bottle", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        st.session_state.image_bytes = uploaded.getvalue()
        st.session_state.image_name = uploaded.name
    st.text_area("Design request", key="request_text", height=90)
    st.button(
        "Submit request",
        use_container_width=True,
        on_click=_on_submit_request,
    )
    if st.button("Analyze image and requirements", use_container_width=True):
        st.session_state.pending_analyze = True
        st.rerun()
    if st.session_state.ui_notice:
        st.warning(st.session_state.ui_notice)
        st.session_state.ui_notice = ""

    if st.session_state.scene_observation is not None:
        _render_scene_observation(
            st.session_state.scene_observation,
            st.session_state.scene_observation_meta or {},
        )
    elif (st.session_state.scene_observation_meta or {}).get("error"):
        st.caption(st.session_state.scene_observation_meta["error"])

    st.markdown("**Conversation**")
    if not st.session_state.chat:
        st.caption("No messages yet. Submit a request or load a demo scenario.")
    for item in st.session_state.chat:
        with st.chat_message(item["role"] if item["role"] in ("user", "assistant") else "assistant"):
            st.markdown(item["text"])

    if state.stage == WorkflowStage.REQUEST_INFORMATION and state.clarifications:
        st.markdown("**Clarification questions**")
        st.caption("These questions come from InteractionAgent via the Orchestrator.")
        for question in state.clarifications:
            field = question.field
            widget_key = f"ans_{field}"
            current_val = st.session_state.answers.get(field)
            if widget_key not in st.session_state:
                if field in NUMERIC_FIELDS:
                    st.session_state[widget_key] = (
                        float(current_val) if current_val not in (None, "") else 0.0
                    )
                elif field in SELECT_FIELDS:
                    options = SELECT_FIELDS[field]
                    st.session_state[widget_key] = (
                        current_val if current_val in options else options[0]
                    )
                else:
                    st.session_state[widget_key] = (
                        str(current_val) if current_val not in (None, "") else ""
                    )
            label = FIELD_LABELS.get(field, question.question)
            if field in NUMERIC_FIELDS:
                st.number_input(label, key=widget_key, help=question.question)
            elif field in SELECT_FIELDS:
                st.selectbox(
                    label, SELECT_FIELDS[field], key=widget_key, help=question.question
                )
            else:
                st.text_input(label, key=widget_key, help=question.question)
            st.session_state.answers[field] = st.session_state[widget_key]
        if "no_drill" not in st.session_state:
            st.session_state.no_drill = True
        extra = st.checkbox("No drilling allowed", key="no_drill")
        if extra:
            st.session_state.answers["attachment_notes"] = "clamp only, no drilling"

    st.button(
        "Continue design",
        type="primary",
        use_container_width=True,
        on_click=_on_continue_design,
    )

with center:
    st.subheader("Design view")
    if st.session_state.image_bytes:
        st.image(st.session_state.image_bytes, caption=st.session_state.image_name, use_container_width=True)
    else:
        st.info("No photo uploaded. The fixture workflow does not require an image yet.")

    if state.requirements is not None:
        req = state.requirements
        st.markdown("**Interpreted requirements**")
        st.write(
            {
                FIELD_LABELS["filled_bottle_mass_kg"]: req.payload.filled_mass_kg,
                "Volume (L)": req.payload.volume_l,
                FIELD_LABELS["bottle_diameter_mm"]: req.object_geometry.bottle_diameter_mm,
                FIELD_LABELS["desk_thickness_mm"]: req.environment.desk_thickness_mm,
                FIELD_LABELS["attachment_method"]: req.attachment.method,
                FIELD_LABELS["allowed_contact_region"]: req.attachment.allowed_contact_region,
                FIELD_LABELS["max_protrusion_mm"]: req.design_envelope.max_protrusion_mm,
                FIELD_LABELS["manufacturing_method"]: req.manufacturing.method,
                FIELD_LABELS["material"]: req.manufacturing.material,
            }
        )

    if state.contract_error:
        st.markdown("**Why the workflow blocked**")
        rows = requirement_geometry_rows(state)
        if rows:
            st.markdown("Requirement vs Geometry")
            for row in rows:
                op = "≠" if row["disagree"] else "="
                unit = " mm" if "(mm)" in row["label"] else (" kg" if "(kg)" in row["label"] else "")
                st.write(
                    f"{row['label']}: **{row['requirement']}{unit}** {op} **{row['geometry']}{unit}**"
                )
        st.info(
            "The system refuses to silently choose between conflicting engineering inputs."
        )
        st.error(state.contract_error)

    if (
        state.feasibility is not None
        and not state.feasibility.feasible
        and not state.contract_error
    ):
        st.markdown("**Why the design cannot proceed**")
        st.error("DESIGN REQUIREMENTS INFEASIBLE")
        for violation in state.feasibility.violations:
            st.write(f"- `{violation.code}`: {violation.message}")
        st.caption("Analysis and Topology remain NOT STARTED until requirements are revised.")

    if (
        state.candidate_fit is not None
        and not state.candidate_fit.fits
        and not state.contract_error
    ):
        st.markdown("**Why the candidate was rejected**")
        st.error("CANDIDATE GEOMETRY REJECTED")
        for check in state.candidate_fit.checks:
            st.write(f"- `{check.name}`: **{check.status.value.upper()}** — {check.message}")
        st.caption("STRUCTURE remains NOT STARTED until the candidate fits the requirements.")

    if state.geometry is not None:
        st.info(geometry_provenance_text(st.session_state.get("mode_geom")))

    candidate = state.imported_candidate or imported_candidate_for_mode(
        st.session_state.get("mode_geom")
    )
    if candidate is not None:
        st.markdown("**IMPORTED CANDIDATE MESH**")
        st.caption(candidate.provenance)
        representation = st.radio(
            "Candidate representation",
            ["Surface Mesh", "Point / Particle Representation"],
            horizontal=True,
            key="candidate_representation",
        )
        if representation == "Point / Particle Representation":
            st.plotly_chart(
                imported_candidate_particle_figure(candidate),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                imported_candidate_mesh_figure(candidate),
                use_container_width=True,
            )
        _render_candidate_panel(candidate, state.candidate_fit)

    if state.geometry is not None and state.structure is None and candidate is None:
        st.plotly_chart(geometry_figure(state.geometry), use_container_width=True)
        st.caption("Simplified environment + payload. Not reconstructed CAD.")
    if state.structure is not None:
        st.plotly_chart(
            structure_figure(state.structure, state.geometry),
            use_container_width=True,
        )
        s = state.structure
        load = s.load_cases[0] if s.load_cases else None
        st.write(
            f"iteration {s.iteration} · thickness {s.parameters.support_thickness_mm:g} mm · "
            f"braces {s.parameters.brace_count} · "
            f"{len(s.nodes)} nodes · {len(s.members)} members · "
            f"load case `{load.load_case_id if load else 'n/a'}`"
        )
        if load:
            st.write(f"gravity force: {abs(load.force_N[2]):.2f} N")

    loop_cards = iteration_cards(state)
    if loop_cards:
        st.markdown("**Design Iterations**")
        st.caption("Deterministic mock structure/analysis loop. Not physical safety certification.")
        for card in loop_cards:
            changes = card["requested_changes"]
            if changes:
                change_text = "; ".join(
                    f"{c['action']} {c['parameter']} ({c['reason']})" for c in changes
                )
            else:
                change_text = "none"
            st.markdown(
                f'<div class="loop-card"><b>Iteration {card["iteration"]}</b><br>'
                f"support thickness: {card['support_thickness_mm']:g} mm<br>"
                f"brace count: {card['brace_count']}<br>"
                f"displacement: {card['max_displacement_mm']:.2f} mm<br>"
                f"stress: {card['max_stress_mpa']:.2f} MPa<br>"
                f"review: {card['review']}<br>"
                f"requested changes: {change_text}</div>",
                unsafe_allow_html=True,
            )

    if state.topology is not None:
        st.markdown("**Topology optimization**")
        t = state.topology
        if t.is_mock:
            st.warning("MOCK PLACEHOLDER — Shohom fixture. No printable mesh yet.")
        st.write(
            {
                "model": t.model,
                "volume_fraction": t.volume_fraction,
                "mass_reduction_pct": t.mass_reduction_pct,
                "geometry_ref": t.optimized_geometry_ref,
                "is_mock": t.is_mock,
            }
        )
        mesh_path = Path(t.optimized_geometry_ref)
        if mesh_path.suffix.lower() in {".stl", ".obj"} and mesh_path.exists():
            st.success(f"Mesh available for later display: {mesh_path}")
        else:
            st.caption("STL/OBJ hook: display a real mesh here when a file path exists.")

    if state.cad is not None:
        cad = state.cad
        st.markdown("**CAD / printable output**")
        st.write({"filename": cad.filename, "format": cad.format, "is_mock": cad.is_mock})
        cad_path = Path(cad.filename)
        if cad_path.suffix.lower() in {".stl", ".obj"} and cad_path.exists():
            st.success(f"CAD file ready: {cad_path}")
        else:
            st.caption("Future STL hook: no real CAD file on disk.")
        if cad.is_mock:
            st.warning("SIMULATED / MOCK DATA — PHYSICAL SAFETY: UNVERIFIED")

with right:
    st.subheader("Engineering workflow")
    timeline = design_loop_timeline(state)
    if timeline:
        for index, step in enumerate(timeline):
            klass = "stage-card"
            st.markdown(
                f'<div class="{klass}"><div class="stage-title">{step["title"]} '
                f'{_badge_html(step["badge"])}</div></div>',
                unsafe_allow_html=True,
            )
            if index < len(timeline) - 1:
                st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
        st.markdown("---")
    for index, name in enumerate(PIPELINE):
        badge = status[name]
        klass = "stage-card current" if name == current else "stage-card"
        st.markdown(
            f'<div class="{klass}"><div class="stage-title">{name} {_badge_html(badge)}</div></div>',
            unsafe_allow_html=True,
        )
        if index < len(PIPELINE) - 1:
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

    st.markdown("**Inspect input / output**")
    cards = inspect_cards(state)
    if not cards:
        st.caption("No completed stages yet.")
    for card in cards:
        with st.expander(f"{card['title']} — inspect"):
            mock_flag = "MOCK" if card["is_mock"] else "LIVE / OURS"
            st.write("component owner:", card["owner"])
            st.write("mode:", mock_flag)
            if card["provenance"]:
                st.write("provenance:", card["provenance"])
            st.write("input schema:", card["input_schema"])
            st.json(card["input"])
            st.write("output schema:", card["output_schema"])
            st.json(card["output"])
            if card["is_mock"] and card["title"] in {
                "ANALYSIS",
                "DESIGN REVIEW",
                "TOPOLOGY OPTIMIZATION",
                "CAD OUTPUT",
            }:
                st.warning("SIMULATED / MOCK DATA — PHYSICAL SAFETY: UNVERIFIED")
            with st.expander("Raw JSON (debug)"):
                dump = None
                mapping = {
                    "REQUIREMENTS": state.requirements,
                    "REGISTRATION": state.registration,
                    "GEOMETRY": state.geometry,
                    "CANDIDATE FIT": state.candidate_fit,
                    "FEASIBILITY": state.feasibility,
                    "STRUCTURE": state.structure,
                    "ANALYSIS": state.analysis,
                    "DESIGN REVIEW": state.design_review,
                    "TOPOLOGY OPTIMIZATION": state.topology,
                    "VERIFICATION": state.verification,
                    "CAD OUTPUT": state.cad,
                }
                obj = mapping.get(card["title"])
                dump = obj.model_dump(mode="json") if obj is not None else {}
                st.code(json.dumps(dump, indent=2)[:8000])

st.divider()
with st.expander("Execution Trace", expanded=False):
    st.caption("Rendered from Orchestrator.trace, not a hand-maintained UI list.")
    if not orch.trace:
        st.write("No events yet.")
    for event in orch.trace:
        fields = ", ".join(event.fields_changed) if event.fields_changed else "(none)"
        st.markdown(
            f"**{event.stage_executed.value.upper()}**  \n"
            f"{fields} changed  \n"
            f"→ `{event.next_stage.value}`  \n"
            f"<span style='color:#6b7280;font-size:0.85rem'>"
            f"{event.component} · {event.action}"
            f"{' · ' + event.notes if event.notes else ''}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
