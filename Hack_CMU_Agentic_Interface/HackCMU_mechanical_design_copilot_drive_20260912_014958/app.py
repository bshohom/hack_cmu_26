"""Mechanical Design Copilot — thin Streamlit view over the existing orchestrator."""

from __future__ import annotations

import base64
import html
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st
import streamlit.components.v1 as components

from fixtures import load_clarifications, load_user_request
from geometry_sources import (
    GEOM_ADAPTIVE,
    GEOM_GOLDEN,
    GEOM_GENERATED,
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
    list_cursor_models,
    observation_to_requirements_update,
    pick_catalog_model_id,
    resolve_param_values,
)
from orchestrator import Orchestrator
from imported_candidate import CANDIDATES
from providers import get_provider
from reasoning import (
    clear_reasoning_traces,
    reasoning_traces,
    set_reasoning_mode,
    set_reasoning_provider,
)
from reasoning_contracts import ReasoningMode, ReasoningOutcome
from tools.warmstart import generate_warm_start
from tools.registration import (
    CAPTURE_GUIDANCE,
    DEFAULT_OUT_ROOT as REGISTRATION_OUT_ROOT,
    MAX_IMAGES,
    MIN_IMAGES,
    RECOMMENDED_IMAGES,
    UploadedPhoto,
    run_sample_registration,
    run_registration,
    stage_uploads,
)
from schemas import (
    InteractionResult,
    MassProvenance,
    MissingInformation,
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
from ui_failure import STAGE_RECONSTRUCTION, detect_ui_failure
from ui_retry import (
    REGENERATE_DESIGN,
    RETRY_TOPOLOGY,
    progress_caption,
    retry_action,
)
from ui_run_log import requirements_fingerprint, should_reuse_warm_start
from ui_flow import (
    ActionSpec,
    WORKSPACE_TAB_OPTIMIZATION,
    WORKSPACE_TAB_SCENE,
    WORKSPACE_TABS,
    action_spec,
    backend_missing_fields,
    design_summary_rows,
    missing_detail_count,
    need_details_title,
    reconstructed_scene_status,
    scene_status_text,
    stepper_states,
    workspace_tab_after_event,
)
from ui_viz import (
    geometry_figure,
    imported_candidate_mesh_figure,
    imported_candidate_particle_figure,
    optimized_design_figure,
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
    "supported_load_kg",
    "payload_size_mm",
    "required_reach_mm",
    "wall_clearance_mm",
}
SELECT_FIELDS = {
    "attachment_method": ["clamp", "screws", "adhesive"],
    "manufacturing_method": ["3d_print", "fdm", "sla"],
}

# Presentation only. Widgets still appear only when the backend asks for the field.
FIELD_COPY: Dict[str, Tuple[str, Optional[str]]] = {
    "filled_bottle_mass_kg": ("Payload mass", "kg"),
    "bottle_diameter_mm": ("Payload size", "mm"),
    "bottle_height_mm": ("Payload height", "mm"),
    "desk_thickness_mm": ("Desk thickness", "mm"),
    "attachment_method": ("How should it attach?", None),
    "allowed_contact_region": ("Where should it attach?", None),
    "max_protrusion_mm": ("Maximum reach", "mm"),
    "manufacturing_method": ("How will it be made?", None),
    "material": ("Material", None),
    "max_part_mass_kg": ("Max part mass", "kg"),
    "supported_load_kg": ("Supported load", "kg"),
    "payload_size_mm": ("Object size", "mm"),
    "required_reach_mm": ("Required reach", "mm"),
    "wall_clearance_mm": ("Clearance", "mm"),
    "attachment_structure": ("What should it attach to?", None),
    "handle_location": ("Where should the handle be?", None),
    "mounting_region": ("Where can it mount?", None),
    "drilling_allowed": ("Is drilling allowed?", None),
}
CATEGORY_CHOICES: Dict[str, List[Tuple[str, str]]] = {
    "attachment_method": [
        ("clamp", "Clamp"),
        ("screws", "Screws"),
        ("adhesive", "Adhesive"),
        ("other", "Other"),
    ],
    "allowed_contact_region": [
        ("desk_front_edge", "Front edge"),
        ("desk_side_edge", "Side edge"),
        ("desk_underside", "Underneath"),
        ("desk_top", "Top surface"),
        ("other", "Other"),
    ],
    "manufacturing_method": [
        ("3d_print", "3D print"),
        ("fdm", "FDM"),
        ("sla", "SLA"),
        ("other", "Other"),
    ],
    "attachment_structure": [
        ("bed_frame", "Bed frame"),
        ("headboard", "Headboard"),
        ("mattress_edge", "Mattress edge"),
        ("wall", "Wall"),
        ("other", "Other"),
    ],
    "handle_location": [
        ("bedside", "Bedside"),
        ("headboard", "Headboard"),
        ("mattress_edge", "Mattress edge"),
        ("other", "Other"),
    ],
    "mounting_region": [
        ("bed_frame", "Bed frame"),
        ("headboard", "Headboard"),
        ("wall", "Wall"),
        ("other", "Other"),
    ],
}
_NUMERIC_SUFFIXES = ("_mm", "_kg", "_n", "_pa", "_s")
_BOOL_PREFIXES = ("is_", "has_", "no_", "allow_")

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


ACTION_ANCHOR_ID = "mdc-current-action"
API_NOT_CONNECTED = "API not connected"
GENERATION_FAILED_VALIDATION = "Starting design needs revision"
BRAND_TITLE = "On TOP of the World"
BRAND_TAGLINE = "Snap it. TOPtimize it. Print it."
BRAND_SUB = (
    "Take a photo. Get a lightweight custom part designed to fit your space, "
    "powered by TOPology optimization."
)
SCENE_PHOTO_HELP = (
    f"Minimum {MIN_IMAGES} photos. {RECOMMENDED_IMAGES}–{MAX_IMAGES} overlapping views work best."
)


def reconstruction_photo_status(count: int, min_images: int = MIN_IMAGES) -> Dict[str, Any]:
    """Capture-contract copy for the reconstruction uploader. 14–18 stays out of this."""
    n = max(0, int(count))
    needed = max(0, int(min_images) - n)
    noun = "photo" if needed == 1 else "photos"
    return {
        "count": n,
        "min_images": int(min_images),
        "ready": needed == 0,
        "needed": needed,
        "count_label": f"{n} photo" if n == 1 else f"{n} photos",
        "min_label": f"minimum {min_images}",
        "action": f"Add {needed} more {noun}" if needed else "Ready to reconstruct",
    }


def generator_status_message(configured: bool, warm_start: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Product-facing Grok status. Missing API is never treated as a validation failure."""
    if not configured:
        return API_NOT_CONNECTED
    if warm_start and warm_start.get("ok") is False:
        return GENERATION_FAILED_VALIDATION
    return None


def field_widget_kind(question: Any) -> str:
    """categorical | numeric | boolean | text. Prefers question metadata when present."""
    kind = str(getattr(question, "kind", None) or getattr(question, "type", None) or "").lower()
    if kind in {"categorical", "choice", "enum", "select"}:
        return "categorical"
    if kind in {"numeric", "number", "float", "int"}:
        return "numeric"
    if kind in {"bool", "boolean"}:
        return "boolean"
    if kind in {"text", "string"}:
        return "text"
    if _field_choices(question):
        return "categorical"
    field = getattr(question, "field", "") or ""
    unit = str(getattr(question, "unit", None) or "")
    if field in NUMERIC_FIELDS or unit in {"mm", "kg", "N", "Pa", "s"} or field.endswith(_NUMERIC_SUFFIXES):
        return "numeric"
    if field.startswith(_BOOL_PREFIXES):
        return "boolean"
    return "text"


def field_display_label(question: Any) -> str:
    field = getattr(question, "field", "") or ""
    if field in FIELD_COPY:
        return FIELD_COPY[field][0]
    text = (getattr(question, "question", None) or field.replace("_", " ")).strip()
    return text.split("(")[0].strip().rstrip("?") or field


def field_unit(question: Any) -> Optional[str]:
    unit = getattr(question, "unit", None)
    if unit:
        return str(unit)
    field = getattr(question, "field", "") or ""
    if field in FIELD_COPY:
        return FIELD_COPY[field][1]
    if field.endswith("_mm"):
        return "mm"
    if field.endswith("_kg"):
        return "kg"
    return None


def _field_choices(question: Any) -> List[Tuple[str, str]]:
    raw = (
        getattr(question, "options", None)
        or getattr(question, "choices", None)
        or getattr(question, "enum", None)
    )
    if raw:
        out: List[Tuple[str, str]] = []
        for item in raw:
            if isinstance(item, (tuple, list)) and item:
                value = str(item[0])
                label = str(item[1] if len(item) > 1 else item[0])
            else:
                value = str(item)
                label = value.replace("_", " ").strip().title()
            out.append((value, label))
        if out and not any(value == "other" or label.lower() == "other" for value, label in out):
            out.append(("other", "Other"))
        return out
    field = getattr(question, "field", "") or ""
    if field in CATEGORY_CHOICES:
        return list(CATEGORY_CHOICES[field])
    if field in SELECT_FIELDS:
        return [(opt, opt.replace("_", " ").strip().title()) for opt in SELECT_FIELDS[field]] + [
            ("other", "Other")
        ]
    return []


def mark_ui_transition(store: Dict[str, Any]) -> None:
    """Record that a user-triggered workflow transition just happened."""
    store["scroll_to_action"] = True


def consume_scroll_to_action(store: Dict[str, Any]) -> bool:
    """True once after a transition. Later reruns do not scroll."""
    return bool(store.pop("scroll_to_action", False))


def _mock_fixtures_enabled() -> bool:
    return bool(st.session_state.get("enable_mock_fixtures"))


def _topology_live() -> bool:
    if not _mock_fixtures_enabled():
        return True
    return st.session_state.get("mode_topo") == "Live"


def _topology_options():
    """Solver knobs from the sidebar (only when Topology = Live)."""
    from schemas import TopologySolverOptions

    if not _topology_live():
        return None
    return TopologySolverOptions(
        element_size_mm=float(st.session_state.get("topo_elem", 4.0)),
        max_iters=int(st.session_state.get("topo_iters", 40)),
        time_budget_s=float(st.session_state.get("topo_budget", 150)),
        device="auto",
    )


def _run_print(message: str) -> None:
    """Immediate terminal log for the Streamlit process. Does not change workflow state."""
    print(message, flush=True)


def _warm_start_mesh_path(candidate=None) -> str:
    cand = candidate
    if cand is None:
        orch = st.session_state.get("orch")
        cand = getattr(orch, "imported_candidate", None) if orch is not None else None
    if cand is None:
        cand = st.session_state.get("warm_start_candidate")
    path = getattr(cand, "mesh_path", None) if cand is not None else None
    if path:
        return str(path)
    ws = st.session_state.get("warm_start_result") or {}
    out_dir = ws.get("out_dir") or ""
    if out_dir:
        stl = Path(out_dir) / "grok_warmstart.stl"
        if stl.is_file():
            return str(stl)
        return str(out_dir)
    return ""


def _warm_start_is_valid(candidate=None) -> bool:
    ws = st.session_state.get("warm_start_result") or {}
    if ws.get("ok") is False:
        return False
    path = _warm_start_mesh_path(candidate)
    if path and Path(path).is_file() and Path(path).stat().st_size > 0:
        return True
    if candidate is not None and getattr(candidate, "mesh_path", None):
        return Path(candidate.mesh_path).is_file()
    return bool(ws.get("ok") and candidate is not None)


def _session_warm_start_candidate():
    cand = st.session_state.get("warm_start_candidate")
    if cand is not None and _warm_start_is_valid(cand):
        return cand
    orch = st.session_state.get("orch")
    existing = getattr(orch, "imported_candidate", None) if orch is not None else None
    if existing is not None and getattr(existing, "task", "") == "generated" and _warm_start_is_valid(existing):
        return existing
    return None


def _warm_start_generator():
    """Generator callable for the GEOMETRY stage when the geometry source is Grok-generated."""
    if effective_geometry_mode(st.session_state.get("mode_geom")) != GEOM_GENERATED:
        return None
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    provider = get_provider("grok")
    if not provider.configured:
        return None

    def _generate(requirements):
        force = bool(st.session_state.get("force_regenerate_design"))
        reused = _session_warm_start_candidate()
        current_fp = requirements_fingerprint(requirements)
        stored_fp = st.session_state.get("warm_start_fingerprint") or ""
        if should_reuse_warm_start(
            warm_start_ok=_warm_start_is_valid(reused),
            has_candidate=reused is not None,
            fingerprint=stored_fp,
            current_fingerprint=current_fp,
            force_regen=force,
        ):
            _run_print("[GROK] GENERATION skipped — valid warm-start mesh already exists")
            _run_print(f"[RUN] warm start path = {_warm_start_mesh_path(reused)}")
            return reused
        st.session_state.force_regenerate_design = False
        _run_print("[GROK] GENERATION_STARTED")
        st.session_state.warm_start_result = None
        st.session_state.failure_log_card = None
        orch = st.session_state.get("orch")
        if orch is not None:
            orch.warm_start_error = None
            if getattr(orch, "state", None) is not None:
                orch.state.notes = ""
                orch.state.feasibility = None
        try:
            result = generate_warm_start(
                requirements,
                provider,
                name="grok_warmstart",
                log=lambda msg: _run_print(f"[GROK] {msg}"),
            )
        except Exception:
            _run_print("[GROK] GENERATION_FAILED")
            _run_print("[RUN] EXCEPTION at GROK_GENERATION")
            traceback.print_exc()
            raise
        st.session_state.warm_start_result = {
            "ok": result.ok,
            "model": result.model,
            "attempts": result.attempts,
            "latency_s": round(result.latency_s, 1),
            "out_dir": result.out_dir,
            "script_path": result.script_path,
            "problems": result.problems,
            "error": result.error[-600:],
        }
        if not result.ok:
            _run_print("[GROK] GENERATION_FAILED")
            raise RuntimeError(result.error or "; ".join(result.problems) or "generation failed")
        st.session_state.warm_start_candidate = result.candidate
        st.session_state.warm_start_fingerprint = requirements_fingerprint(requirements)
        _run_print("[GROK] GENERATION_FINISHED")
        _run_print(f"[RUN] warm start path = {getattr(result.candidate, 'mesh_path', '')}")
        return result.candidate

    return _generate


def _candidate_name() -> str:
    name = st.session_state.get("candidate_name", "cupholder")
    return name if name in CANDIDATES else "cupholder"


def _registration_measurements() -> Optional[Dict[str, Any]]:
    """surfcap target.json -> mm measurements with confidence (None when no path is given)."""
    path = (st.session_state.get("reg_target_path") or "").strip()
    if not path:
        st.session_state.registration_meas = None
        return None
    cached = st.session_state.get("registration_meas")
    if cached and (
        cached.get("source_mesh") == str(Path(path).expanduser().resolve())
        or cached.get("target_json") == path
    ):
        return cached
    try:
        from to_agent.ingest.surfcap import measurements_from_path

        meas = measurements_from_path(path)
    except Exception as exc:  # noqa: BLE001 — unreadable file or to_agent missing
        meas = {"target_json": path, "error": f"{type(exc).__name__}: {exc}", "confidence": 0.0, "prefill": False}
    st.session_state.registration_meas = meas
    return meas


def _live_registration():
    """RegistrationOutput from a trusted surfcap measurement, else None (mock fixture stays)."""
    from schemas import RegistrationOutput

    meas = st.session_state.get("registration_meas")
    if not meas or meas.get("error") or not meas.get("confidence"):
        return None
    normal = tuple(float(v) for v in (meas.get("mount_normal") or (0.0, 0.0, 1.0)))
    return RegistrationOutput(
        is_mock=False,
        frame_id="desk_edge_frame",
        desk_plane_point_mm=(0.0, 0.0, float(meas.get("desk_thickness_mm") or 0.0)),
        desk_plane_normal=normal,
        confidence=float(meas["confidence"]),
        notes=(
            f"surfcap registration ({meas.get('thickness_source')}); desk thickness "
            f"{meas.get('desk_thickness_mm')} mm; {'; '.join(meas.get('notes', [])[:3])}"
        ),
    )


def _run_grounding_surface_registration() -> None:
    """Stage UI uploads (or use a local folder), run surfcap, and queue its target path."""
    st.session_state.registration_error = ""
    uploads = (
        st.session_state.get("reg_capture_photos")
        or st.session_state.get("scene_photo")
        or []
    )
    folder_text = (st.session_state.get("reg_capture_folder") or "").strip()
    try:
        if uploads:
            capture_dir = (
                REGISTRATION_OUT_ROOT
                / "captures"
                / (
                    f"capture_{time.strftime('%Y%m%d_%H%M%S')}_"
                    f"{time.time_ns() % 1_000_000:06d}"
                )
            )
            photos = [
                UploadedPhoto(name=item.name, data=item.getvalue())
                for item in uploads
            ]
            source, capture_warnings = stage_uploads(photos, capture_dir)
        elif folder_text:
            source = Path(folder_text).expanduser()
            capture_warnings = []
        else:
            raise ValueError("Upload a photo set or provide a local capture folder.")

        result = run_registration(
            source,
            table_prompt=st.session_state.get("reg_table_prompt") or "table",
        )
        result.warnings = list(dict.fromkeys(capture_warnings + result.warnings))
        st.session_state.registration_run = result
        if not result.ok:
            st.session_state.registration_error = result.message
            return
        st.session_state.pending_reg_target_path = result.target_json
        st.session_state.registration_meas = result.measurements
        _queue_workspace_tab(workspace_tab_after_event(reconstruction=True))
    except Exception as exc:  # noqa: BLE001 — show capture/tool failures in the UI
        st.session_state.registration_run = None
        st.session_state.registration_error = f"{type(exc).__name__}: {exc}"


def _load_table_a_registration() -> None:
    """Run the callable registration skill against the bundled table_a capture."""
    st.session_state.registration_error = ""
    result = run_sample_registration("table_a")
    st.session_state.registration_run = result
    if not result.ok:
        st.session_state.registration_error = result.message
        return
    measurement_path = result.artifacts.get("hybrid_mesh") or result.target_json
    st.session_state.pending_reg_target_path = measurement_path
    st.session_state.registration_meas = result.measurements
    _queue_workspace_tab(workspace_tab_after_event(reconstruction=True))


def _new_orchestrator() -> Orchestrator:
    mode = effective_geometry_mode(st.session_state.get("mode_geom"))
    return Orchestrator(
        fixtures=fixtures_for_geometry_mode(mode, topology_live=_topology_live()),
        imported_candidate=imported_candidate_for_mode(mode, _candidate_name()),
        topology_options=_topology_options(),
        warm_start_generator=_warm_start_generator(),
    )


HOOK_CASE_MESSAGE = "I want a hook clamped under my desk edge to hang a 5 kg bag about 100 mm out from the edge."
SHELF_CASE_MESSAGE = "I want a small shelf that lifts my stapler 100 mm above the desk with a flat top."
SMALL_BOTTLE_CASE_MESSAGE = "Design a desk-clamped holder for a 0.5 kg, 66 mm diameter water bottle."
SMALL_BOTTLE_CASE_ANSWERS = {
    "filled_bottle_mass_kg": 0.5,
    "bottle_diameter_mm": 66.0,
    "bottle_height_mm": 130.0,
    "desk_thickness_mm": 20.0,
    "attachment_method": "clamp",
    "allowed_contact_region": "desk_front_edge",
    "attachment_notes": "clamp only, no drilling",
    "max_protrusion_mm": 100.0,
    "manufacturing_method": "3d_print",
    "material": "PLA",
    "max_part_mass_kg": 0.4,
}
HOOK_CASE_ANSWERS = {
    "filled_bottle_mass_kg": 5.0,
    "bottle_diameter_mm": 30.0,
    "bottle_height_mm": 300.0,
    "desk_thickness_mm": 20.0,
    "attachment_method": "clamp",
    "allowed_contact_region": "desk_front_edge",
    "attachment_notes": "clamp only, no drilling",
    "max_protrusion_mm": 110.0,
    "manufacturing_method": "3d_print",
    "material": "PLA",
    "max_part_mass_kg": 0.3,
}
SHELF_CASE_ANSWERS = {
    "filled_bottle_mass_kg": 0.5,
    "bottle_diameter_mm": 60.0,
    "bottle_height_mm": 40.0,
    "desk_thickness_mm": 20.0,
    "attachment_method": "free_standing",
    "allowed_contact_region": "desk_top",
    "attachment_notes": "stands on the desk, no fasteners",
    "max_protrusion_mm": 120.0,
    "manufacturing_method": "3d_print",
    "material": "PLA",
    "max_part_mass_kg": 0.4,
}


def apply_pending_request_prefill(store: Dict[str, Any]) -> None:
    """Copy pending prefill into the widget key before the widget is created.

    `request_text` is owned by the text_area. Call this only at the start of a
    script run, before `st.text_area(..., key="request_text")`.
    """
    pending = store.get("pending_request_prefill")
    if pending is not None:
        store["request_text"] = pending
        store["pending_request_prefill"] = None


def apply_pending_registration_path(store: Dict[str, Any]) -> None:
    """Move a completed run's target path into the text widget before construction."""
    pending = store.get("pending_reg_target_path")
    if pending is not None:
        store["reg_target_path"] = pending
        store["pending_reg_target_path"] = None


def apply_trusted_registration_answers(store: Dict[str, Any]) -> None:
    """Trusted surfcap measurements can satisfy a clarification field without showing it."""
    meas = store.get("registration_meas") or {}
    if not meas.get("prefill") or meas.get("desk_thickness_mm") is None:
        return
    from agents.interaction import classify_design_task, clarification_specs_for_task

    orch = store.get("orch")
    req = getattr(getattr(orch, "state", None), "requirements", None)
    task = getattr(req, "task_kind", None) or classify_design_task(
        store.get("submitted_request") or store.get("request_text") or ""
    )
    asked = {spec["field"] for spec in clarification_specs_for_task(task)}
    if "desk_thickness_mm" not in asked:
        return
    answers = store.setdefault("answers", {})
    if answers.get("desk_thickness_mm") in (None, "", 0, 0.0):
        answers["desk_thickness_mm"] = float(meas["desk_thickness_mm"])


def apply_pending_workspace_tab(store: Dict[str, Any]) -> None:
    """Move an auto-selected workspace tab into the widget key before construction."""
    pending = store.get("pending_workspace_tab")
    if pending in WORKSPACE_TABS:
        store["workspace_tab"] = pending
    store["pending_workspace_tab"] = None


def _queue_workspace_tab(tab: str) -> None:
    st.session_state.pending_workspace_tab = tab
    st.session_state.workspace_tab = tab


REASONING_CHOICES = ["Mock", "K2 Horizon", "Grok", "Cursor"]


def _default_reasoning_choice() -> str:
    """Grok is the product default. Mock is developer-only and is never pre-selected."""
    return "Grok"


def _geometry_mode_label(mode: str) -> str:
    if mode == GEOM_GOLDEN:
        return "Golden Fixture  —  regression / integration test"
    if mode == GEOM_LIVE:
        return "Live Geometry  —  not connected"
    if mode == GEOM_IMPORTED:
        return "Imported Candidate Geometry"
    if mode == GEOM_GENERATED:
        return "Generated Warm Start (Grok)"
    return "Adaptive Synthetic Mock"


def _default_geometry_mode() -> str:
    """Product default: Grok warm-start generation. Mock / adaptive is developer-only."""
    return GEOM_GENERATED


def _init_session() -> None:
    if "enable_mock_fixtures" not in st.session_state:
        st.session_state.enable_mock_fixtures = False
    if st.session_state.get("mode_geom") not in GEOM_MODE_OPTIONS:
        st.session_state.mode_geom = _default_geometry_mode()
    if not st.session_state.enable_mock_fixtures:
        st.session_state.mode_topo = "Live"
    elif st.session_state.get("mode_topo") not in ("Mock Fixture", "Live"):
        st.session_state.mode_topo = "Live"
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
    if "pending_reg_target_path" not in st.session_state:
        st.session_state.pending_reg_target_path = None
    if "registration_run" not in st.session_state:
        st.session_state.registration_run = None
    if "registration_error" not in st.session_state:
        st.session_state.registration_error = ""
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
    if "pending_registration" not in st.session_state:
        st.session_state.pending_registration = False
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = EXAMPLE_CHOICES[0]
    if "request_text" not in st.session_state and not st.session_state.pending_request_prefill:
        st.session_state.request_text = HAPPY_PATH_MESSAGE
    if "scroll_to_action" not in st.session_state:
        st.session_state.scroll_to_action = False
    if "workspace_tab" not in st.session_state:
        st.session_state.workspace_tab = WORKSPACE_TAB_SCENE
    if "pending_workspace_tab" not in st.session_state:
        st.session_state.pending_workspace_tab = None
    if "focus_constraint_field" not in st.session_state:
        st.session_state.focus_constraint_field = None
    if "warm_start_result" not in st.session_state:
        st.session_state.warm_start_result = None
    if "warm_start_candidate" not in st.session_state:
        st.session_state.warm_start_candidate = None
    if "warm_start_fingerprint" not in st.session_state:
        st.session_state.warm_start_fingerprint = None
    if "force_regenerate_design" not in st.session_state:
        st.session_state.force_regenerate_design = False
    if "progress_kind" not in st.session_state:
        st.session_state.progress_kind = None


def _apply_pending_prefills() -> None:
    apply_pending_request_prefill(st.session_state)
    apply_pending_registration_path(st.session_state)
    apply_trusted_registration_answers(st.session_state)
    apply_pending_workspace_tab(st.session_state)


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
    clear_reasoning_traces()
    st.session_state.orch = _new_orchestrator()
    st.session_state.chat = []
    st.session_state.answers = {}
    st.session_state.compare = None
    st.session_state.cursor_compare = None
    st.session_state.scene_observation = None
    st.session_state.scene_observation_meta = {}
    st.session_state.ui_notice = ""
    st.session_state.last_chat_error_key = None
    st.session_state.warm_start_result = None
    st.session_state.warm_start_candidate = None
    st.session_state.warm_start_fingerprint = None
    st.session_state.force_regenerate_design = False
    st.session_state.progress_kind = None
    st.session_state.focus_constraint_field = None
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
    _queue_workspace_tab(WORKSPACE_TAB_SCENE)


def _append(role: str, text: str, error_key: Optional[tuple] = None) -> None:
    if error_key is not None:
        if st.session_state.get("last_chat_error_key") == error_key:
            return
        st.session_state.last_chat_error_key = error_key
    st.session_state.chat.append({"role": role, "text": text})


def _current_failure():
    orch = st.session_state.get("orch")
    state = getattr(orch, "state", None)
    if state is None:
        return None
    return detect_ui_failure(
        state,
        warm_start=st.session_state.get("warm_start_result"),
        registration_error=st.session_state.get("registration_error") or "",
        registration_run=st.session_state.get("registration_run"),
        warm_start_error=getattr(orch, "warm_start_error", "") or "",
    )


def _assistant_after_interaction(state: DesignState) -> str:
    failure = _current_failure()
    if failure is not None:
        return failure.headline
    if state.stage == WorkflowStage.REJECTED:
        return f"Out of scope. {state.reject_reason or 'This request cannot continue.'}"
    if state.stage == WorkflowStage.REQUEST_INFORMATION:
        if state.feasibility is not None and not state.feasibility.feasible:
            return state.feasibility.message or "Requirements conflict. Revise the values below."
        if state.candidate_fit is not None and not state.candidate_fit.fits:
            return state.candidate_fit.message or "This design does not fit. Revise the values below."
        missing = missing_detail_count(state, st.session_state.get("answers"))
        return need_details_title(missing)
    return "Ready to optimize."


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


def _refresh_clarifications_from_backend(orch: Orchestrator) -> None:
    """Keep the widgets in sync with InteractionAgent._missing after a bounce."""
    req = orch.state.requirements
    if req is None:
        return
    result = orch.interaction._decide(req)
    orch.state.clarifications = result.questions
    orch.state.missing_information = [
        MissingInformation(field=q.field, reason=q.question, priority=q.priority)
        for q in result.questions
    ]


def _preserve_supplied_answers(state: DesignState) -> None:
    """Keep already-entered answers after the backend asks for remaining fields."""
    answers = dict(st.session_state.get("answers") or {})
    req = state.requirements
    if req is None:
        st.session_state.answers = answers
        return
    extras = dict(req.task_answers or {})
    mapped = {
        "filled_bottle_mass_kg": req.payload.filled_mass_kg,
        "supported_load_kg": extras.get("supported_load_kg") or req.payload.filled_mass_kg,
        "bottle_diameter_mm": req.object_geometry.bottle_diameter_mm,
        "payload_size_mm": extras.get("payload_size_mm") or req.object_geometry.bottle_diameter_mm,
        "desk_thickness_mm": req.environment.desk_thickness_mm,
        "attachment_method": req.attachment.method,
        "allowed_contact_region": req.attachment.allowed_contact_region,
        "attachment_structure": extras.get("attachment_structure"),
        "handle_location": extras.get("handle_location"),
        "mounting_region": extras.get("mounting_region") or req.attachment.allowed_contact_region,
        "drilling_allowed": extras.get("drilling_allowed"),
        "max_protrusion_mm": req.design_envelope.max_protrusion_mm,
        "required_reach_mm": extras.get("required_reach_mm"),
        "manufacturing_method": req.manufacturing.method,
        "wall_clearance_mm": extras.get("wall_clearance_mm") or req.design_envelope.max_width_mm,
    }
    for key, value in {**mapped, **extras}.items():
        if value in (None, "") or answers.get(key) not in (None, ""):
            continue
        answers[key] = value
    st.session_state.answers = answers


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


def _load_live_case(candidate: str, message: str, answers: Dict[str, Any]) -> None:
    """Preset: imported candidate + live topology, request and answers prefilled."""
    answers = dict(answers)
    registration = st.session_state.get("registration_meas") or {}
    if registration.get("prefill") and registration.get("desk_thickness_mm") is not None:
        answers["desk_thickness_mm"] = float(registration["desk_thickness_mm"])
    st.session_state.mode_geom = GEOM_IMPORTED
    st.session_state.candidate_name = candidate
    st.session_state.mode_topo = "Live"
    _queue_request_text(message)
    _reset(ingest_message=message, prefill_happy=False)
    st.session_state.answers = answers


def _on_bottle_case() -> None:
    answers = dict(SMALL_BOTTLE_CASE_ANSWERS)
    registration = st.session_state.get("registration_meas") or {}
    if registration.get("prefill") and registration.get("desk_thickness_mm") is not None:
        answers["desk_thickness_mm"] = float(registration["desk_thickness_mm"])
    st.session_state.mode_geom = GEOM_ADAPTIVE
    st.session_state.mode_topo = "Live"
    _queue_request_text(SMALL_BOTTLE_CASE_MESSAGE)
    _reset(ingest_message=SMALL_BOTTLE_CASE_MESSAGE, prefill_happy=False)
    st.session_state.answers = answers


def _on_hook_case() -> None:
    _load_live_case("desk_bag_hook", HOOK_CASE_MESSAGE, HOOK_CASE_ANSWERS)


def _on_shelf_case() -> None:
    _load_live_case("stapler_shelf", SHELF_CASE_MESSAGE, SHELF_CASE_ANSWERS)


def _on_rejected() -> None:
    _queue_request_text(REJECTED_MESSAGE)
    _reset(ingest_message=REJECTED_MESSAGE, prefill_happy=False)


def _on_reset_session() -> None:
    st.session_state.mode_geom = _default_geometry_mode()
    st.session_state.mode_reason = _default_reasoning_choice()
    st.session_state.enable_mock_fixtures = False
    st.session_state.mode_topo = "Live"
    _queue_request_text(HAPPY_PATH_MESSAGE)
    _reset(clear_image=True)
    mark_ui_transition(st.session_state)


EXAMPLE_CHOICES = (
    "Happy Path",
    "Missing information",
    "Rejected",
    "Desk hook (live TO)",
    "Stapler shelf (live TO)",
    "Small bottle holder (live TO)",
)
_EXAMPLE_LOADERS = {
    "Happy Path": _on_happy_path,
    "Missing information": _on_missing_info,
    "Rejected": _on_rejected,
    "Desk hook (live TO)": _on_hook_case,
    "Stapler shelf (live TO)": _on_shelf_case,
    "Small bottle holder (live TO)": _on_bottle_case,
}


def _on_load_example() -> None:
    loader = _EXAMPLE_LOADERS.get(st.session_state.get("example_choice") or "Happy Path")
    if loader is not None:
        loader()
        mark_ui_transition(st.session_state)


def _on_primary_action() -> None:
    """One entry point for the current-action button. Same backend calls as before."""
    sync_answer_widgets(st.session_state)
    spec = action_spec(
        st.session_state.orch.state,
        st.session_state.get("answers"),
        failure=_current_failure(),
    )
    from_optimize = spec.kind == "optimize" or spec.button == "Run optimization"
    if from_optimize:
        _run_print("[RUN] RUN_OPT_CLICKED")
        _run_print(f"[RUN] current geometry source = {st.session_state.get('mode_geom')}")
        reused = _session_warm_start_candidate()
        _run_print(f"[RUN] warm start exists = {bool(reused is not None or _warm_start_is_valid())}")
        _run_print(f"[RUN] warm start path = {_warm_start_mesh_path(reused)}")
        _run_print(
            f"[RUN] stage={getattr(st.session_state.orch.state.stage, 'value', st.session_state.orch.state.stage)} "
            f"spec.kind={spec.kind} button={spec.button!r}"
        )
    if spec.kind == "start":
        if from_optimize:
            _run_print("[RUN] BLOCKED: action_spec rerouted to start")
        _on_submit_request()
    elif spec.kind in ("continue", "optimize"):
        _continue_design(from_optimize=from_optimize)
    elif spec.kind == "retry":
        if from_optimize:
            _run_print("[RUN] BLOCKED: action_spec rerouted to retry")
        _retry_same_constraints()
    elif spec.kind == "revise":
        if from_optimize:
            _run_print("[RUN] BLOCKED: action_spec rerouted to revise")
        _reset()
        mark_ui_transition(st.session_state)
    else:
        if from_optimize:
            _run_print(f"[RUN] BLOCKED: unhandled action kind={spec.kind}")
    if from_optimize:
        _run_print("[RUN] RERUN_REQUESTED")


def _retry_same_constraints() -> None:
    """Dispatch Retry optimization vs regenerate from the current failure stage."""
    sync_answer_widgets(st.session_state)
    failure = _current_failure()
    orch = st.session_state.get("orch")
    req = getattr(getattr(orch, "state", None), "requirements", None)
    stored_fp = st.session_state.get("warm_start_fingerprint") or ""
    current_fp = requirements_fingerprint(req)
    action = retry_action(
        failure_stage=getattr(failure, "stage", "") or "",
        has_valid_warm_start=_session_warm_start_candidate() is not None,
        requirements_changed=bool(stored_fp and current_fp and stored_fp != current_fp),
        force_regenerate=False,
    )
    if action == RETRY_TOPOLOGY:
        _retry_topology_only()
    else:
        _regenerate_design()


def _retry_topology_only() -> None:
    """Reuse the validated warm-start and rebuild topology only."""
    _run_print("[RUN] RETRY_OPTIMIZATION")
    sync_answer_widgets(st.session_state)
    orch: Orchestrator = st.session_state.orch
    reused = _session_warm_start_candidate()
    if reused is not None:
        orch.imported_candidate = reused
        if getattr(orch, "state", None) is not None:
            orch.state.imported_candidate = reused
    path = _warm_start_mesh_path(reused)
    _run_print(f"[RUN] reusing warm start path = {path}")
    mark_ui_transition(st.session_state)
    st.session_state.progress_kind = RETRY_TOPOLOGY
    if _topology_live() or orch.warm_start_generator is not None:
        _run_with_progress(orch, progress_kind=RETRY_TOPOLOGY, runner=orch.retry_topology)
    else:
        orch.retry_topology()
    _append("assistant", _user_result_message(orch.state))
    if orch.state.topology is not None or orch.state.geometry is not None or orch.state.structure is not None:
        _queue_workspace_tab(
            workspace_tab_after_event(
                design=orch.state.geometry is not None or orch.state.structure is not None,
                topology=orch.state.topology is not None,
            )
        )


def _regenerate_design() -> None:
    """Clear the candidate and call the geometry generator again."""
    _run_print("[RUN] REGENERATE_DESIGN")
    sync_answer_widgets(st.session_state)
    answers = dict(st.session_state.get("answers") or {})
    message = st.session_state.get("submitted_request") or _current_request_text()
    st.session_state.force_regenerate_design = True
    st.session_state.warm_start_result = None
    st.session_state.warm_start_candidate = None
    st.session_state.warm_start_fingerprint = None
    st.session_state.orch = _new_orchestrator()
    st.session_state.answers = answers
    mark_ui_transition(st.session_state)
    st.session_state.progress_kind = REGENERATE_DESIGN
    _sync_orch_fixtures()
    if not message:
        st.session_state.ui_notice = "Write a design request first."
        return
    orch: Orchestrator = st.session_state.orch
    orch.ingest_user_request(message)
    orch.apply_answers(_answers_to_update())
    st.session_state.answers = answers
    if orch.state.stage not in (
        WorkflowStage.COMPLETE,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUEST_INFORMATION,
        WorkflowStage.DESIGN_REVIEW_FAILED,
    ):
        if _topology_live() or orch.warm_start_generator is not None:
            _run_with_progress(orch, progress_kind=REGENERATE_DESIGN)
        else:
            orch.run()
    _append("assistant", _user_result_message(orch.state))
    if orch.state.topology is not None or orch.state.geometry is not None or orch.state.structure is not None:
        _queue_workspace_tab(
            workspace_tab_after_event(
                design=orch.state.geometry is not None or orch.state.structure is not None,
                topology=orch.state.topology is not None,
            )
        )


def _focus_constraint_field(field: str) -> None:
    st.session_state.focus_constraint_field = field
    mark_ui_transition(st.session_state)


@st.dialog("Full log", width="large")
def _failure_log_dialog() -> None:
    card = st.session_state.get("failure_log_card")
    if card is None:
        st.write("No log available.")
        return
    st.caption(card.stage.replace("_", " "))
    st.code(card.raw_log or "(empty)", language="text")
    if st.button("Close"):
        st.rerun()


def _on_submit_request() -> None:
    message = _current_request_text()
    st.session_state.submitted_request = message
    mark_ui_transition(st.session_state)
    if not message:
        st.session_state.ui_notice = "Write a design request first."
        return
    orch: Orchestrator = st.session_state.orch
    if orch.state.stage != WorkflowStage.REQUIREMENTS or orch.state.requirements is not None:
        _reset()
    _ingest(message)


def _stored_choice_value(field: str, value: Any) -> Any:
    """Map a pill label back to the backend stored value when needed."""
    if value in (None, ""):
        return value
    text = str(value)
    for stored, label in CATEGORY_CHOICES.get(field, ()):
        if text in (stored, label):
            return stored
    for stored in SELECT_FIELDS.get(field, ()):
        if text == stored or text == stored.replace("_", " ").strip().title():
            return stored
    return value


def sync_answer_widgets(store: Dict[str, Any]) -> None:
    """Copy current clarification widget values before an on-click callback consumes them."""
    answers = store.setdefault("answers", {})
    for key in list(store.keys()):
        if not key.startswith("ans_") or key.endswith("_other"):
            continue
        field = key.removeprefix("ans_")
        raw = store[key]
        if raw in (None, ""):
            continue
        mapped = _stored_choice_value(field, raw)
        if mapped == "other":
            custom = store.get(f"ans_{field}_other")
            mapped = (custom or "").strip() or mapped
        answers[field] = mapped
    if store.get("no_drill"):
        answers["attachment_notes"] = "clamp only, no drilling"


def _on_continue_design() -> None:
    sync_answer_widgets(st.session_state)
    _continue_design()


def _sync_orch_fixtures() -> None:
    orch: Orchestrator = st.session_state.orch
    if orch.state.geometry is not None:
        _run_print("[RUN] _sync_orch_fixtures skipped — geometry already present")
        return
    mode = st.session_state.get("mode_geom")
    existing = orch.imported_candidate
    orch.fixtures = fixtures_for_geometry_mode(mode, topology_live=_topology_live())
    live_reg = _live_registration()
    if live_reg is not None:
        orch.fixtures.registration = live_reg
    imported = imported_candidate_for_mode(mode, _candidate_name())
    if imported is not None:
        orch.imported_candidate = imported
    else:
        reused = existing if _warm_start_is_valid(existing) else _session_warm_start_candidate()
        orch.imported_candidate = reused
        if reused is not None:
            _run_print("[RUN] reusing existing warm-start candidate; Grok will not regenerate")
    orch.topology_options = _topology_options()
    orch.warm_start_generator = _warm_start_generator()


def _continue_design(*, from_optimize: bool = False) -> None:
    orch: Orchestrator = st.session_state.orch
    mark_ui_transition(st.session_state)
    _sync_orch_fixtures()
    if orch.state.stage == WorkflowStage.REQUIREMENTS and orch.state.requirements is None:
        message = _current_request_text()
        if not message:
            if from_optimize:
                _run_print("[RUN] BLOCKED: no design request text")
            st.session_state.ui_notice = "Write a design request first."
            return
        st.session_state.submitted_request = message
        _ingest(message)
    if orch.state.stage == WorkflowStage.REQUEST_INFORMATION:
        update = _answers_to_update()
        _append("user", _format_answers(update))
        orch.apply_answers(update)
        if orch.state.stage == WorkflowStage.REQUEST_INFORMATION:
            _refresh_clarifications_from_backend(orch)
            _preserve_supplied_answers(orch.state)
        if not from_optimize or orch.state.stage == WorkflowStage.REQUEST_INFORMATION:
            _append("assistant", _assistant_after_interaction(orch.state))
        if from_optimize:
            _run_print(
                f"[RUN] apply_answers -> stage={getattr(orch.state.stage, 'value', orch.state.stage)}"
            )
    if orch.state.contract_error:
        if from_optimize:
            _run_print(f"[RUN] BLOCKED: contract_error {orch.state.contract_error}")
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
        if from_optimize:
            _run_print(
                f"[RUN] starting orch.run() from stage={getattr(orch.state.stage, 'value', orch.state.stage)}"
            )
        if _topology_live() or orch.warm_start_generator is not None:
            _run_with_progress(orch)
        else:
            orch.run()
        if from_optimize:
            _run_print(
                f"[RUN] RESULT_STORED stage={getattr(orch.state.stage, 'value', orch.state.stage)} "
                f"topology={orch.state.topology is not None} "
                f"geometry={orch.state.geometry is not None}"
            )
        warm_failed = bool(
            getattr(orch, "warm_start_error", None)
            or (st.session_state.get("warm_start_result") or {}).get("ok") is False
        )
        if warm_failed:
            if from_optimize:
                _run_print("[RUN] failure stage = warm_start_generation_failed")
            _preserve_supplied_answers(orch.state)
        elif orch.state.stage == WorkflowStage.REQUEST_INFORMATION:
            _refresh_clarifications_from_backend(orch)
            _preserve_supplied_answers(orch.state)
            missing = backend_missing_fields(orch.state, st.session_state.get("answers"))
            if from_optimize:
                _run_print(f"[RUN] bounced to request_information missing={missing}")
        if orch.state.contract_error:
            _append(
                "assistant",
                format_mismatch_message(orch.state),
                error_key=chat_error_key(orch.state),
            )
        else:
            _append("assistant", _user_result_message(orch.state))
    else:
        if from_optimize:
            _run_print(
                f"[RUN] BLOCKED: orch.run() skipped because stage="
                f"{getattr(orch.state.stage, 'value', orch.state.stage)}"
            )
    if orch.state.topology is not None or orch.state.geometry is not None or orch.state.structure is not None:
        _queue_workspace_tab(
            workspace_tab_after_event(
                design=orch.state.geometry is not None or orch.state.structure is not None,
                topology=orch.state.topology is not None,
            )
        )


def _render_reasoning_traces() -> None:
    """Show what the reasoning model was asked, what it returned, and why it was rejected."""
    traces = reasoning_traces()
    if not traces:
        return
    failed = [t for t in traces if t.outcome is not ReasoningOutcome.LIVE]
    blocked = [t for t in traces if t.outcome is ReasoningOutcome.BLOCKED]
    st.markdown("**Reasoning**")
    if blocked:
        st.error(
            f"{len(blocked)} reasoning call(s) BLOCKED in developer mode. The workflow stopped "
            "rather than falling back to the deterministic tables."
        )
    elif failed:
        st.warning(
            f"{len(failed)} reasoning call(s) fell back to the deterministic tables — those "
            "decisions are overfit to the demo problems, not reasoned."
        )
    for t in traces:
        icon = {"live": "✅", "fallback": "⚠️", "blocked": "⛔"}[t.outcome.value]
        with st.expander(f"{icon} {t.summary()}", expanded=bool(blocked) and t in blocked):
            st.write({
                "task": t.task.value, "mode": t.mode.value, "provider": t.provider,
                "model": t.model, "outcome": t.outcome.value,
                "latency_s": round(t.total_latency_s, 1), "attempts": len(t.attempts),
            })
            if t.reason:
                st.caption(t.reason)
            for a in t.attempts:
                label = "accepted" if a.ok else f"rejected — {a.failure_kind.value if a.failure_kind else 'unknown'}"
                st.markdown(f"*attempt {a.attempt}: {label} ({a.latency_s:.1f}s)*")
                if a.error:
                    st.code(a.error, language="text")
                if a.raw_excerpt and not a.ok:
                    st.code(a.raw_excerpt, language="json")
            if t.prompt_excerpt:
                with st.expander("prompt sent", expanded=False):
                    st.code(t.prompt_excerpt, language="text")


def _run_with_progress(orch: Orchestrator, *, progress_kind: str | None = None, runner=None) -> None:
    """Run the live stages with a progress bar and a wall-time expectation."""
    status = st.empty()
    bar = st.progress(0.0)
    started = time.monotonic()
    state = {"total": None}
    kind = progress_kind or st.session_state.get("progress_kind")
    if not kind:
        reused = orch.imported_candidate is not None and _warm_start_is_valid(orch.imported_candidate)
        if reused:
            kind = RETRY_TOPOLOGY if st.session_state.get("progress_kind") == RETRY_TOPOLOGY else "optimize"
        elif orch.warm_start_generator is not None:
            kind = "generate"
        else:
            kind = "optimize"
    caption = progress_caption(kind)

    def on_progress(it: int, total: int, compliance: float) -> None:
        state["total"] = total
        elapsed = time.monotonic() - started
        remaining = (elapsed / it) * (total - it) if it else 0.0
        bar.progress(min(it / max(total, 1), 1.0))
        status.caption(
            f"Optimizing — iteration {it}/{total} · compliance {compliance:.4g} · "
            f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
        )

    orch.topology_progress = on_progress
    status.caption(caption)
    try:
        with st.spinner(caption):
            (runner or orch.run)()
    except Exception:
        _run_print("[RUN] EXCEPTION at _run_with_progress/orch.run")
        traceback.print_exc()
        raise
    finally:
        orch.topology_progress = None
        st.session_state.progress_kind = None
        bar.empty()
        status.empty()


def _format_answers(update: RequirementsUpdate) -> str:
    data = {k: v for k, v in update.model_dump().items() if v is not None}
    if not data:
        return "Details sent."
    bits = []
    for key, value in data.items():
        label = FIELD_COPY[key][0] if key in FIELD_COPY else key.replace("_", " ")
        bits.append(f"{label} {value}")
    return " · ".join(bits)


def _user_result_message(state: DesignState) -> str:
    failure = _current_failure()
    if failure is not None:
        return failure.headline
    if state.stage == WorkflowStage.REJECTED:
        return _assistant_after_interaction(state)
    if state.stage == WorkflowStage.REQUEST_INFORMATION:
        return _assistant_after_interaction(state)
    if state.stage == WorkflowStage.DESIGN_REVIEW_FAILED:
        return "Design review did not pass."
    if state.stage == WorkflowStage.TOPOLOGY_FAILED:
        return "Optimization did not finish."
    if state.stage == WorkflowStage.VERIFICATION_FAILED:
        return "Verification did not pass."
    if state.topology is not None:
        accept = state.topology.acceptance or {}
        if accept.get("acceptance_status") == "unresolved_not_converged" or state.topology.converged is False:
            return "Optimization preview."
        if state.stage == WorkflowStage.COMPLETE:
            return "Design ready."
        return "Optimization finished."
    if state.stage == WorkflowStage.COMPLETE:
        return "Design ready."
    return "Ready to optimize."


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
        if t.is_mock:
            lines.append(f"Topology: MOCK — volume fraction {t.volume_fraction}. {t.notes}".rstrip())
        else:
            lines.append(
                f"Topology: LIVE — compliance {t.compliance:.4g}, volume fraction {t.volume_fraction:.3f}, "
                f"mass change vs warm start {-t.mass_reduction_pct:+.0f}%, {t.iterations} iterations in "
                f"{t.wall_time_s:.0f} s ({t.solver_status}); STL: {t.optimized_geometry_ref}"
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
    set_reasoning_mode(
        ReasoningMode.DEVELOPER if st.session_state.get("developer_mode") else ReasoningMode.PRODUCT
    )


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
    mesh_cols = st.columns(4)
    mesh_cols[0].metric("Vertices", candidate.vertex_count or "n/a")
    mesh_cols[1].metric("Faces", candidate.face_count or "n/a")
    mesh_cols[2].metric("Watertight", "yes" if candidate.watertight else "no")
    mesh_cols[3].metric("Components", candidate.connected_components or "n/a")
    st.markdown("**Key dimensions**")
    dims = {
        "task": candidate.task,
        "desk compatibility (mm)": (
            [candidate.compatible_desk_min_mm, candidate.compatible_desk_max_mm]
            if candidate.compatible_desk_min_mm is not None
            else None
        ),
    }
    if candidate.inner_diameter_mm is not None:
        dims["inner diameter (mm)"] = candidate.inner_diameter_mm
        dims["outer diameter (mm)"] = candidate.outer_diameter_mm
        dims["holder height (mm)"] = candidate.holder_height_mm
    if candidate.dimensions:
        for key in (
            "hook_opening",
            "desk_face_to_hook_centerline",
            "nominal_concept_target_load",
            "clamp_internal_gap",
            "platform_length",
            "platform_width",
            "lift_height",
        ):
            if key in candidate.dimensions:
                dims[key] = candidate.dimensions[key]
    if candidate.clamp_reach_mm is not None:
        dims["reach / protrusion (mm)"] = candidate.clamp_reach_mm
    st.write(dims)
    st.markdown("**Fit against requirements**")
    if fit is None:
        return
        return

    def _status(check) -> str:
        if check.status.value == "n/a":
            return "N/A"
        return check.status.value.upper()

    st.write({check.name.replace("_", " "): _status(check) for check in fit.checks})
    st.write({"overall": "PASS" if fit.fits else "FAIL"})


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
    _append("assistant", _assistant_after_interaction(orch.state))
    mark_ui_transition(st.session_state)


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


def _stl_path(state: DesignState) -> Optional[Path]:
    for raw in (
        getattr(state.cad, "filename", None) if state.cad is not None else None,
        getattr(state.topology, "optimized_geometry_ref", None) if state.topology is not None else None,
    ):
        if not raw or str(raw).startswith("mock://"):
            continue
        path = Path(raw)
        if path.suffix.lower() in {".stl", ".obj"} and path.exists():
            return path
    return None


def _choice_label(choices: Sequence[Tuple[str, str]], value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value)
    for stored, label in choices:
        if stored == text or label == text:
            return label
    return "Other"


def _render_categorical_field(question: Any, widget_key: str, current_val: Any) -> Any:
    choices = _field_choices(question)
    labels = [label for _value, label in choices]
    stored_by_label = {label: value for value, label in choices}
    existing = st.session_state.get(widget_key, current_val)
    mapped = _choice_label(choices, existing)
    if mapped is not None:
        st.session_state[widget_key] = mapped
        if mapped == "Other" and existing not in labels and existing not in (None, "", "other"):
            st.session_state.setdefault(f"{widget_key}_other", str(existing))
    elif widget_key in st.session_state and st.session_state[widget_key] not in labels:
        del st.session_state[widget_key]
    label = field_display_label(question)
    picked = st.pills(label, labels, key=widget_key, help=question.question)
    if picked == "Other":
        custom = st.text_input("Describe", key=f"{widget_key}_other")
        return (custom or "").strip()
    if picked:
        return stored_by_label.get(picked, picked)
    if current_val not in (None, ""):
        return current_val
    return ""


def _render_clarification_fields(state: DesignState) -> None:
    """Only the backend's current missing fields. No invented measurement list."""
    if state.stage != WorkflowStage.REQUEST_INFORMATION:
        return
    questions = list(state.clarifications or [])
    if not questions:
        orch = st.session_state.get("orch")
        if orch is not None:
            _refresh_clarifications_from_backend(orch)
            questions = list(orch.state.clarifications or [])
    if not questions:
        return
    reg_meas = st.session_state.get("registration_meas") or {}
    for question in questions:
        field = question.field
        if (
            field == "desk_thickness_mm"
            and reg_meas.get("prefill")
            and reg_meas.get("desk_thickness_mm") is not None
        ):
            st.session_state.answers[field] = float(reg_meas["desk_thickness_mm"])
            continue
        widget_key = f"ans_{field}"
        current_val = st.session_state.answers.get(field)
        if (
            field == "desk_thickness_mm"
            and current_val in (None, "", 0.0)
            and reg_meas.get("prefill")
            and widget_key not in st.session_state
        ):
            current_val = reg_meas["desk_thickness_mm"]
        kind = field_widget_kind(question)
        label = field_display_label(question)
        unit = field_unit(question)
        help_txt = question.question
        if field == "desk_thickness_mm" and reg_meas.get("desk_thickness_mm") is not None:
            help_txt = (
                f"{question.question} Registration: {reg_meas['desk_thickness_mm']} mm "
                f"(confidence {reg_meas.get('confidence', 0):.2f})."
            )
        if kind == "numeric":
            if widget_key not in st.session_state:
                st.session_state[widget_key] = (
                    float(current_val) if current_val not in (None, "") else 0.0
                )
            cols = st.columns([4, 1]) if unit else [st.container()]
            with cols[0]:
                st.number_input(label, key=widget_key, help=help_txt)
            if unit:
                with cols[1]:
                    st.markdown(f'<div class="mdc-unit">{html.escape(unit)}</div>', unsafe_allow_html=True)
            st.session_state.answers[field] = st.session_state[widget_key]
            if field == "desk_thickness_mm" and reg_meas.get("prefill"):
                reg_val = float(reg_meas["desk_thickness_mm"])
                entered = float(st.session_state[widget_key] or 0.0)
                if entered not in (0.0,) and abs(entered - reg_val) > 2.0:
                    st.warning(f"Registration measured {reg_val:g} mm.")
        elif kind == "boolean":
            if widget_key not in st.session_state:
                st.session_state[widget_key] = bool(current_val)
            st.checkbox(label, key=widget_key, help=help_txt)
            st.session_state.answers[field] = st.session_state[widget_key]
        elif kind == "categorical":
            st.session_state.answers[field] = _render_categorical_field(
                question, widget_key, current_val
            )
        else:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = (
                    str(current_val) if current_val not in (None, "") else ""
                )
            st.text_input(label, key=widget_key, help=help_txt)
            st.session_state.answers[field] = st.session_state[widget_key]


def _render_scene_inputs() -> None:
    """Photo slot for the conversation column. Reconstruction details live in the sidebar."""
    uploaded = st.file_uploader(
        "Scene photos",
        type=["png", "jpg", "jpeg", "webp", "heic"],
        key="scene_photo",
        accept_multiple_files=True,
        help=SCENE_PHOTO_HELP,
    )
    files = list(uploaded or [])
    status = reconstruction_photo_status(len(files), MIN_IMAGES)
    st.caption(f"{status['count_label']} · {status['min_label']}")
    if not status["ready"]:
        st.markdown(status["action"])
    if files:
        first = files[0]
        st.session_state.image_bytes = first.getvalue()
        st.session_state.image_name = first.name
        preview = files[:4]
        cols = st.columns(len(preview))
        for col, item in zip(cols, preview):
            with col:
                st.image(item.getvalue(), use_container_width=True)
                st.markdown(f"**{html.escape(item.name)}**")
        if len(files) > 4:
            st.markdown(f"+{len(files) - 4} more")
    elif st.session_state.get("image_bytes"):
        name = st.session_state.get("image_name") or "photo"
        thumb, meta = st.columns([1, 3])
        with thumb:
            st.image(st.session_state.image_bytes, use_container_width=True)
        with meta:
            st.markdown(f"**{html.escape(name)}**", unsafe_allow_html=True)
    if st.button(
        "Reconstruct scene",
        use_container_width=True,
        disabled=not status["ready"],
    ):
        st.session_state.pending_registration = True
        st.rerun()
    if st.session_state.get("registration_error") and not st.session_state.get("registration_run"):
        st.error(st.session_state.registration_error)


def _phase_is_blocked(state: DesignState) -> bool:
    if _current_failure() is not None:
        return True
    if state.contract_error:
        return True
    if state.stage in (
        WorkflowStage.REJECTED,
        WorkflowStage.DESIGN_REVIEW_FAILED,
        WorkflowStage.TOPOLOGY_FAILED,
        WorkflowStage.VERIFICATION_FAILED,
    ):
        return True
    if state.feasibility is not None and not state.feasibility.feasible:
        return True
    if state.candidate_fit is not None and not state.candidate_fit.fits:
        return True
    return False


def _render_phase_stepper(phase: str, state: DesignState) -> None:
    parts = []
    items = list(stepper_states(phase, blocked=_phase_is_blocked(state)))
    for index, (name, kind) in enumerate(items):
        icon = {"done": "✓", "current": "●", "blocked": "!", "todo": "○"}.get(kind, "○")
        parts.append(
            f'<div class="mdc-phase mdc-phase-{kind}">'
            f'<div class="mdc-phase-icon">{icon}</div>'
            f'<div class="mdc-phase-label">{name.upper()}</div>'
            f"</div>"
        )
        if index < len(items) - 1:
            rail_kind = "done" if kind == "done" else "todo"
            parts.append(f'<div class="mdc-phase-rail mdc-phase-rail-{rail_kind}"></div>')
    st.markdown(f'<div class="mdc-phase-bar">{"".join(parts)}</div>', unsafe_allow_html=True)


def _render_design_summary(state: DesignState) -> None:
    rows = design_summary_rows(
        state,
        scene_status_text(
            has_photo=bool(st.session_state.get("image_bytes")),
            registration=st.session_state.get("registration_meas"),
            geometry_mode=_geometry_mode_label(st.session_state.get("mode_geom") or ""),
        ),
        answers=st.session_state.get("answers"),
    )
    if not rows:
        return
    st.markdown("**Design summary**")
    for label, value in rows:
        st.write(f"**{label}.** {value}")


def _render_primary_button(spec: ActionSpec, state: DesignState) -> None:
    if spec.kind == "download":
        path = _stl_path(state)
        if path is not None:
            st.download_button(
                spec.button or "Download STL",
                data=path.read_bytes(),
                file_name=path.name,
                mime="model/stl",
                type="primary",
                use_container_width=True,
            )
        return
    if spec.button:
        st.button(
            spec.button,
            type="primary",
            use_container_width=True,
            on_click=_on_primary_action,
        )


def _render_conversation() -> None:
    chat = list(st.session_state.get("chat") or [])
    if not chat:
        return
    latest = chat[-2:] if len(chat) >= 2 else chat
    older = chat[: -len(latest)] if len(chat) > len(latest) else []
    for item in latest:
        role = item["role"] if item["role"] in ("user", "assistant") else "assistant"
        with st.chat_message(role):
            st.markdown(item["text"])
    if older:
        with st.expander("View history"):
            for item in older:
                role = item["role"] if item["role"] in ("user", "assistant") else "assistant"
                who = "You" if role == "user" else "Assistant"
                st.markdown(f"**{who}.** {item['text']}")


def _render_request_line() -> None:
    text = (st.session_state.get("submitted_request") or _current_request_text() or "").strip()
    if text:
        shown = text if len(text) < 140 else text[:137] + "…"
        st.markdown(f'<div class="mdc-request">{html.escape(shown)}</div>', unsafe_allow_html=True)
    with st.expander("Edit request", expanded=False):
        st.text_area("Design request", key="request_text", height=90)


def _render_focused_constraint(field: str) -> None:
    from types import SimpleNamespace

    question = SimpleNamespace(field=field, question=FIELD_COPY.get(field, (field.replace("_", " "), None))[0])
    current = st.session_state.get("answers", {}).get(field)
    widget_key = f"ans_{field}"
    st.markdown('<div class="mdc-field-focus">', unsafe_allow_html=True)
    if field_widget_kind(question) == "numeric":
        if widget_key not in st.session_state:
            st.session_state[widget_key] = float(current) if current not in (None, "") else 0.0
        unit = field_unit(question)
        cols = st.columns([4, 1]) if unit else [st.container()]
        with cols[0]:
            st.number_input(field_display_label(question), key=widget_key)
        if unit:
            with cols[1]:
                st.markdown(f'<div class="mdc-unit">{html.escape(unit)}</div>', unsafe_allow_html=True)
        st.session_state.setdefault("answers", {})[field] = st.session_state[widget_key]
    else:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = "" if current in (None,) else current
        st.text_input(field_display_label(question), key=widget_key)
        st.session_state.setdefault("answers", {})[field] = st.session_state[widget_key]
    st.markdown("</div>", unsafe_allow_html=True)


def _render_failure_card(card) -> None:
    st.session_state.failure_log_card = card
    st.markdown(f'<div class="mdc-action-title">{html.escape(card.headline)}</div>', unsafe_allow_html=True)
    for item in card.findings:
        st.markdown(f"• {html.escape(item.what)} {html.escape(item.action)}")
    focus = st.session_state.get("focus_constraint_field")
    if focus:
        _render_focused_constraint(focus)
    cols = st.columns(3 if card.secondary_label else 2)
    with cols[0]:
        if card.primary_label == "Add / replace photos":
            st.button(
                card.primary_label,
                type="primary",
                use_container_width=True,
                on_click=mark_ui_transition,
                args=(st.session_state,),
            )
        else:
            st.button(
                card.primary_label,
                type="primary",
                use_container_width=True,
                on_click=_retry_same_constraints,
            )
    if card.secondary_label:
        with cols[1]:
            if card.secondary_label == "Regenerate design":
                st.button(
                    card.secondary_label,
                    use_container_width=True,
                    on_click=_regenerate_design,
                )
            else:
                field = card.constraint_field or "max_protrusion_mm"
                st.button(
                    card.secondary_label,
                    use_container_width=True,
                    on_click=_focus_constraint_field,
                    args=(field,),
                )
    with cols[-1]:
        if st.button(card.log_label, use_container_width=True):
            _failure_log_dialog()


def _render_grok_product_status() -> None:
    """Main-column Grok status. API-not-connected is not a validation failure."""
    if effective_geometry_mode(st.session_state.get("mode_geom")) != GEOM_GENERATED:
        return
    message = generator_status_message(
        get_provider("grok").configured,
        st.session_state.get("warm_start_result"),
    )
    if message == API_NOT_CONNECTED:
        st.warning(message)


def _render_required_action(spec: ActionSpec, state: DesignState) -> None:
    st.markdown(f'<div id="{ACTION_ANCHOR_ID}" class="mdc-action">', unsafe_allow_html=True)
    failure = _current_failure()
    if failure is None:
        _render_grok_product_status()
    if failure is not None:
        _render_failure_card(failure)
        if st.session_state.ui_notice:
            st.warning(st.session_state.ui_notice)
            st.session_state.ui_notice = ""
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if spec.title:
        st.markdown(f'<div class="mdc-action-title">{html.escape(spec.title)}</div>', unsafe_allow_html=True)
    if spec.subtitle and spec.kind in ("revise", "download", "none"):
        st.markdown(f'<div class="mdc-action-sub">{html.escape(spec.subtitle)}</div>', unsafe_allow_html=True)
    if spec.kind in ("continue", "optimize") and (
        spec.kind == "continue" or spec.phase in ("Describe", "Capture")
    ):
        _render_scene_inputs()
    if spec.kind in ("continue", "optimize"):
        _render_clarification_fields(state)
    if st.session_state.ui_notice:
        st.warning(st.session_state.ui_notice)
        st.session_state.ui_notice = ""
    if state.contract_error:
        st.error(state.contract_error)
    _render_primary_button(spec, state)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_left_column(spec: ActionSpec, state: DesignState) -> None:
    """Request, latest exchange, then the current required action."""
    if spec.kind == "start":
        st.markdown(f'<div id="{ACTION_ANCHOR_ID}" class="mdc-action">', unsafe_allow_html=True)
        if spec.title:
            st.markdown(
                f'<div class="mdc-action-title">{html.escape(spec.title)}</div>',
                unsafe_allow_html=True,
            )
        st.text_area("Design request", key="request_text", height=110)
        _render_grok_product_status()
        _render_scene_inputs()
        if st.session_state.ui_notice:
            st.warning(st.session_state.ui_notice)
            st.session_state.ui_notice = ""
        _render_primary_button(spec, state)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    _render_request_line()
    _render_required_action(spec, state)
    _render_conversation()

    if st.session_state.scene_observation is not None:
        with st.expander("Scene observation", expanded=False):
            _render_scene_observation(
                st.session_state.scene_observation,
                st.session_state.scene_observation_meta or {},
            )


def _placeholder(text: str) -> None:
    st.markdown(f'<div class="mdc-placeholder">{text}</div>', unsafe_allow_html=True)


def _current_candidate(state: DesignState):
    orch_candidate = getattr(st.session_state.get("orch"), "imported_candidate", None)
    return (
        state.imported_candidate
        or orch_candidate
        or imported_candidate_for_mode(st.session_state.get("mode_geom"), _candidate_name())
    )


_SCENE_GLB_FILES = (
    ("hybrid_mesh_viewer", "target_mesh_hybrid.glb"),
    ("viewer", "target.glb"),
)
_SCENE_VIEW_FILES = (
    ("top_view", "debug/world_top.png"),
    ("side_view", "debug/world_side.png"),
)
_GLB_VIEWER_MAX_BYTES = 18_000_000
_SCENE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def _ok_registration_run():
    run = st.session_state.get("registration_run")
    if run is not None and getattr(run, "ok", False):
        return run
    return None


def _existing_file(raw: Any) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    if path.is_file() and path.stat().st_size > 0:
        return path
    return None


def _artifact_file(run, *keys: str) -> Optional[Path]:
    artifacts = getattr(run, "artifacts", None) or {}
    for key in keys:
        found = _existing_file(artifacts.get(key))
        if found is not None:
            return found
    return None


def _reconstruction_file(run, key: str, relative: str) -> Optional[Path]:
    found = _artifact_file(run, key)
    if found is not None:
        return found
    out_dir = getattr(run, "out_dir", "") or ""
    if out_dir:
        return _existing_file(Path(out_dir) / relative)
    return None


def reconstruction_visuals(run) -> Dict[str, Any]:
    """Real surfcap visuals only. Uploaded photos are not reconstruction."""
    if run is None or not getattr(run, "ok", False):
        return {"glb": None, "views": [], "reconstructed": False}
    glb = None
    for key, relative in _SCENE_GLB_FILES:
        glb = _reconstruction_file(run, key, relative)
        if glb is not None:
            break
    views = [
        path
        for key, relative in _SCENE_VIEW_FILES
        if (path := _reconstruction_file(run, key, relative)) is not None
    ]
    return {"glb": glb, "views": views, "reconstructed": glb is not None or bool(views)}


def _processed_photo_count(run) -> int:
    uploads = st.session_state.get("scene_photo") or st.session_state.get("reg_capture_photos") or []
    if uploads:
        return len(list(uploads))
    capture = getattr(run, "capture_dir", "") or ""
    folder = Path(capture) if capture else None
    if folder is not None and folder.is_dir():
        return sum(1 for path in folder.iterdir() if path.suffix.lower() in _SCENE_IMAGE_SUFFIXES)
    if st.session_state.get("image_bytes"):
        return 1
    return 0


def _render_glb_viewer(path: Path) -> bool:
    data = path.read_bytes()
    if not data or len(data) > _GLB_VIEWER_MAX_BYTES:
        return False
    payload = base64.b64encode(data).decode("ascii")
    components.html(
        f"""
        <script type="module" src="https://unpkg.com/@google/model-viewer@4.0.0/dist/model-viewer.min.js"></script>
        <model-viewer
          src="data:model/gltf-binary;base64,{payload}"
          camera-controls
          touch-action="pan-y"
          shadow-intensity="0.35"
          exposure="1"
          style="width:100%;height:640px;background:#111827;border-radius:12px;">
        </model-viewer>
        """,
        height=650,
    )
    return True


def _scene_status_line(text: str) -> None:
    st.markdown(
        f'<div class="mdc-scene-status">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _render_uploaded_photos(label: str) -> bool:
    uploads = list(st.session_state.get("scene_photo") or [])
    if uploads:
        st.caption(label)
        preview = uploads[:4]
        cols = st.columns(len(preview))
        for col, item in zip(cols, preview):
            with col:
                data = item.getvalue() if hasattr(item, "getvalue") else item
                st.image(data, use_container_width=True)
                st.caption(label)
        return True
    if st.session_state.get("image_bytes"):
        st.caption(label)
        st.image(st.session_state.image_bytes, use_container_width=True)
        return True
    return False


def _render_scene_failure(card) -> None:
    st.markdown(
        f'<div class="mdc-action-title">{html.escape(card.headline)}</div>',
        unsafe_allow_html=True,
    )
    for item in card.findings:
        st.markdown(f"• {html.escape(item.what)} {html.escape(item.action)}")


def _render_scene_workspace() -> None:
    run = st.session_state.get("registration_run")
    visuals = reconstruction_visuals(_ok_registration_run())
    photo_count = _processed_photo_count(run)
    failure = _current_failure()
    reconstruction_failed = (
        failure is not None and getattr(failure, "stage", None) == STAGE_RECONSTRUCTION
    )

    if visuals["reconstructed"]:
        shown = False
        if visuals["glb"] is not None:
            shown = _render_glb_viewer(visuals["glb"])
        if not shown and visuals["views"]:
            cols = st.columns(len(visuals["views"]))
            for col, path in zip(cols, visuals["views"]):
                col.image(str(path), use_container_width=True)
            shown = True
        _scene_status_line(reconstructed_scene_status(reconstructed=True, photo_count=photo_count))
        return

    if reconstruction_failed:
        _render_scene_failure(failure)
        _render_uploaded_photos("Source photo")
        return

    status = reconstructed_scene_status(reconstructed=False, photo_count=photo_count)
    if _render_uploaded_photos("Photo preview"):
        _scene_status_line(status)
        return
    _placeholder(status)


def _render_design_workspace(state: DesignState) -> None:
    candidate = _current_candidate(state)
    has_geom = state.geometry is not None
    has_struct = state.structure is not None
    shown = False
    if candidate is not None:
        shown = True
        st.caption(
            CANDIDATES[candidate.candidate_name].label
            if candidate.candidate_name in CANDIDATES
            else candidate.candidate_name
        )
        representation = st.radio(
            "Candidate representation",
            ["Surface Mesh", "Point / Particle Representation"],
            horizontal=True,
            key="candidate_representation",
        )
        if representation == "Point / Particle Representation":
            st.plotly_chart(imported_candidate_particle_figure(candidate), use_container_width=True)
        else:
            st.plotly_chart(imported_candidate_mesh_figure(candidate), use_container_width=True)
        _render_candidate_panel(candidate, state.candidate_fit)
        if candidate.task == "generated":
            ws = st.session_state.get("warm_start_result") or {}
            with st.expander("Generated warm start — provenance", expanded=False):
                st.write(
                    {
                        "model": ws.get("model"),
                        "attempts": ws.get("attempts"),
                        "latency_s": ws.get("latency_s"),
                        "watertight": candidate.watertight,
                        "components": candidate.connected_components,
                        "faces": candidate.face_count,
                        "frame": candidate.frame,
                        "script": ws.get("script_path"),
                    }
                )
    elif has_geom and not has_struct:
        shown = True
        st.plotly_chart(geometry_figure(state.geometry), use_container_width=True)
    elif st.session_state.get("warm_start_result") and not st.session_state["warm_start_result"].get("ok"):
        shown = True
        st.caption(GENERATION_FAILED_VALIDATION)
    if has_struct:
        shown = True
        if st.session_state.get("warm_start_result") and not st.session_state["warm_start_result"].get("ok"):
            st.caption("Concept generated")
        st.plotly_chart(structure_figure(state.structure, state.geometry), use_container_width=True)
        s = state.structure
        st.caption(
            f"iteration {s.iteration} · thickness {s.parameters.support_thickness_mm:g} mm · "
            f"braces {s.parameters.brace_count} · {len(s.nodes)} nodes · {len(s.members)} members"
        )
        if state.analysis is not None:
            a = state.analysis
            c1, c2, c3 = st.columns(3)
            c1.metric("Displacement", f"{a.max_displacement_mm:.2f} mm")
            c2.metric("Stress", f"{a.max_stress_pa / 1e6:.2f} MPa")
            c3.metric("FoS", f"{a.factor_of_safety:g}" if a.factor_of_safety is not None else "n/a")
        loop_cards = iteration_cards(state)
        if loop_cards:
            last = loop_cards[-1]
            st.caption(
                f"Review {last['review']} · displacement {last['max_displacement_mm']:.2f} mm · "
                f"stress {last['max_stress_mpa']:.2f} MPa"
            )
    if not shown:
        _placeholder("Waiting for input")


def _render_optimization_workspace(state: DesignState) -> None:
    if state.topology is None:
        _placeholder("Waiting for input")
        return
    t = state.topology
    accept = t.acceptance or {}
    preview = accept.get("acceptance_status") == "unresolved_not_converged" or t.converged is False
    if preview:
        st.caption("Not converged — preview only.")
    if t.is_mock:
        st.caption("Preview only.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compliance", f"{t.compliance:.3g}")
        m2.metric("Volume fraction", f"{t.volume_fraction:.2f}")
        m3.metric("Mass vs warm start", f"{-t.mass_reduction_pct:+.0f}%")
        m4.metric("Wall time", f"{t.wall_time_s or 0:.0f} s")
    mesh_path = Path(t.optimized_geometry_ref)
    if mesh_path.suffix.lower() in {".stl", ".obj"} and mesh_path.exists():
        st.plotly_chart(
            optimized_design_figure(t.artifacts.get("candidate_mesh"), str(mesh_path)),
            use_container_width=True,
        )
        img_cols = st.columns(2)
        for col, key, caption in (
            (img_cols[0], "history_png", "Convergence history"),
            (img_cols[1], "render_png", "Density isosurface"),
        ):
            img = t.artifacts.get(key)
            if img and Path(img).exists():
                col.image(img, caption=caption, use_container_width=True)
    if t.post_check is not None:
        pc = t.post_check
        fos = f"{pc.factor_of_safety:.2f}" if pc.factor_of_safety is not None else "n/a"
        st.caption(
            f"Post-TO check: displacement {pc.max_displacement_mm:.3f} mm · "
            f"von Mises {pc.max_stress_pa / 1e6:.2f} MPa · FoS {fos}"
        )
    if state.verification is not None:
        env = accept.get("within_envelope")
        st.write(
            {
                "connectivity": (
                    "attached"
                    if accept.get("supports_attached") and accept.get("loads_attached")
                    else "pending"
                ),
                "envelope": (
                    "inside"
                    if env
                    else ("unresolved" if env == "frame_unresolved" else "pending")
                ),
                "convergence": (
                    "preview"
                    if accept.get("acceptance_status") == "unresolved_not_converged"
                    else accept.get("acceptance_status") or "pending"
                ),
            }
        )
        if engineering_evidence_is_mocked(state):
            st.caption("Physical safety unverified — mock engineering evidence is present.")
    if state.cad is not None:
        cad_path = Path(state.cad.filename)
        if cad_path.suffix.lower() in {".stl", ".obj"} and cad_path.exists():
            st.caption(cad_path.name)


def _render_right_column(state: DesignState) -> None:
    """Primary visual workspace: reconstructed scene, then design, then optimization."""
    tab = st.segmented_control(
        "Workspace",
        WORKSPACE_TABS,
        key="workspace_tab",
        required=True,
        label_visibility="collapsed",
        width="stretch",
    ) or st.session_state.get("workspace_tab") or WORKSPACE_TAB_SCENE
    if tab == WORKSPACE_TAB_SCENE:
        _render_scene_workspace()
    elif tab == WORKSPACE_TAB_OPTIMIZATION:
        _render_optimization_workspace(state)
    else:
        _render_design_workspace(state)


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


def _render_registration_debug() -> None:
    """Surfcap controls and evidence. Developer sidebar only."""
    with st.expander("Reconstruct grounding surface", expanded=False):
        for step in CAPTURE_GUIDANCE:
            st.write(f"- {step}")
        st.button(
            "Run bundled table_a capture",
            use_container_width=True,
            on_click=_load_table_a_registration,
        )
        st.file_uploader(
            "Capture photos",
            type=["jpg", "jpeg", "png", "heic"],
            accept_multiple_files=True,
            key="reg_capture_photos",
        )
        st.text_input(
            "Or use a local capture folder",
            key="reg_capture_folder",
            placeholder="/path/to/photos",
        )
        st.text_input("Surface prompt", key="reg_table_prompt", value="table")
        if st.button("Run grounding-surface reconstruction", use_container_width=True):
            st.session_state.pending_registration = True
            st.rerun()
        if st.session_state.registration_error:
            st.error(st.session_state.registration_error)
        run = st.session_state.registration_run
        if run is not None and getattr(run, "ok", False):
            meas = getattr(run, "measurements", None) or {}
            st.write(
                {
                    "confidence": meas.get("confidence"),
                    "thickness (mm)": meas.get("desk_thickness_mm"),
                    "thickness provenance": meas.get("thickness_provenance"),
                    "mount extent (mm)": meas.get("mount_extent_mm"),
                    "hybrid mesh extent (mm)": meas.get("mesh_extent_mm"),
                    "hybrid mesh watertight": meas.get("mesh_watertight"),
                    "hybrid mesh faces": meas.get("mesh_faces"),
                    "elapsed_s": getattr(run, "elapsed_s", None),
                }
            )
            warnings = getattr(run, "warnings", None) or []
            if warnings:
                with st.expander("Registration warnings"):
                    for warning in warnings:
                        st.write(f"- {warning}")
            if getattr(run, "log", ""):
                with st.expander("Surfcap log"):
                    st.code(run.log, language="text")


def _render_developer_controls(*, include_registration_input: bool = True) -> None:
    """Technical controls. Presentation only; same widgets and keys as before."""
    st.header("Examples")
    st.selectbox(
        "Example",
        EXAMPLE_CHOICES,
        key="example_choice",
        label_visibility="collapsed",
    )
    st.button("Load example", use_container_width=True, on_click=_on_load_example)
    st.button("Reset session", use_container_width=True, on_click=_on_reset_session)

    st.divider()
    st.header("Component Mode")
    st.caption(
        "The product path is Grok warm-start generation and live topology. "
        "Mock fixtures stay off unless you opt in below. Structure, analysis, design review, "
        "and safety status remain unverified. Golden Fixture is a regression test."
    )
    st.checkbox(
        "Enable mock fixtures",
        key="enable_mock_fixtures",
        help="Developer-only. Off by default. When off, topology stays Live and mock "
             "execution is not presented as a product result.",
    )
    _render_registration_debug()
    if not _mock_fixtures_enabled():
        st.session_state.mode_topo = "Live"
    if include_registration_input:
        st.text_input(
            "Registration output (surfcap target.json or scene mesh .ply, optional)",
            key="reg_target_path",
            placeholder="…/HackCMU/Generated_Scene_meshes/desk.ply  or  …/out/<scene>/target.json",
            help="Output of Aman's photo registration pipeline. Measurements are used only when their confidence is high; otherwise the user is asked.",
        )
    _reg = _registration_measurements()
    if _reg is None:
        st.caption("Registration: not provided — desk thickness comes from the structured questions.")
    elif _reg.get("error"):
        st.warning(f"Registration file could not be read: {_reg['error']}")
    elif _reg.get("prefill"):
        st.caption(
            f"Registration LIVE: desk thickness {_reg['desk_thickness_mm']} mm "
            f"({_reg['thickness_source']}, confidence {_reg['confidence']:.2f}) — pre-fills the question for confirmation."
        )
    else:
        st.caption(
            f"Registration confidence {_reg['confidence']:.2f} is too low to trust "
            f"(desk thickness {_reg.get('desk_thickness_mm')} mm); the user will be asked. {'; '.join(_reg.get('notes', [])[:2])}"
        )
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
        st.selectbox(
            "Candidate",
            list(CANDIDATES),
            key="candidate_name",
            format_func=lambda n: CANDIDATES[n].label or n,
            help="Each candidate is an STL/OBJ + _dimensions.txt + _particles.obj triple.",
        )
        _files = CANDIDATES[_candidate_name()]
        st.caption(f"{_files.mesh} · {_files.dimensions} · {_files.particles}")
    elif st.session_state.mode_geom == GEOM_GENERATED:
        st.caption(geometry_provenance_text(GEOM_GENERATED))
        _grok = get_provider("grok")
        if _grok.configured:
            st.caption(f"Generator model: {_grok.model} (writes a trimesh script; up to 3 validation retries).")
        else:
            st.warning(f"{API_NOT_CONNECTED}. {_grok.not_connected_reason}")
    else:
        st.caption(geometry_provenance_text(GEOM_ADAPTIVE))
    st.radio("Analysis", ["Mock Fixture", "Live"], index=0, disabled=True, key="mode_analysis")
    st.caption("Live analysis is not connected. This control is informational and is not a product toggle.")
    if _mock_fixtures_enabled():
        st.radio(
            "Topology",
            ["Mock Fixture", "Live"],
            key="mode_topo",
            help="Developer-only. Live runs SIMP on torch-fem (to_agent). Mock Fixture is a labelled placeholder.",
        )
    else:
        st.caption("Topology: Live (product default). Enable mock fixtures to select a placeholder.")
    if _topology_live():
        with st.expander("Topology settings", expanded=False):
            st.number_input("Element size (mm)", min_value=2.0, max_value=10.0, value=4.0, step=0.5, key="topo_elem")
            st.number_input("Max iterations", min_value=2, max_value=120, value=40, step=1, key="topo_iters")
            st.number_input("Time budget (s)", min_value=30, max_value=900, value=150, step=30, key="topo_budget")
        st.caption(
            "Live topology is required on the product path. A failure is shown as unavailable, "
            "not replaced by a mock presented as a real result."
        )
    else:
        st.caption("Mock fixture selected in developer tools. This is not the product path.")
    reasoning_choice = st.radio(
        "Reasoning Provider",
        REASONING_CHOICES,
        key="mode_reason",
        help="Selects which model answers structured reasoning calls. Any provider that "
             "supports them can replace Grok.",
    )
    st.caption(
        "Scope: warm-start geometry generation and scene observation. The typed reasoning "
        "tasks (measurement planning, hazard assessment, mechanics) are implemented and "
        "tested but **not yet called by the workflow** — questions and mechanics still come "
        "from the deterministic tables."
    )
    st.checkbox(
        "Developer mode (block on reasoning failure)",
        key="developer_mode",
        help="Product mode falls back to the deterministic tables when reasoning fails. "
             "Developer mode refuses to fall back and shows the full reasoning trace instead, "
             "so the gap can be fixed rather than hidden.",
    )
    _sync_reasoning_provider(reasoning_choice)
    _provider_now = get_provider({"Mock": "mock", "K2 Horizon": "k2_horizon", "Grok": "grok", "Cursor": "cursor"}[reasoning_choice])
    if not _provider_now.supports_structured():
        st.warning(
            f"{reasoning_choice} cannot answer structured reasoning calls"
            + (f": {_provider_now.not_connected_reason}" if _provider_now.not_connected_reason else "")
            + ". Engineering decisions will use the deterministic tables."
        )
    elif st.session_state.get("developer_mode"):
        st.caption(
            "Developer mode: when a reasoning task runs, a failure blocks instead of falling "
            "back, and the full trace is reported below."
        )
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
            if st.button("Analyze image and requirements", use_container_width=True):
                st.session_state.pending_analyze = True
                st.rerun()
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


st.set_page_config(
    page_title=BRAND_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)
_init_session()
_apply_pending_prefills()
if st.session_state.pop("pending_analyze", False):
    with st.spinner("Analyzing image and requirements with Cursor..."):
        _analyze_image_and_requirements()
if st.session_state.pop("pending_registration", False):
    with st.spinner("Reconstructing scene…"):
        _run_grounding_surface_registration()
    _apply_pending_prefills()

st.markdown(
    """
    <style>
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {
      font-family: Helvetica, Arial, sans-serif;
    }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] .stButton > button,
    [data-testid="stAppViewContainer"] .stDownloadButton > button,
    [data-testid="stAppViewContainer"] .stTextInput input,
    [data-testid="stAppViewContainer"] .stTextArea textarea,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stButton > button {
      font-family: Helvetica, Arial, sans-serif;
    }
    [data-testid="stIconMaterial"],
    [data-testid="stHeader"] [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stFileUploader"] [data-testid="stIconMaterial"],
    [data-testid="stChatMessageAvatar"] [data-testid="stIconMaterial"],
    [data-testid="stChatMessageAvatarUser"] [data-testid="stIconMaterial"],
    [data-testid="stChatMessageAvatarAssistant"] [data-testid="stIconMaterial"],
    .material-icons,
    .material-icons-outlined,
    .material-icons-round,
    .material-icons-sharp,
    .material-symbols-outlined,
    .material-symbols-rounded,
    .material-symbols-sharp {
      font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                   "Material Symbols Sharp", "Material Icons",
                   "Material Icons Outlined", "Material Icons Round" !important;
      font-style: normal !important;
      font-weight: 400 !important;
      font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
      letter-spacing: normal !important;
      text-transform: none !important;
      line-height: 1 !important;
      -webkit-font-smoothing: antialiased;
    }
    .stage-card {padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;margin-bottom:8px;background:#fff;}
    .stage-card.current {border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,0.15);}
    .stage-title {font-weight:700;font-size:0.92rem;margin-bottom:4px;}
    .mdc-phase-bar {
      display:flex;align-items:center;justify-content:space-between;
      width:100%;gap:0;margin:4px 0 18px 0;padding:2px 0 14px 0;
      border-bottom:1px solid #e5e7eb;
    }
    .mdc-phase {
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      gap:2px;padding:2px 6px;background:transparent;min-width:0;flex:0 0 auto;
    }
    .mdc-phase-icon {font-size:0.95rem;font-weight:700;line-height:1;min-height:1rem;}
    .mdc-phase-label {font-size:0.95rem;font-weight:650;letter-spacing:0.08em;}
    .mdc-phase-rail {flex:1 1 24px;height:2px;margin:0 8px;background:#e5e7eb;align-self:center;}
    .mdc-phase-rail-done {background:#86efac;}
    .mdc-phase-done {color:#047857;}
    .mdc-phase-done .mdc-phase-label {font-weight:650;}
    .mdc-phase-current {color:#1d4ed8;}
    .mdc-phase-current .mdc-phase-label {font-weight:800;}
    .mdc-phase-blocked {color:#b91c1c;}
    .mdc-phase-blocked .mdc-phase-label {font-weight:800;}
    .mdc-phase-todo {color:#9ca3af;}
    .mdc-phase-todo .mdc-phase-label {font-weight:600;}
    .mdc-brand {margin:0 0 6px 0;}
    .mdc-brand-title {font-size:2.55rem;font-weight:800;letter-spacing:-0.03em;line-height:1.05;margin:0;}
    .mdc-brand-tagline {font-size:1.35rem;font-weight:650;letter-spacing:-0.01em;margin:8px 0 0 0;color:#111827;}
    .mdc-brand-sub {font-size:0.95rem;font-weight:400;color:#6b7280;margin:6px 0 0 0;max-width:42rem;line-height:1.4;}
    .mdc-action-title {font-size:1.55rem;font-weight:750;letter-spacing:-0.02em;margin:4px 0 10px 0;}
    .mdc-action-sub {color:#4b5563;margin:0 0 10px 0;}
    .mdc-request {color:#111827;font-size:0.95rem;margin:0 0 8px 0;}
    .mdc-unit {color:#6b7280;font-size:0.95rem;padding-top:2.1rem;}
    .mdc-placeholder {
      color:#6b7280;background:#f9fafb;border:1px dashed #d1d5db;
      border-radius:10px;padding:14px 12px;margin:0 0 16px 0;
    }
    .mdc-scene-status {color:#6b7280;font-size:0.85rem;margin:8px 0 0 0;}
    .mdc-field-focus {
      border:1px solid #2563eb;box-shadow:0 0 0 3px rgba(37,99,235,0.15);
      border-radius:10px;padding:8px 10px;margin:8px 0 12px 0;
    }
    @media (max-width: 900px) {
      div[data-testid="stHorizontalBlock"] {flex-direction:column !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

orch: Orchestrator = st.session_state.orch
state = orch.state
status = pipeline_status(state)
current = highlighted_stage(state)
spec = action_spec(state, st.session_state.get("answers"), failure=_current_failure())

with st.sidebar:
    _render_developer_controls(include_registration_input=True)
    st.divider()
    st.markdown("**Raw workflow states**")
    timeline = design_loop_timeline(state)
    if timeline:
        for index, step in enumerate(timeline):
            st.markdown(
                f'<div class="stage-card"><div class="stage-title">{step["title"]} '
                f'{_badge_html(step["badge"])}</div></div>',
                unsafe_allow_html=True,
            )
    for index, name in enumerate(PIPELINE):
        badge = status[name]
        klass = "stage-card current" if name == current else "stage-card"
        st.markdown(
            f'<div class="{klass}"><div class="stage-title">{name} {_badge_html(badge)}</div></div>',
            unsafe_allow_html=True,
        )
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
    st.markdown("**Execution trace**")
    if not orch.trace:
        st.caption("No events yet.")
    for event in orch.trace:
        fields = ", ".join(event.fields_changed) if event.fields_changed else "(none)"
        st.markdown(
            f"**{event.stage_executed.value.upper()}** → `{event.next_stage.value}`  \n"
            f"{fields} · {event.component} · {event.action}"
        )
    _render_reasoning_traces()

st.markdown(
    f"""
    <div class="mdc-brand">
      <div class="mdc-brand-title">{html.escape(BRAND_TITLE)}</div>
      <div class="mdc-brand-tagline">{html.escape(BRAND_TAGLINE)}</div>
      <div class="mdc-brand-sub">{html.escape(BRAND_SUB)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
_render_phase_stepper(spec.phase, state)

left, right = st.columns([0.9, 1.25], gap="large")
with left:
    _render_left_column(spec, state)
with right:
    _render_right_column(state)

if consume_scroll_to_action(st.session_state):
    components.html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          const el = doc.getElementById("{ACTION_ANCHOR_ID}");
          if (el) {{
            el.scrollIntoView({{behavior: "smooth", block: "center"}});
          }}
        }})();
        </script>
        """,
        height=0,
    )
