"""Chronological run log for the current optimization attempt. Presentation only."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def init_run_log(store: Dict[str, Any]) -> None:
    store.setdefault("run_log", [])
    store.setdefault("optimization_status", None)
    store.setdefault("pending_optimize", False)
    store.setdefault("optimize_reject_reason", "")
    store.setdefault("grok_called_this_run", False)
    store.setdefault("last_run_milestone", "")


def clear_run_log(store: Dict[str, Any]) -> None:
    store["run_log"] = []
    store["grok_called_this_run"] = False
    store["last_run_milestone"] = ""
    store["optimize_reject_reason"] = ""


def append_run_log(store: Dict[str, Any], event: str, detail: str = "") -> Dict[str, Any]:
    entry = {"time": _now(), "event": str(event), "detail": str(detail or "")}
    log = store.setdefault("run_log", [])
    log.append(entry)
    store["last_run_milestone"] = event
    return entry


def format_run_log(store: Dict[str, Any]) -> str:
    lines = []
    for item in store.get("run_log") or []:
        detail = f" {item['detail']}" if item.get("detail") else ""
        lines.append(f"{item.get('time', '')} {item.get('event', '')}{detail}".rstrip())
    return "\n".join(lines) if lines else "(no events yet)"


def optimize_click_decision(
    *,
    kind: str,
    has_request: bool,
    answers_complete: bool,
    stage: str = "",
) -> Optional[str]:
    """None means proceed. A string is why the click must not start optimization."""
    if kind == "optimize" and not has_request:
        return "Run optimization was not started: write a design request first."
    if kind == "optimize" and not answers_complete and stage == "request_information":
        return "Run optimization was not started: required details are still missing."
    if kind in {"start", "download", "none", "revise"}:
        return f"Run optimization was not started: current action is {kind}."
    return None


def requirements_fingerprint(requirements: Any) -> str:
    if requirements is None:
        return ""
    payload = getattr(requirements, "payload", None)
    geom = getattr(requirements, "object_geometry", None)
    env = getattr(requirements, "environment", None)
    envelope = getattr(requirements, "design_envelope", None)
    attach = getattr(requirements, "attachment", None)
    mfg = getattr(requirements, "manufacturing", None)
    extras = getattr(requirements, "task_answers", None) or {}
    extra_keys = (
        "required_reach_mm",
        "supported_load_kg",
        "attachment_structure",
        "handle_location",
        "mounting_region",
        "drilling_allowed",
        "payload_size_mm",
        "wall_clearance_mm",
    )
    parts = [
        getattr(requirements, "task_kind", ""),
        getattr(requirements, "user_message", ""),
        getattr(payload, "filled_mass_kg", None),
        getattr(geom, "kind", None),
        getattr(geom, "bottle_diameter_mm", None),
        getattr(env, "desk_thickness_mm", None),
        getattr(envelope, "max_protrusion_mm", None),
        getattr(attach, "method", None),
        getattr(attach, "allowed_contact_region", None),
        getattr(mfg, "method", None),
        *[extras.get(key) for key in extra_keys],
    ]
    return "|".join("" if item is None else str(item) for item in parts)


def should_reuse_warm_start(
    *,
    warm_start_ok: bool,
    has_candidate: bool,
    fingerprint: str,
    current_fingerprint: str,
    force_regen: bool,
) -> bool:
    if force_regen or not warm_start_ok or not has_candidate:
        return False
    if fingerprint and current_fingerprint and fingerprint != current_fingerprint:
        return False
    return True


def last_milestone(store: Dict[str, Any]) -> str:
    return str(store.get("last_run_milestone") or "")
