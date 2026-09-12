"""Cursor SDK adapter for live image + requirement understanding.

This is a ReasoningProvider helper, not a chat-completions client.
It must not mutate DesignState. Exact dimensions from an image are not
authoritative engineering inputs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from schemas import SceneObservation

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")

SCENE_TASK_INSTRUCTION = """You are analyzing a mechanical-design scene for a downstream deterministic engineering workflow.

Inspect the image and user request.

Return only information that can reasonably be inferred from the image and text.

Do not invent exact dimensions unless a known reference or explicit user measurement is present.

Identify:
- payload/object
- environment/support
- likely attachment region
- plausible attachment mechanisms
- obvious design constraints
- what engineering measurements are still required
- uncertainty

Do not claim structural safety.
Do not perform FEM.
Do not perform topology optimization.
Do not generate final CAD.

Return structured JSON only, matching this schema:
{
  "detected_payload_type": string,
  "detected_support_type": string,
  "likely_attachment_regions": [string],
  "likely_attachment_methods": [string],
  "visible_constraints": [string],
  "inferred_values": object,
  "missing_measurements": [string],
  "uncertainties": [string],
  "assumptions": [string],
  "confidence": number between 0 and 1,
  "source": "cursor_live"
}

Put exact numeric dimensions in inferred_values only when the user text or a visible scale/reference makes them reliable. Otherwise list them in missing_measurements and uncertainties.
"""

NOT_CONNECTED = "Cursor live reasoning not connected"

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Catalog IDs are discovered at runtime. These substrings only reorder
# already-returned IDs; they never invent a model name.
_PREFERRED_ID_FRAGMENTS = (
    "gpt-5.6-sol",
    "gpt-5.6",
    "gpt-5",
    "claude-opus",
    "opus",
    "claude-sonnet-5",
    "composer-2.5",
    "composer",
)

# Recommended defaults. Applied only when the live catalog contains the ID
# and the named parameter/value pair.
DEFAULT_MODEL_ID = "gpt-5.6-sol"
DEFAULT_PARAM_HINTS = {
    "context": "272k",
    "reasoning": "high",
    "fast": "false",
}
COMPARE_A_MODEL_ID = "gpt-5.6-sol"
COMPARE_A_PARAM_HINTS = {
    "context": "272k",
    "reasoning": "high",
    "fast": "false",
}
COMPARE_B_MODEL_ID = "claude-sonnet-5"
COMPARE_B_PARAM_HINTS = {
    "thinking": "true",
    "effort": "high",
}

_catalog_cache: Optional[Tuple[List[Dict[str, Any]], str]] = None


def cursor_api_key() -> str:
    return (os.getenv("CURSOR_API_KEY") or "").strip()


def python_supports_sdk() -> bool:
    return sys.version_info >= (3, 10)


def sdk_import_error() -> str:
    if not python_supports_sdk():
        return (
            "cursor-sdk requires Python >= 3.10. "
            "Mock / Adaptive workflow is still available."
        )
    try:
        import cursor_sdk  # noqa: F401
    except ImportError as exc:
        return f"cursor-sdk is not installed ({exc}). {NOT_CONNECTED}."
    return ""


def is_cursor_configured() -> bool:
    return bool(cursor_api_key()) and not sdk_import_error()


def not_connected_reason() -> str:
    if not cursor_api_key():
        return NOT_CONNECTED + ". Set CURSOR_API_KEY."
    err = sdk_import_error()
    if err:
        return err
    return ""


def _model_to_dict(model: Any) -> Dict[str, Any]:
    parameters = []
    for param in getattr(model, "parameters", ()) or ():
        values = []
        for item in getattr(param, "values", ()) or ():
            values.append(
                {
                    "value": getattr(item, "value", ""),
                    "display_name": getattr(item, "display_name", ""),
                }
            )
        parameters.append(
            {
                "id": getattr(param, "id", ""),
                "display_name": getattr(param, "display_name", ""),
                "values": values,
            }
        )
    variants = []
    for variant in getattr(model, "variants", ()) or ():
        params = []
        for item in getattr(variant, "params", ()) or ():
            params.append(
                {
                    "id": getattr(item, "id", ""),
                    "value": getattr(item, "value", ""),
                }
            )
        variants.append(
            {
                "display_name": getattr(variant, "display_name", ""),
                "description": getattr(variant, "description", ""),
                "is_default": bool(getattr(variant, "is_default", False)),
                "params": params,
            }
        )
    return {
        "id": getattr(model, "id", ""),
        "display_name": getattr(model, "display_name", ""),
        "description": getattr(model, "description", ""),
        "parameters": parameters,
        "variants": variants,
    }


def _sort_catalog(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def rank(item: Dict[str, Any]) -> Tuple[int, str]:
        mid = str(item.get("id") or "").lower()
        for index, fragment in enumerate(_PREFERRED_ID_FRAGMENTS):
            if fragment in mid:
                return (index, mid)
        return (len(_PREFERRED_ID_FRAGMENTS), mid)

    return sorted(models, key=rank)


def find_catalog_model(
    catalog: List[Dict[str, Any]], model_id: str
) -> Optional[Dict[str, Any]]:
    wanted = (model_id or "").strip()
    for item in catalog:
        if item.get("id") == wanted:
            return item
    return None


def pick_catalog_model_id(
    catalog: List[Dict[str, Any]], preferred_id: str
) -> str:
    ids = [item.get("id") or "" for item in catalog if item.get("id")]
    if preferred_id in ids:
        return preferred_id
    return ids[0] if ids else ""


def _is_1m_context(value: str) -> bool:
    text = str(value).strip().lower().replace(" ", "")
    if "272" in text:
        return False
    return text in {"1m", "1000k", "1024k", "1048576"} or text.startswith("1m")


def _param_values(param: Dict[str, Any]) -> List[str]:
    values = []
    for item in param.get("values") or []:
        value = item.get("value")
        if value is None or value == "":
            continue
        values.append(str(value))
    return values


def _match_param(model: Dict[str, Any], hint_name: str) -> Optional[Dict[str, Any]]:
    hint = (hint_name or "").strip().lower().replace("-", "_")
    if not hint:
        return None
    parameters = model.get("parameters") or []
    for param in parameters:
        pid = str(param.get("id") or "").strip().lower().replace("-", "_")
        if pid == hint:
            return param
    for param in parameters:
        display = str(param.get("display_name") or "").strip().lower().replace("-", "_")
        pid = str(param.get("id") or "").strip().lower().replace("-", "_")
        if display == hint or hint in pid.split("_") or hint in display.split(" "):
            return param
    return None


def _match_param_value(param: Dict[str, Any], hint_value: str) -> Optional[str]:
    allowed = _param_values(param)
    if not allowed:
        return None
    hint = str(hint_value).strip()
    for value in allowed:
        if value == hint:
            return value
    hint_l = hint.lower()
    for value in allowed:
        if value.lower() == hint_l:
            return value
    for value in allowed:
        if hint_l in value.lower() or value.lower() in hint_l:
            return value
    return None


def resolve_param_values(
    model: Optional[Dict[str, Any]],
    hints: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Keep only catalog-backed parameter names and values.

    Context defaults never select a 1M window.
    """
    if not model:
        return {}
    resolved: Dict[str, str] = {}
    for hint_name, hint_value in (hints or {}).items():
        param = _match_param(model, hint_name)
        if param is None:
            continue
        pid = str(param.get("id") or "").strip()
        if not pid:
            continue
        matched = _match_param_value(param, str(hint_value))
        if matched is None:
            continue
        if pid.lower() == "context" or hint_name.lower() == "context":
            if _is_1m_context(matched):
                continue
        resolved[pid] = matched

    context_param = _match_param(model, "context")
    if context_param is not None:
        pid = str(context_param.get("id") or "").strip()
        if pid and pid not in resolved:
            for value in _param_values(context_param):
                if not _is_1m_context(value):
                    # Prefer 272k-like values if present, else first non-1M.
                    if "272" in value.lower():
                        resolved[pid] = value
                        break
            if pid not in resolved:
                for value in _param_values(context_param):
                    if not _is_1m_context(value):
                        resolved[pid] = value
                        break
    return resolved


def filter_params_for_model(
    model: Optional[Dict[str, Any]], params: Optional[Dict[str, str]]
) -> Dict[str, str]:
    if not model or not params:
        return {}
    allowed: Dict[str, List[str]] = {}
    for param in model.get("parameters") or []:
        pid = str(param.get("id") or "").strip()
        if pid:
            allowed[pid] = _param_values(param)
    filtered: Dict[str, str] = {}
    for key, value in params.items():
        if key in allowed and str(value) in allowed[key]:
            filtered[key] = str(value)
    return filtered


@dataclass
class SceneCallResult:
    observation: Optional[SceneObservation] = None
    error: str = ""
    latency_s: float = 0.0
    model_id: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    image_accepted: Optional[bool] = None
    schema_valid: bool = False
    raw_text: str = ""


def list_cursor_models(force: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    """Discover account models. Never guess IDs."""
    global _catalog_cache
    if _catalog_cache is not None and not force:
        return _catalog_cache
    if not cursor_api_key():
        return [], NOT_CONNECTED + ". Set CURSOR_API_KEY."
    err = sdk_import_error()
    if err:
        return [], err
    try:
        from cursor_sdk import Cursor

        models = Cursor.models.list(api_key=cursor_api_key())
        catalog = _sort_catalog([_model_to_dict(model) for model in models])
        catalog = [item for item in catalog if item.get("id")]
        if not catalog:
            return [], "Cursor.models.list() returned no model IDs for this account."
        _catalog_cache = (catalog, "")
        return _catalog_cache
    except Exception as exc:  # noqa: BLE001 — surface catalog failures to the UI
        return [], f"Cursor.models.list() failed: {exc}"


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Cursor agent returned empty text.")
    for candidate in (
        raw,
        *([m.group(1) for m in _JSON_FENCE.finditer(raw)] or []),
    ):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    match = _JSON_OBJECT.search(raw)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Cursor agent did not return a JSON object.")


def _normalize_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def parse_scene_observation(payload: Any) -> SceneObservation:
    if isinstance(payload, SceneObservation):
        observation = payload
    elif isinstance(payload, str):
        observation = SceneObservation.model_validate(extract_json_object(payload))
    else:
        observation = SceneObservation.model_validate(payload)
    data = observation.model_dump()
    data["source"] = "cursor_live"
    data["confidence"] = _normalize_confidence(data.get("confidence"))
    if not isinstance(data.get("inferred_values"), dict):
        data["inferred_values"] = {}
    return SceneObservation.model_validate(data)


def observation_to_requirements_update(_observation: SceneObservation) -> Dict[str, Any]:
    """SceneObservation never becomes authoritative requirements by itself."""
    return {}


def _user_prompt(user_text: str) -> str:
    return (
        f"{SCENE_TASK_INSTRUCTION}\n\n"
        f"User design request:\n{user_text.strip()}\n"
    )


def observe_scene(
    image_path: str,
    user_text: str,
    model_id: str,
    model_params: Optional[Dict[str, str]] = None,
    catalog: Optional[List[Dict[str, Any]]] = None,
) -> SceneCallResult:
    """Send image + text to a Cursor agent."""
    started = time.perf_counter()
    model_name = (model_id or "").strip()
    params = dict(model_params or {})
    result = SceneCallResult(model_id=model_name, params=params)

    def _finish(error: str = "", observation: Optional[SceneObservation] = None) -> SceneCallResult:
        result.error = error
        result.observation = observation
        result.schema_valid = observation is not None
        result.latency_s = time.perf_counter() - started
        return result

    if not is_cursor_configured():
        return _finish(not_connected_reason())
    if not model_name:
        return _finish("No Cursor model selected. Inspect the account catalog first.")
    if not image_path or not os.path.isfile(image_path):
        result.image_accepted = False
        return _finish("Upload an image before analyzing.")
    if not (user_text or "").strip():
        return _finish("Write a design request first.")

    models = catalog
    if models is None:
        models, _catalog_error = list_cursor_models()
    model_entry = find_catalog_model(models, model_name)
    if model_entry is not None:
        params = filter_params_for_model(model_entry, params)
        result.params = params
    elif params:
        # Do not send parameters that were not confirmed against the catalog.
        params = {}
        result.params = {}

    workspace = tempfile.mkdtemp(prefix="cursor_scene_")
    try:
        from cursor_sdk import (
            Agent,
            AgentOptions,
            LocalAgentOptions,
            ModelParameterValue,
            ModelSelection,
            SDKImage,
            UserMessage,
        )

        image = SDKImage.from_file(image_path)
        selection: Any = model_name
        if params:
            selection = ModelSelection(
                id=model_name,
                params=tuple(
                    ModelParameterValue(id=key, value=value)
                    for key, value in params.items()
                ),
            )
        options = AgentOptions(
            model=selection,
            api_key=cursor_api_key(),
            local=LocalAgentOptions(cwd=workspace),
            tools=[],
        )
        with Agent.create(options) as agent:
            used = getattr(agent, "model", None)
            used_id = getattr(used, "id", None) or model_name
            result.model_id = used_id
            used_params = {}
            for item in getattr(used, "params", ()) or ():
                pid = getattr(item, "id", "")
                pval = getattr(item, "value", "")
                if pid:
                    used_params[str(pid)] = str(pval)
            if used_params:
                result.params = used_params
            run = agent.send(
                UserMessage(
                    text=_user_prompt(user_text),
                    images=[image],
                )
            )
            result.image_accepted = True
            text = run.text() or ""
        result.raw_text = text
        try:
            observation = parse_scene_observation(text)
        except Exception as exc:  # noqa: BLE001 — invalid model JSON is a call error
            return _finish(str(exc))
        return _finish(observation=observation)
    except Exception as exc:  # noqa: BLE001 — adapter must not crash the UI
        message = str(exc)
        lowered = message.lower()
        if any(token in lowered for token in ("image", "mime", "png", "jpeg", "webp")):
            if result.image_accepted is None:
                result.image_accepted = False
        return _finish(message)
    finally:
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except OSError:
            pass


def write_upload_to_temp(image_bytes: bytes, image_name: Optional[str] = None) -> str:
    suffix = os.path.splitext(image_name or "")[1].lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    handle = tempfile.NamedTemporaryFile(prefix="scene_upload_", suffix=suffix, delete=False)
    try:
        handle.write(image_bytes)
        handle.flush()
        return handle.name
    finally:
        handle.close()
