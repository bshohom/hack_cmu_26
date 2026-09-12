"""Reasoning-provider abstraction.

Agents and the orchestrator do not import K2 or Grok directly.
Credentials come from environment variables, never from source.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")

from agents.interaction import InteractionAgent
from agents.structure import StructureAgent
from cursor_adapter import (
    SceneCallResult,
    is_cursor_configured,
    not_connected_reason,
    observe_scene,
    write_upload_to_temp,
)
from schemas import (
    InteractionResult,
    ReasoningEffort,
    ReasoningResult,
    ReasoningRole,
    SceneObservation,
    StructureInput,
    StructureOutput,
)
from state import DesignState

ROLE_EFFORT = {
    ReasoningRole.INTERACTION: ReasoningEffort.LOW,
    ReasoningRole.GEOMETRY: ReasoningEffort.MEDIUM,
    ReasoningRole.STRUCTURE: ReasoningEffort.HIGH,
    ReasoningRole.ANALYSIS_INTERPRETATION: ReasoningEffort.MEDIUM,
    ReasoningRole.DESIGN_REVIEW: ReasoningEffort.HIGH,
}


class ReasoningProvider(ABC):
    name: str = "base"
    configured: bool = False
    not_connected_reason: str = ""

    def supports_structured(self) -> bool:
        """True when this provider can answer schema-constrained reasoning calls.

        These drive real engineering decisions (what to measure, how the load is reacted,
        how to revise a failed check), so they are not tied to any one vendor.
        """
        return False

    def structured_json(self, messages: List[Dict[str, Any]], timeout: int = 90) -> str:
        """Raw assistant text for a multi-turn JSON conversation (see reasoning_harness)."""
        raise NotImplementedError(f"{self.name} does not implement structured_json")

    def supports_vision(self) -> bool:
        return False

    @abstractmethod
    def complete(
        self,
        role: ReasoningRole,
        design_state: DesignState,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ) -> ReasoningResult:
        raise NotImplementedError

    def generate_structure(
        self, inp: StructureInput
    ) -> Tuple[Optional[StructureOutput], str, float]:
        """Return (output_or_none, error_or_empty, latency_s)."""
        return None, "Live implementation not connected yet.", 0.0

    def detect_missing(
        self, message: str
    ) -> Tuple[Optional[InteractionResult], str, float]:
        return None, "Live implementation not connected yet.", 0.0

    def observe_scene(
        self,
        image_bytes: bytes,
        user_text: str,
        model_id: str = "",
        image_name: Optional[str] = None,
        model_params: Optional[Dict[str, str]] = None,
        catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> SceneCallResult:
        return SceneCallResult(
            error="Live scene observation is not connected for this provider.",
            model_id=model_id,
            params=dict(model_params or {}),
        )


class MockReasoningProvider(ReasoningProvider):
    name = "mock"
    configured = True

    def complete(
        self,
        role: ReasoningRole,
        design_state: DesignState,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ) -> ReasoningResult:
        effort = reasoning_effort or ROLE_EFFORT[role]
        return ReasoningResult(
            role=role,
            effort=effort,
            notes=(
                f"mock {role.value} reasoning at {effort.value} effort "
                f"(stage={design_state.stage.value})"
            ),
            is_mock=True,
        )

    def generate_structure(
        self, inp: StructureInput
    ) -> Tuple[Optional[StructureOutput], str, float]:
        started = time.perf_counter()
        output = StructureAgent().run(inp)
        return output, "", time.perf_counter() - started

    def detect_missing(
        self, message: str
    ) -> Tuple[Optional[InteractionResult], str, float]:
        started = time.perf_counter()
        output = InteractionAgent().assess(message)
        return output, "", time.perf_counter() - started


class _OpenAICompatibleProvider(ReasoningProvider):
    """Shared HTTPS JSON client for OpenAI-style chat completions.

    Every provider on this transport (Grok, K2, any OpenAI-compatible endpoint) gets the
    structured reasoning calls and image understanding, so live reasoning is not the
    privilege of one vendor's SDK.
    """

    def __init__(self) -> None:
        self._reasoning_by_content: Dict[str, str] = {}

    def supports_structured(self) -> bool:
        return bool(self.configured)

    def structured_json(self, messages: List[Dict[str, Any]], timeout: int = 90) -> str:
        return self._chat(self._messages_with_reasoning(messages), timeout=timeout)

    def supports_vision(self) -> bool:
        return bool(self.configured)

    def observe_scene(
        self,
        image_bytes: bytes,
        user_text: str,
        model_id: str = "",
        image_name: Optional[str] = None,
        model_params: Optional[Dict[str, str]] = None,
        catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> SceneCallResult:
        """Image + text -> SceneObservation, via OpenAI-style image content parts."""
        import base64

        from cursor_adapter import SCENE_TASK_INSTRUCTION

        if not self.configured:
            return SceneCallResult(error=self.not_connected_reason, model_id=model_id or self.model,
                                   params=dict(model_params or {}))
        started = time.perf_counter()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": "Reply with JSON only. Match the requested schema."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{SCENE_TASK_INSTRUCTION}\n\nUser request: {user_text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            },
        ]
        try:
            content = self._chat(messages, timeout=120)
            observation = SceneObservation.model_validate(json.loads(content))
            observation.source = f"{self.name}_live"
            return SceneCallResult(observation=observation, model_id=model_id or self.model,
                                   params=dict(model_params or {}), latency_s=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 — a model without vision fails here
            return SceneCallResult(
                error=(
                    f"{self.name} scene observation failed ({type(exc).__name__}: {exc}). "
                    f"Model {self.model!r} may not accept images; set a vision-capable model."
                ),
                model_id=model_id or self.model,
                params=dict(model_params or {}),
                latency_s=time.perf_counter() - started,
            )

    def _chat(self, messages: List[Dict[str, Any]], timeout: int = 90) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.api_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            self._reasoning_by_content[content] = reasoning
        return content

    def _messages_with_reasoning(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """IFM/K2 requires reasoning on historical assistant turns."""
        prepared = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "assistant" and "reasoning" not in item:
                content = item.get("content") or ""
                item["reasoning"] = self._reasoning_by_content.get(content if isinstance(content, str) else "", " ")
            prepared.append(item)
        return prepared

    def complete(
        self,
        role: ReasoningRole,
        design_state: DesignState,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ) -> ReasoningResult:
        effort = reasoning_effort or ROLE_EFFORT[role]
        if not self.configured:
            return ReasoningResult(
                role=role,
                effort=effort,
                notes=self.not_connected_reason,
                is_mock=True,
            )
        prompt = (
            f"Role={role.value}. Effort={effort.value}. "
            "Return one short JSON object {\"notes\": \"...\"} about the current "
            "design stage. Do not invent FEM numbers or claim the design is safe."
        )
        try:
            content = self._chat(
                self._messages_with_reasoning(
                    [{"role": "user", "content": prompt}]
                )
            )
            notes = content
            try:
                notes = json.loads(content).get("notes", content)
            except json.JSONDecodeError:
                pass
            return ReasoningResult(
                role=role, effort=effort, notes=str(notes), is_mock=False
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return ReasoningResult(
                role=role,
                effort=effort,
                notes=f"{self.name} call failed: {exc}",
                is_mock=True,
            )

    def _json_completion(self, prompt: str, schema_model):
        started = time.perf_counter()
        if not self.configured:
            return None, self.not_connected_reason, 0.0
        try:
            content = self._chat(
                self._messages_with_reasoning(
                    [
                        {
                            "role": "system",
                            "content": "Reply with JSON only. Match the requested schema.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
            )
            parsed = json.loads(content)
            return (
                schema_model.model_validate(parsed),
                "",
                time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 — surface any provider failure
            return None, str(exc), time.perf_counter() - started

    def generate_structure(
        self, inp: StructureInput
    ) -> Tuple[Optional[StructureOutput], str, float]:
        prompt = (
            "Produce a StructureOutput JSON object for this StructureInput. "
            "Include load_case_id, nodes, members, load_paths, load_cases. "
            f"Input JSON:\n{inp.model_dump_json()}"
        )
        return self._json_completion(prompt, StructureOutput)

    def detect_missing(
        self, message: str
    ) -> Tuple[Optional[InteractionResult], str, float]:
        prompt = (
            "Produce an InteractionResult JSON object for this user message. "
            "decision must be proceed, request_information, or reject_or_escalate. "
            f"Message: {message}"
        )
        return self._json_completion(prompt, InteractionResult)


class K2ReasoningProvider(_OpenAICompatibleProvider):
    name = "k2_horizon"

    def __init__(self) -> None:
        super().__init__()
        self.api_key = (
            os.environ.get("IFM_API_KEY")
            or os.environ.get("K2_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        base = os.environ.get("IFM_API_BASE", "https://api.ifm.ai/v1").rstrip("/")
        self.api_url = f"{base}/chat/completions"
        self.model = os.environ.get("IFM_MODEL", "IFM/K2-Horizon-375B-A23B")
        self.configured = bool(self.api_key)
        self.not_connected_reason = (
            ""
            if self.configured
            else "K2 Horizon not connected. Set IFM_API_KEY or K2_API_KEY."
        )


class CursorReasoningProvider(ReasoningProvider):
    """Cursor agent SDK adapter. Live capability is scene observation only."""

    name = "cursor"

    def __init__(self) -> None:
        self.configured = is_cursor_configured()
        self.not_connected_reason = not_connected_reason()

    def complete(
        self,
        role: ReasoningRole,
        design_state: DesignState,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ) -> ReasoningResult:
        effort = reasoning_effort or ROLE_EFFORT[role]
        return ReasoningResult(
            role=role,
            effort=effort,
            notes=(
                "Cursor live reasoning is reserved for SceneObservation. "
                "It does not mutate DesignState or drive CAD/FEM/topology."
            ),
            is_mock=not self.configured,
        )

    def generate_structure(
        self, inp: StructureInput
    ) -> Tuple[Optional[StructureOutput], str, float]:
        return None, "Cursor live structure generation is not connected.", 0.0

    def detect_missing(
        self, message: str
    ) -> Tuple[Optional[InteractionResult], str, float]:
        return None, "Use SceneObservation, then InteractionAgent, for missing measurements.", 0.0

    def observe_scene(
        self,
        image_bytes: bytes,
        user_text: str,
        model_id: str = "",
        image_name: Optional[str] = None,
        model_params: Optional[Dict[str, str]] = None,
        catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> SceneCallResult:
        if not self.configured:
            return SceneCallResult(
                error=self.not_connected_reason,
                model_id=model_id,
                params=dict(model_params or {}),
            )
        path = write_upload_to_temp(image_bytes, image_name)
        try:
            return observe_scene(
                path,
                user_text,
                model_id,
                model_params=model_params,
                catalog=catalog,
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class GrokReasoningProvider(_OpenAICompatibleProvider):
    name = "grok"

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()
        load_dotenv(Path(__file__).resolve().parent / ".env")
        self.api_key = (
            os.environ.get("GROK_API_KEY")
            or os.environ.get("XAI_API_KEY")
            or ""
        )
        base = os.environ.get("GROK_API_BASE", "https://api.x.ai/v1").rstrip("/")
        self.api_url = f"{base}/chat/completions"
        self.model = os.environ.get("GROK_MODEL", "grok-3")
        self.configured = bool(self.api_key)
        self.not_connected_reason = (
            ""
            if self.configured
            else "Grok not connected. Set GROK_API_KEY or XAI_API_KEY."
        )


def get_provider(name: str) -> ReasoningProvider:
    if name == "k2_horizon":
        return K2ReasoningProvider()
    if name == "grok":
        return GrokReasoningProvider()
    if name == "cursor":
        return CursorReasoningProvider()
    return MockReasoningProvider()
