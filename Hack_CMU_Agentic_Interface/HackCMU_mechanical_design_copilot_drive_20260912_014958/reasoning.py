"""Keep run_reasoning_agent stable for the orchestrator.

The active provider is mock by default so workflow semantics do not change.
"""

from __future__ import annotations

from typing import Optional

from providers import MockReasoningProvider, ReasoningProvider
from schemas import ReasoningEffort, ReasoningResult, ReasoningRole
from state import DesignState

_provider: ReasoningProvider = MockReasoningProvider()


def set_reasoning_provider(provider: ReasoningProvider) -> None:
    global _provider
    _provider = provider


def get_reasoning_provider() -> ReasoningProvider:
    return _provider


def run_reasoning_agent(
    role: ReasoningRole,
    design_state: DesignState,
    reasoning_effort: Optional[ReasoningEffort] = None,
) -> ReasoningResult:
    """Reasoning hook used by Orchestrator. Must not invent FEM or safety."""
    return _provider.complete(role, design_state, reasoning_effort)
