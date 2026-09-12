"""Retry vs regenerate decisions. Presentation/control only; no solver logic."""

from __future__ import annotations

from typing import Optional

from ui_failure import STAGE_NONCONVERGED, STAGE_OPTIMIZATION, STAGE_WARM_START

RETRY_TOPOLOGY = "retry_topology"
REGENERATE_DESIGN = "regenerate_design"

TOPOLOGY_RETRY_STAGES = {STAGE_OPTIMIZATION, STAGE_NONCONVERGED}

PROGRESS_RETRY_TOPOLOGY = "Retrying topology optimization..."
PROGRESS_GENERATE = "Generating the warm-start mesh from your measurements…"
PROGRESS_FROM_REQUIREMENTS = "Building the design from your requirements…"
PROGRESS_OPTIMIZE = "Generating preliminary optimized design…"


def retry_action(
    *,
    failure_stage: str = "",
    has_valid_warm_start: bool,
    requirements_changed: bool = False,
    force_regenerate: bool = False,
) -> str:
    """Choose topology-only retry vs full geometry regeneration.

    Topology failure must not destroy a validated candidate.
    """
    if force_regenerate:
        return REGENERATE_DESIGN
    if (
        failure_stage in TOPOLOGY_RETRY_STAGES
        and has_valid_warm_start
        and not requirements_changed
    ):
        return RETRY_TOPOLOGY
    if failure_stage == STAGE_WARM_START:
        return REGENERATE_DESIGN
    if has_valid_warm_start and not requirements_changed and failure_stage in TOPOLOGY_RETRY_STAGES:
        return RETRY_TOPOLOGY
    return REGENERATE_DESIGN


def progress_caption(kind: Optional[str]) -> str:
    if kind == RETRY_TOPOLOGY:
        return PROGRESS_RETRY_TOPOLOGY
    if kind == REGENERATE_DESIGN or kind == "generate":
        return PROGRESS_GENERATE
    if kind == "from_requirements":
        return PROGRESS_FROM_REQUIREMENTS
    return PROGRESS_OPTIMIZE
