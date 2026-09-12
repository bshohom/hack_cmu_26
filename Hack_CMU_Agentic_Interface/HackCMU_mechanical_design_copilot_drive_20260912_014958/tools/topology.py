"""Topology optimization tool interface.

The orchestrator does not care whether a later implementation uses a
surrogate, SIMP, or another numerical method.

Live path (Shohom): `to_agent.integration.agentic.run_topology` — SIMP on torch-fem. With a
candidate mesh (imported or generated) the problem is built in that mesh's frame and warm
started; with no candidate it is built from the requirements alone and designed from
scratch, so a missing or failed warm start still yields a printable result. The mock below
is returned only when to_agent is unavailable, the optimizer fails, or TO_AGENT_MODE=off.
"""

from __future__ import annotations

import os
from pathlib import Path

from schemas import TopologyInput, TopologyOutput

OUT_ROOT = Path(__file__).resolve().parent.parent / "generated" / "topology"

# TO_AGENT_MODE:
#   "live" - a real optimization is required; failure raises TopologyUnavailable
#   "auto" - degrade to the labelled mock if the optimizer is unavailable or fails
#   "off"  - always return the mock (used by the test suite)
DEFAULT_MODE = "live"


class TopologyUnavailable(RuntimeError):
    """Live optimization was required and could not run. Carries the underlying reason."""


def _mode() -> str:
    return os.environ.get("TO_AGENT_MODE", DEFAULT_MODE).strip().lower()


def _fail(inp: TopologyInput, reason: str) -> TopologyOutput:
    """Mock in `auto`/`off`; a hard failure in `live` so nothing downstream sees a result."""
    if _mode() == "live":
        raise TopologyUnavailable(reason)
    return _mock(inp, reason)


def _mock(inp: TopologyInput, reason: str) -> TopologyOutput:
    vf = inp.target_volume_fraction
    return TopologyOutput(
        is_mock=True,
        optimized_geometry_ref="mock://optimized_mesh",
        compliance=1.0,
        volume_fraction=vf,
        mass_reduction_pct=round((1.0 - vf) * 100.0, 1),
        solver_status="mock_converged",
        model="placeholder-surrogate",
        notes=reason,
    )


def _import_adapter():
    """Lazy import: pulls torch / torch-fem only when a live run is requested."""
    from to_agent.integration.agentic import run_topology  # type: ignore[import-not-found]

    return run_topology


def estimate_seconds(inp: TopologyInput) -> float | None:
    """Predicted wall time for this problem, for a progress display. None if unavailable."""
    try:
        from to_agent.integration.agentic import estimate_topology  # type: ignore[import-not-found]

        return estimate_topology(inp.model_dump(mode="json"))
    except Exception:  # noqa: BLE001 — the estimate is advisory only
        return None


def run_topology_optimization(inp: TopologyInput, log=None, progress=None) -> TopologyOutput:
    """Live SIMP via to_agent (warm started, or from scratch).

    In the default `live` mode a failure raises TopologyUnavailable rather than returning a
    placeholder, so a run can never reach COMPLETE on a design that was never optimized.
    """
    if _mode() == "off":
        return _mock(inp, "TO_AGENT_MODE=off: mock placeholder")
    try:
        run_topology = _import_adapter()
    except ImportError as exc:
        return _fail(inp, f"to_agent not importable in this environment: {exc}")
    try:
        result = run_topology(inp.model_dump(mode="json"), out_root=OUT_ROOT, log=log, progress=progress)
    except Exception as exc:  # noqa: BLE001 — reported as-is; `live` re-raises via _fail
        return _fail(inp, f"live topology optimization failed: {type(exc).__name__}: {exc}")
    out = TopologyOutput.model_validate(result)
    # A result that names a file which is not on disk is not a result.
    ref = Path(out.optimized_geometry_ref or "")
    if not (ref.is_file() and ref.stat().st_size > 0):
        return _fail(
            inp,
            f"optimizer reported success but produced no mesh file at {out.optimized_geometry_ref!r}",
        )
    return out
