"""Topology optimization tool interface.

The orchestrator does not care whether a later implementation uses a
surrogate, SIMP, or another numerical method.

Live path (Shohom): `to_agent.integration.agentic.run_topology` — SIMP on torch-fem using
the imported/generated candidate mesh as warm start. Runs when a candidate is present and
`to_agent` is importable in this environment; otherwise (or on any failure) the mock below
is returned with the reason in `notes`, so the workflow never blocks on the optimizer.
Set TO_AGENT_MODE=off to force the mock.
"""

from __future__ import annotations

import os
from pathlib import Path

from schemas import TopologyInput, TopologyOutput

OUT_ROOT = Path(__file__).resolve().parent.parent / "generated" / "topology"


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


def run_topology_optimization(inp: TopologyInput, log=None) -> TopologyOutput:
    """Live SIMP via to_agent when possible; deterministic mock otherwise."""
    if os.environ.get("TO_AGENT_MODE", "auto") == "off":
        return _mock(inp, "TO_AGENT_MODE=off: mock placeholder")
    if inp.candidate is None:
        return _mock(inp, "no candidate warm-start mesh; mock placeholder")
    try:
        run_topology = _import_adapter()
    except ImportError as exc:
        return _mock(inp, f"to_agent not importable in this environment: {exc}")
    try:
        result = run_topology(inp.model_dump(mode="json"), out_root=OUT_ROOT, log=log)
    except Exception as exc:  # noqa: BLE001 — any optimizer failure degrades to the mock
        return _mock(inp, f"live topology optimization failed: {type(exc).__name__}: {exc}")
    return TopologyOutput.model_validate(result)
