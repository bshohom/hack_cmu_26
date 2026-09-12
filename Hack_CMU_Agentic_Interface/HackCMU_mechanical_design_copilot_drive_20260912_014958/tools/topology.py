"""Topology optimization tool interface.

The orchestrator does not care whether a later implementation uses a
surrogate, SIMP, or another numerical method.
"""

from __future__ import annotations

from schemas import TopologyInput, TopologyOutput


def run_topology_optimization(inp: TopologyInput) -> TopologyOutput:
    """Mock deterministic topology optimization.

    TODO(Shohom): replace this placeholder with the surrogate topology
    optimization pipeline.
    """
    vf = inp.target_volume_fraction
    return TopologyOutput(
        is_mock=True,
        optimized_geometry_ref="mock://optimized_mesh",
        compliance=1.0,
        volume_fraction=vf,
        mass_reduction_pct=round((1.0 - vf) * 100.0, 1),
        solver_status="mock_converged",
        model="placeholder-surrogate",
    )
