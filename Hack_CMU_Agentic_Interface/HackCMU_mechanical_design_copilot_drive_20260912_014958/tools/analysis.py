"""Seed-sizing heuristic for the structural layout. NOT an analysis.

This picks starting member thicknesses and brace counts for the geometry that is handed to
the optimizer. It is a closed-form monotone rule — thicker members and more braces score
better — and it reads ONLY `support_thickness_mm` and `brace_count`. Payload force,
material and actual geometry do not enter, so its numbers are not predictions of anything
physical and carry no accept/reject authority over a design.

The real gate is the post-optimization linear FE check (`to_agent.integration.postcheck`),
evaluated in `Orchestrator._acceptance_verdict`. That is the only place a design can be
accepted or rejected on strength.

Naming note: the functions below keep their historical `mock_loop_*` names because tests
and the review loop import them; `is_mock=True` on the output is what the UI keys off.
"""

from __future__ import annotations

from schemas import AnalysisInput, AnalysisOutput, StructureOutput

SEED_SIZING_DISCLAIMER = (
    "SEED-SIZING HEURISTIC, NOT ANALYSIS. Closed-form function of support thickness and "
    "brace count only; load, material and geometry are not inputs. Used to size the "
    "starting layout. It cannot accept or reject a design — the post-optimization FE "
    "check does that."
)
# Back-compat for importers of the old name.
MOCK_LOOP_DISCLAIMER = SEED_SIZING_DISCLAIMER


def mock_loop_displacement_mm(structure: StructureOutput) -> float:
    """Seed-sizing score in mm: falls as thickness and braces grow. Iteration 0 is ~8."""
    thickness = structure.parameters.support_thickness_mm
    braces = structure.parameters.brace_count
    stiffness = thickness + 4.0 * max(braces - 1, 0)
    return 32.0 / max(stiffness, 1e-6)


def mock_loop_stress_pa(structure: StructureOutput) -> float:
    """Companion seed-sizing score in Pa. Not a stress prediction."""
    thickness = structure.parameters.support_thickness_mm
    braces = structure.parameters.brace_count
    stress_mpa = 24.0 / max(thickness + 2.0 * max(braces - 1, 0), 1e-6)
    return stress_mpa * 1.0e6


def run_analysis(inp: AnalysisInput, never_pass: bool = False) -> AnalysisOutput:
    """Return simulated placeholder numbers. is_mock is always True."""
    reactions = [tuple(-x for x in inp.load_force_N)]
    if never_pass:
        max_disp = 99.0
        max_stress = 9.9e7
    else:
        max_disp = mock_loop_displacement_mm(inp.structure)
        max_stress = mock_loop_stress_pa(inp.structure)

    return AnalysisOutput(
        load_case_id=inp.load_case_id,
        load_force_N=inp.load_force_N,
        is_mock=True,
        max_displacement_mm=max_disp,
        max_stress_pa=max_stress,
        factor_of_safety=None,
        reaction_forces_N=reactions,
        is_safety_validation=False,
        solver="seed-sizing-heuristic",
        solver_status="seed_sizing_only",
        disclaimer=SEED_SIZING_DISCLAIMER,
    )
