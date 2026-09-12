"""Deterministic structural analysis tool (mock FEM).

The LLM constructs the AnalysisInput. This tool returns typed results.
It must never be treated as safety validation.

The default mock is a parametric loop-testing model: thicker members and more
braces reduce displacement and stress. Numbers are not physically accurate.
"""

from __future__ import annotations

from schemas import AnalysisInput, AnalysisOutput, StructureOutput

MOCK_LOOP_DISCLAIMER = (
    "DETERMINISTIC MOCK RESPONSE FOR LOOP TESTING. "
    "Not real FEM. Do not treat as engineering validation."
)


def mock_loop_displacement_mm(structure: StructureOutput) -> float:
    """stiffness grows with thickness and extra braces; iteration 0 is ~8 mm."""
    thickness = structure.parameters.support_thickness_mm
    braces = structure.parameters.brace_count
    stiffness = thickness + 4.0 * max(braces - 1, 0)
    return 32.0 / max(stiffness, 1e-6)


def mock_loop_stress_pa(structure: StructureOutput) -> float:
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
        solver="deterministic-mock-loop",
        solver_status="simulated_only",
        disclaimer=MOCK_LOOP_DISCLAIMER,
    )
