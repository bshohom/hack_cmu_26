"""Requirement feasibility gate and structure / analysis / review loop."""

from __future__ import annotations

import unittest

from orchestrator import Orchestrator
from schemas import (
    RequirementsUpdate,
    ReviewDecision,
    SafetyStatus,
    WorkflowStage,
)
from tools.analysis import mock_loop_displacement_mm

DEMO_MESSAGE = (
    "I want a cup holder attached to this desk that supports a full 1 L bottle."
)


def _feasible_update(**overrides) -> RequirementsUpdate:
    payload = dict(
        filled_bottle_mass_kg=1.1,
        bottle_diameter_mm=100.0,
        bottle_height_mm=250.0,
        desk_thickness_mm=24.0,
        attachment_method="clamp",
        allowed_contact_region="desk_front_edge",
        max_protrusion_mm=150.0,
        manufacturing_method="3d_print",
        material="PLA",
        max_part_mass_kg=0.3,
    )
    payload.update(overrides)
    return RequirementsUpdate(**payload)


def _run(update: RequirementsUpdate, **orch_kwargs) -> Orchestrator:
    orch = Orchestrator(**orch_kwargs)
    orch.ingest_user_request(DEMO_MESSAGE)
    orch.apply_answers(update)
    orch.run()
    return orch


class FeasibilityGateTests(unittest.TestCase):
    def test_payload_exceeds_tiny_envelope_is_blocked_before_structure(self) -> None:
        orch = _run(
            _feasible_update(bottle_diameter_mm=100.0, max_protrusion_mm=0.2)
        )
        self.assertEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertIsNotNone(orch.state.feasibility)
        self.assertFalse(orch.state.feasibility.feasible)
        self.assertTrue(orch.state.feasibility.required_user_revision)
        codes = [v.code for v in orch.state.feasibility.violations]
        self.assertIn("payload_exceeds_design_envelope", codes)
        self.assertIsNotNone(orch.state.geometry)
        self.assertIsNone(orch.state.structure)
        self.assertIsNone(orch.state.analysis)
        self.assertIsNone(orch.state.topology)
        self.assertIn("INFEASIBLE", orch.state.feasibility.message)

    def test_feasible_payload_starts_structure(self) -> None:
        orch = _run(_feasible_update(bottle_diameter_mm=100.0, max_protrusion_mm=150.0))
        self.assertIsNotNone(orch.state.structure)
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertIsNotNone(orch.state.feasibility)
        self.assertTrue(orch.state.feasibility.feasible)


class DesignLoopTests(unittest.TestCase):
    def test_iteration_zero_fails_then_parameters_and_analysis_change(self) -> None:
        orch = _run(_feasible_update())
        self.assertGreaterEqual(len(orch.state.design_iterations), 2)
        first = orch.state.design_iterations[0]
        second = orch.state.design_iterations[1]
        self.assertEqual(first.iteration, 0)
        self.assertEqual(first.review.decision, ReviewDecision.REVISE)
        self.assertGreater(first.analysis.max_displacement_mm, 3.0)
        self.assertEqual(first.structure.parameters.support_thickness_mm, 4.0)
        self.assertEqual(first.structure.parameters.brace_count, 1)
        self.assertEqual(second.iteration, 1)
        self.assertGreater(
            second.structure.parameters.support_thickness_mm,
            first.structure.parameters.support_thickness_mm,
        )
        self.assertGreater(
            second.structure.parameters.brace_count,
            first.structure.parameters.brace_count,
        )
        self.assertNotEqual(
            first.analysis.max_displacement_mm,
            second.analysis.max_displacement_mm,
        )
        self.assertLess(
            second.analysis.max_displacement_mm,
            first.analysis.max_displacement_mm,
        )
        self.assertNotEqual(first.structure, second.structure)

    def test_eventual_pass_starts_topology(self) -> None:
        orch = _run(_feasible_update())
        self.assertEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertIsNotNone(orch.state.topology)
        self.assertTrue(orch.state.design_iterations)
        last = orch.state.design_iterations[-1]
        self.assertEqual(last.review.decision, ReviewDecision.PASS)
        self.assertLessEqual(last.analysis.max_displacement_mm, 3.0)
        self.assertTrue(any(item.review.decision == ReviewDecision.REVISE for item in orch.state.design_iterations))
        self.assertEqual(orch.state.safety_status, SafetyStatus.UNVERIFIED)

    def test_never_pass_does_not_start_topology(self) -> None:
        orch = _run(_feasible_update(), analysis_never_pass=True)
        self.assertEqual(orch.state.stage, WorkflowStage.DESIGN_REVIEW_FAILED)
        self.assertIsNone(orch.state.topology)
        self.assertIsNone(orch.state.cad)
        self.assertTrue(orch.state.design_iterations)
        self.assertTrue(
            all(item.review.decision == ReviewDecision.REVISE for item in orch.state.design_iterations)
        )
        self.assertEqual(len(orch.state.design_iterations), orch.max_structure_iterations)

    def test_iteration_history_is_retained(self) -> None:
        orch = _run(_feasible_update())
        history = orch.state.design_iterations
        self.assertGreaterEqual(len(history), 2)
        iterations = [item.iteration for item in history]
        self.assertEqual(iterations, list(range(len(history))) )
        for item in history:
            self.assertIsNotNone(item.structure)
            self.assertIsNotNone(item.analysis)
            self.assertIsNotNone(item.review)
            self.assertEqual(item.iteration, item.structure.iteration)
            self.assertEqual(item.iteration, item.review.iteration)
        latest = history[-1]
        self.assertEqual(orch.state.structure.iteration, latest.structure.iteration)
        self.assertEqual(
            orch.state.analysis.max_displacement_mm,
            latest.analysis.max_displacement_mm,
        )
        self.assertNotEqual(history[0].structure.parameters, history[-1].structure.parameters)

    def test_parametric_mock_matches_expected_iteration_zero(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request(DEMO_MESSAGE)
        orch.apply_answers(_feasible_update())
        while orch.state.stage != WorkflowStage.ANALYSIS:
            orch.step()
        orch.step()
        structure = orch.state.structure
        assert structure is not None
        analysis = orch.state.analysis
        assert analysis is not None
        self.assertAlmostEqual(mock_loop_displacement_mm(structure), 8.0)
        self.assertAlmostEqual(analysis.max_displacement_mm, 8.0)
        self.assertTrue(analysis.is_mock)
        self.assertFalse(analysis.is_safety_validation)
        # The loop still sizes the seed geometry, but it no longer presents itself as
        # analysis and no longer has authority to accept or reject a design.
        self.assertIn("SEED-SIZING HEURISTIC, NOT ANALYSIS", analysis.disclaimer)
        self.assertIn("cannot accept or reject a design", analysis.disclaimer)
        self.assertEqual(analysis.solver, "seed-sizing-heuristic")


if __name__ == "__main__":
    unittest.main()
