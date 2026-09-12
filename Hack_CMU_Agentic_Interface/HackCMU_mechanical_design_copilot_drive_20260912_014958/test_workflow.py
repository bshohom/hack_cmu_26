"""Minimal tests for the orchestration skeleton."""

from __future__ import annotations

import unittest

from orchestrator import Orchestrator
from schemas import (
    InteractionDecision,
    RequirementsUpdate,
    SafetyStatus,
    WorkflowStage,
)

DEMO_MESSAGE = (
    "I want a cup holder attached to this desk that supports a full 1 L bottle."
)

DEMO_UPDATE = RequirementsUpdate(
    filled_bottle_mass_kg=1.1,
    bottle_diameter_mm=70.0,
    bottle_height_mm=250.0,
    desk_thickness_mm=24.0,
    attachment_method="clamp",
    allowed_contact_region="desk_front_edge",
    max_protrusion_mm=120.0,
    manufacturing_method="3d_print",
    material="PLA",
    max_part_mass_kg=0.3,
)


class WorkflowTests(unittest.TestCase):
    def test_rejects_human_supporting_structure(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request("I want a 3D-printed stool that supports a person.")
        self.assertEqual(orch.state.stage, WorkflowStage.REJECTED)
        self.assertEqual(orch.state.interaction_decision, InteractionDecision.REJECT_OR_ESCALATE)
        self.assertEqual(orch.state.safety_status, SafetyStatus.REJECTED)
        self.assertTrue(orch.state.reject_reason)

    def test_requests_missing_engineering_fields(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request(DEMO_MESSAGE)
        self.assertEqual(
            orch.state.interaction_decision,
            InteractionDecision.REQUEST_INFORMATION,
        )
        fields = {m.field for m in orch.state.missing_information}
        for required in (
            "filled_bottle_mass_kg",
            "bottle_diameter_mm",
            "desk_thickness_mm",
            "attachment_method",
            "allowed_contact_region",
            "max_protrusion_mm",
            "manufacturing_method",
        ):
            self.assertIn(required, fields)
        self.assertTrue(orch.state.vision_request.want_environment_photo)

    def test_orchestrator_runs_happy_path_with_mock_tools(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request(DEMO_MESSAGE)
        orch.apply_answers(DEMO_UPDATE)
        orch.run()

        self.assertEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertIsNotNone(orch.state.geometry)
        self.assertIsNotNone(orch.state.structure)
        self.assertIsNotNone(orch.state.analysis)
        self.assertIsNotNone(orch.state.topology)
        self.assertIsNotNone(orch.state.cad)
        self.assertIsNotNone(orch.state.verification)

        structure = orch.state.structure
        assert structure is not None
        self.assertTrue(structure.nodes)
        self.assertTrue(structure.members)
        self.assertTrue(structure.attachment_regions)
        self.assertTrue(structure.load_regions)
        self.assertTrue(structure.load_paths)
        self.assertTrue(structure.boundary_conditions)
        self.assertEqual(structure.load_cases[0].region_name, structure.load_regions[0].name)
        self.assertEqual(structure.load_regions[0].name, "cup_cavity")

        analysis = orch.state.analysis
        assert analysis is not None
        self.assertTrue(analysis.is_mock)
        self.assertFalse(analysis.is_safety_validation)
        self.assertEqual(orch.state.safety_status, SafetyStatus.UNVERIFIED)
        assert orch.state.verification is not None
        self.assertTrue(orch.state.verification.artifacts_complete)
        self.assertFalse(orch.state.verification.safety_validated)
        self.assertFalse(orch.state.vision_request.want_environment_photo)

        req = orch.state.requirements
        assert req is not None
        self.assertEqual(req.payload.filled_mass_kg, 1.1)
        self.assertEqual(req.part_mass.max_part_mass_kg, 0.3)
        self.assertEqual(req.object_geometry.bottle_diameter_mm, 70.0)
        self.assertEqual(req.design_envelope.max_protrusion_mm, 120.0)


if __name__ == "__main__":
    unittest.main()
