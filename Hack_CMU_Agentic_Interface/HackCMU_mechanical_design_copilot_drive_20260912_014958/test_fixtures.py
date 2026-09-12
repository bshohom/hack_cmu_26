"""Contract tests for fixture-driven orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from fixtures import (
    EXAMPLE_DIR,
    load_clarifications,
    load_expected_structure,
    load_expected_verification,
    load_integration_fixtures,
    load_user_request,
)
from orchestrator import Orchestrator
from schemas import (
    AnalysisOutput,
    CadOutput,
    GeometryOutput,
    InteractionDecision,
    MassProvenance,
    RegistrationOutput,
    RequirementsUpdate,
    SafetyStatus,
    StructureOutput,
    TopologyOutput,
    VerificationResult,
    WorkflowStage,
)


MISSING_MESSAGE = "I want a cup holder attached to this desk."
REJECT_MESSAGE = "Design a stool that supports a 100 kg person."
PAYLOAD_MASS_KG = 1.1
PART_ESTIMATED_MASS_KG = 0.18
G = 9.81


def _happy_orchestrator() -> Orchestrator:
    orch = Orchestrator(fixtures=load_integration_fixtures())
    orch.ingest_user_request(load_user_request())
    orch.apply_answers(load_clarifications())
    orch.run()
    return orch


class FixtureContractTests(unittest.TestCase):
    def test_fixture_json_validates_against_pydantic_models(self) -> None:
        RegistrationOutput.model_validate_json(
            (EXAMPLE_DIR / "02_registration_output.json").read_text()
        )
        GeometryOutput.model_validate_json(
            (EXAMPLE_DIR / "03_geometry_output.json").read_text()
        )
        StructureOutput.model_validate_json(
            (EXAMPLE_DIR / "04_structure_output.json").read_text()
        )
        AnalysisOutput.model_validate_json(
            (EXAMPLE_DIR / "05_analysis_output.json").read_text()
        )
        TopologyOutput.model_validate_json(
            (EXAMPLE_DIR / "06_topology_output.json").read_text()
        )
        VerificationResult.model_validate_json(
            (EXAMPLE_DIR / "07_verification_output.json").read_text()
        )
        CadOutput.model_validate_json(
            (EXAMPLE_DIR / "08_cad_output.json").read_text()
        )
        RequirementsUpdate.model_validate_json(
            (EXAMPLE_DIR / "01_user_clarifications.json").read_text()
        )
        self.assertIn("message", json.loads((EXAMPLE_DIR / "00_user_request.json").read_text()))

        analysis = AnalysisOutput.model_validate_json(
            (EXAMPLE_DIR / "05_analysis_output.json").read_text()
        )
        topology = TopologyOutput.model_validate_json(
            (EXAMPLE_DIR / "06_topology_output.json").read_text()
        )
        geometry = GeometryOutput.model_validate_json(
            (EXAMPLE_DIR / "03_geometry_output.json").read_text()
        )
        self.assertTrue(analysis.is_mock)
        self.assertFalse(analysis.is_safety_validation)
        self.assertTrue(topology.is_mock)
        self.assertTrue(geometry.is_mock)

    def test_omitted_is_mock_fails_schema_validation(self) -> None:
        registration = json.loads((EXAMPLE_DIR / "02_registration_output.json").read_text())
        del registration["is_mock"]
        with self.assertRaises(ValidationError):
            RegistrationOutput.model_validate(registration)

        geometry = json.loads((EXAMPLE_DIR / "03_geometry_output.json").read_text())
        del geometry["is_mock"]
        with self.assertRaises(ValidationError):
            GeometryOutput.model_validate(geometry)

        analysis = json.loads((EXAMPLE_DIR / "05_analysis_output.json").read_text())
        del analysis["is_mock"]
        with self.assertRaises(ValidationError):
            AnalysisOutput.model_validate(analysis)

        topology = json.loads((EXAMPLE_DIR / "06_topology_output.json").read_text())
        del topology["is_mock"]
        with self.assertRaises(ValidationError):
            TopologyOutput.model_validate(topology)

        cad = json.loads((EXAMPLE_DIR / "08_cad_output.json").read_text())
        del cad["is_mock"]
        with self.assertRaises(ValidationError):
            CadOutput.model_validate(cad)

    def test_happy_path_consumes_teammate_fixtures(self) -> None:
        orch = _happy_orchestrator()
        fixtures = load_integration_fixtures()

        self.assertEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertIsNone(orch.state.contract_error)
        self.assertIsNotNone(orch.state.geometry)
        self.assertIsNotNone(orch.state.structure)
        self.assertIsNotNone(orch.state.analysis)
        self.assertIsNotNone(orch.state.topology)
        self.assertIsNotNone(orch.state.cad)
        self.assertIsNotNone(orch.state.verification)

        self.assertEqual(orch.state.geometry, fixtures.geometry)
        self.assertEqual(orch.state.registration, fixtures.registration)
        self.assertEqual(orch.state.analysis, fixtures.analysis)
        self.assertEqual(orch.state.topology, fixtures.topology)
        self.assertEqual(orch.state.cad, fixtures.cad)
        self.assertEqual(orch.state.structure, load_expected_structure())
        self.assertEqual(orch.state.verification, load_expected_verification())

        self.assertEqual(orch.state.geometry.payload_object.bottle_diameter_mm, 85.0)
        self.assertEqual(orch.state.geometry.environment.desk_thickness_mm, 25.0)
        self.assertEqual(orch.state.geometry.design_envelope.max_protrusion_mm, 130.0)
        self.assertEqual(
            orch.state.registration.frame_id,
            orch.state.geometry.coordinate_frame,
        )

        structure = orch.state.structure
        assert structure is not None
        self.assertEqual(len(structure.nodes), 4)
        self.assertEqual(len(structure.members), 5)
        self.assertEqual({n.id for n in structure.nodes}, {"mount_upper", "mount_lower", "cup_ring", "cup_base"})
        self.assertTrue(structure.boundary_conditions)
        self.assertTrue(structure.load_paths)

        self.assertEqual(orch.state.safety_status, SafetyStatus.UNVERIFIED)
        self.assertTrue(orch.state.verification.artifacts_complete)
        self.assertFalse(orch.state.verification.safety_validated)
        self.assertTrue(orch.state.analysis.is_mock)
        self.assertTrue(orch.state.topology.is_mock)
        self.assertFalse(orch.state.vision_request.want_environment_photo)
        self.assertEqual(orch.state.vision_request.requested_measurements, [])
        self.assertEqual(
            orch.state.resolved_payload_mass.provenance,
            MassProvenance.GEOMETRY_OUTPUT,
        )
        self.assertEqual(orch.state.analysis.load_case_id, "static_gravity")

        self.assertNotIn("VERIFIED", SafetyStatus.__members__)
        self.assertNotIn("SAFE", SafetyStatus.__members__)

        components = [event.component for event in orch.trace]
        self.assertIn("InteractionAgent", components)
        self.assertIn("Yujie fixture", components)
        self.assertIn("StructureAgent", components)
        self.assertIn("analysis fixture", components)
        self.assertIn("Shohom fixture", components)

    def test_payload_mass_is_used_for_gravity_loading(self) -> None:
        orch = _happy_orchestrator()
        force = orch.state.structure.load_cases[0].force_N[2]
        self.assertAlmostEqual(force, -G * PAYLOAD_MASS_KG, places=5)
        self.assertNotAlmostEqual(force, -G * PART_ESTIMATED_MASS_KG, places=2)
        self.assertIn("payload.filled_mass_kg=1.1", orch.state.structure.load_cases[0].notes)
        self.assertNotIn("0.18", orch.state.structure.load_cases[0].notes)

    def test_trace_records_stage_executed_and_next_stage(self) -> None:
        orch = _happy_orchestrator()
        geometry_events = [event for event in orch.trace if event.action == "load_geometry"]
        self.assertEqual(len(geometry_events), 1)
        self.assertEqual(geometry_events[0].stage_executed, WorkflowStage.GEOMETRY)
        self.assertEqual(geometry_events[0].next_stage, WorkflowStage.FEASIBILITY_CHECK)

        feasibility_events = [event for event in orch.trace if event.action == "feasibility_check"]
        self.assertEqual(feasibility_events[0].stage_executed, WorkflowStage.FEASIBILITY_CHECK)
        self.assertEqual(feasibility_events[0].next_stage, WorkflowStage.STRUCTURE)

        structure_events = [event for event in orch.trace if event.action == "run_structure"]
        self.assertEqual(structure_events[0].stage_executed, WorkflowStage.STRUCTURE)
        self.assertEqual(structure_events[0].next_stage, WorkflowStage.ANALYSIS)

        analysis_events = [event for event in orch.trace if event.action == "load_analysis"]
        self.assertEqual(analysis_events[0].stage_executed, WorkflowStage.ANALYSIS)
        self.assertEqual(analysis_events[0].next_stage, WorkflowStage.DESIGN_REVIEW)

        review_events = [event for event in orch.trace if event.action == "review_design"]
        self.assertEqual(review_events[0].stage_executed, WorkflowStage.DESIGN_REVIEW)
        self.assertEqual(review_events[0].next_stage, WorkflowStage.TOPOLOGY_OPTIMIZATION)

    def test_missing_information_blocks_geometry(self) -> None:
        orch = Orchestrator(fixtures=load_integration_fixtures())
        orch.ingest_user_request(MISSING_MESSAGE)
        self.assertEqual(orch.state.interaction_decision, InteractionDecision.REQUEST_INFORMATION)
        self.assertEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)

        fields = {item.field for item in orch.state.missing_information}
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

        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertIsNone(orch.state.geometry)
        self.assertIsNone(orch.state.structure)
        self.assertIsNone(orch.state.analysis)
        self.assertIsNone(orch.state.topology)
        self.assertIsNone(orch.state.registration)

    def test_rejected_request_never_reaches_geometry_or_analysis(self) -> None:
        orch = Orchestrator(fixtures=load_integration_fixtures())
        orch.ingest_user_request(REJECT_MESSAGE)
        self.assertEqual(orch.state.interaction_decision, InteractionDecision.REJECT_OR_ESCALATE)
        self.assertEqual(orch.state.stage, WorkflowStage.REJECTED)
        self.assertEqual(orch.state.safety_status, SafetyStatus.REJECTED)
        self.assertTrue(orch.state.reject_reason)

        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.REJECTED)
        self.assertIsNone(orch.state.geometry)
        self.assertIsNone(orch.state.structure)
        self.assertIsNone(orch.state.analysis)
        self.assertIsNone(orch.state.topology)
        self.assertIsNone(orch.state.cad)
        self.assertIsNone(orch.state.verification)
        self.assertNotIn("Yujie fixture", [event.component for event in orch.trace])
        self.assertNotIn("Shohom fixture", [event.component for event in orch.trace])

    def test_requirement_geometry_mismatch_blocks_progression(self) -> None:
        fixtures = load_integration_fixtures()
        fixtures.geometry.payload_object.bottle_diameter_mm = 99.0
        orch = Orchestrator(fixtures=fixtures)
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("payload diameter_mm", orch.state.contract_error)
        self.assertIsNone(orch.state.structure)
        self.assertIsNone(orch.state.analysis)
        self.assertEqual(orch.state.safety_status, SafetyStatus.NEEDS_REVIEW)

    def test_incompatible_coordinate_frames_block_progression(self) -> None:
        fixtures = load_integration_fixtures()
        fixtures.geometry.coordinate_frame = "some_other_frame"
        orch = Orchestrator(fixtures=fixtures)
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("incompatible coordinate frames", orch.state.contract_error)
        self.assertIsNone(orch.state.structure)
        self.assertEqual(orch.state.safety_status, SafetyStatus.NEEDS_REVIEW)

    def test_missing_registration_blocks_external_geometry(self) -> None:
        fixtures = load_integration_fixtures()
        fixtures.registration = None
        orch = Orchestrator(fixtures=fixtures)
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("RegistrationOutput", orch.state.contract_error)
        self.assertIsNone(orch.state.structure)
        self.assertIsNone(orch.state.analysis)

    def test_geometry_mass_omitted_uses_requirements_with_provenance(self) -> None:
        fixtures = load_integration_fixtures()
        fixtures.geometry.payload_object.filled_mass_kg = None
        orch = Orchestrator(fixtures=fixtures)
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertIsNotNone(orch.state.resolved_payload_mass)
        self.assertEqual(
            orch.state.resolved_payload_mass.provenance,
            MassProvenance.USER_REQUIREMENTS,
        )
        self.assertAlmostEqual(orch.state.resolved_payload_mass.mass_kg, PAYLOAD_MASS_KG)
        self.assertIn("user_requirements", orch.state.resolved_payload_mass.notes)
        force = orch.state.structure.load_cases[0].force_N[2]
        self.assertAlmostEqual(force, -G * PAYLOAD_MASS_KG, places=5)

    def test_conflicting_payload_mass_is_integration_error(self) -> None:
        fixtures = load_integration_fixtures()
        fixtures.geometry.payload_object.filled_mass_kg = 9.9
        orch = Orchestrator(fixtures=fixtures)
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.run()
        self.assertNotEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("filled_mass_kg", orch.state.contract_error)
        self.assertIsNone(orch.state.structure)
        self.assertEqual(orch.state.safety_status, SafetyStatus.NEEDS_REVIEW)

    def test_analysis_fixture_wrong_load_case_is_blocked(self) -> None:
        fixtures = load_integration_fixtures()
        fixtures.analysis.load_case_id = "impact_drop"
        orch = Orchestrator(fixtures=fixtures)
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.run()
        self.assertEqual(orch.state.stage, WorkflowStage.ANALYSIS)
        self.assertIsNone(orch.state.analysis)
        self.assertIsNone(orch.state.topology)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("load_case_id", orch.state.contract_error)

    def test_matching_analysis_fixture_is_accepted(self) -> None:
        orch = _happy_orchestrator()
        self.assertEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertEqual(orch.state.analysis.load_case_id, orch.state.structure.load_cases[0].load_case_id)
        self.assertTrue(
            abs(orch.state.analysis.load_force_N[2] - orch.state.structure.load_cases[0].force_N[2])
            <= 1e-3
        )
        self.assertTrue(orch.state.analysis.is_mock)
        self.assertEqual(orch.state.safety_status, SafetyStatus.UNVERIFIED)

    def test_topology_cannot_run_without_analysis(self) -> None:
        orch = Orchestrator(fixtures=load_integration_fixtures())
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        while orch.state.stage not in (
            WorkflowStage.ANALYSIS,
            WorkflowStage.REQUEST_INFORMATION,
            WorkflowStage.REJECTED,
        ):
            orch.step()
        self.assertEqual(orch.state.stage, WorkflowStage.ANALYSIS)
        orch.state.analysis = None
        orch.state.stage = WorkflowStage.TOPOLOGY_OPTIMIZATION
        orch.step()
        self.assertEqual(orch.state.stage, WorkflowStage.TOPOLOGY_OPTIMIZATION)
        self.assertIsNone(orch.state.topology)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("AnalysisOutput", orch.state.contract_error)

    def test_incomplete_verification_cannot_reach_complete(self) -> None:
        orch = Orchestrator(fixtures=load_integration_fixtures())
        orch.ingest_user_request(load_user_request())
        orch.apply_answers(load_clarifications())
        orch.state.stage = WorkflowStage.VERIFICATION
        orch.step()
        self.assertEqual(orch.state.stage, WorkflowStage.VERIFICATION)
        self.assertIsNotNone(orch.state.verification)
        self.assertFalse(orch.state.verification.artifacts_complete)
        self.assertFalse(orch.state.verification.safety_validated)
        self.assertEqual(orch.state.safety_status, SafetyStatus.NEEDS_REVIEW)

    def test_mock_tools_cannot_mark_design_verified(self) -> None:
        orch = _happy_orchestrator()
        self.assertEqual(orch.state.safety_status, SafetyStatus.UNVERIFIED)
        self.assertFalse(orch.state.analysis.is_safety_validation)
        self.assertFalse(orch.state.verification.safety_validated)
        self.assertTrue(orch.state.analysis.is_mock)
        self.assertTrue(orch.state.topology.is_mock)
        self.assertTrue(orch.state.geometry.is_mock)
        self.assertNotEqual(orch.state.safety_status.value, "verified")
        self.assertNotEqual(orch.state.safety_status.value, "safe")

    def test_final_state_round_trips_through_json(self) -> None:
        orch = _happy_orchestrator()
        payload = orch.state.model_dump(mode="json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final_state.json"
            path.write_text(json.dumps(payload, indent=2))
            loaded = json.loads(path.read_text())
        restored = type(orch.state).model_validate(loaded)
        self.assertEqual(restored, orch.state)
        self.assertEqual(restored.stage, WorkflowStage.COMPLETE)
        self.assertEqual(restored.structure.nodes[0].id, "mount_upper")
        self.assertTrue(restored.verification.artifacts_complete)


if __name__ == "__main__":
    unittest.main()
