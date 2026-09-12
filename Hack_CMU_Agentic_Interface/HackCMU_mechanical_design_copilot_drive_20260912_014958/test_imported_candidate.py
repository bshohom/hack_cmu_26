"""Imported candidate geometry adapter and geometric fit checks."""

from __future__ import annotations

import unittest

from imported_candidate import (
    CANDIDATE_DIR,
    DIMENSIONS_NAME,
    MESH_NAME,
    PARTICLES_NAME,
    check_candidate_fit,
    load_imported_candidate,
)
from orchestrator import Orchestrator
from schemas import (
    CandidateFitStatus,
    RequirementsUpdate,
    UserRequirements,
    WorkflowStage,
)
from geometry_sources import (
    GEOM_IMPORTED,
    fixtures_for_geometry_mode,
    imported_candidate_for_mode,
)

DEMO_MESSAGE = (
    "I want a cup holder attached to this desk that supports a full 1 L bottle."
)


def _update(diameter_mm: float, desk_mm: float) -> RequirementsUpdate:
    return RequirementsUpdate(
        filled_bottle_mass_kg=1.1,
        bottle_diameter_mm=diameter_mm,
        bottle_height_mm=250.0,
        desk_thickness_mm=desk_mm,
        attachment_method="clamp",
        allowed_contact_region="desk_front_edge",
        max_protrusion_mm=150.0,
        manufacturing_method="3d_print",
        material="PLA",
        max_part_mass_kg=0.3,
    )


def _run(diameter_mm: float, desk_mm: float) -> Orchestrator:
    orch = Orchestrator(
        fixtures=fixtures_for_geometry_mode(GEOM_IMPORTED),
        imported_candidate=load_imported_candidate(),
    )
    orch.ingest_user_request(DEMO_MESSAGE)
    orch.apply_answers(_update(diameter_mm, desk_mm))
    orch.run()
    return orch


class ImportedCandidateAdapterTests(unittest.TestCase):
    def test_uses_exact_artifact_filenames(self) -> None:
        self.assertTrue((CANDIDATE_DIR / DIMENSIONS_NAME).is_file())
        self.assertTrue((CANDIDATE_DIR / MESH_NAME).is_file())
        self.assertTrue((CANDIDATE_DIR / PARTICLES_NAME).is_file())
        self.assertEqual(DIMENSIONS_NAME, "cupholder_dimensions.txt")
        self.assertEqual(MESH_NAME, "cupholder_single_piece_PLA.obj")
        self.assertEqual(PARTICLES_NAME, "cupholder_surface_particles.obj")

    def test_parsed_dimensions_and_mesh_stats(self) -> None:
        candidate = load_imported_candidate()
        self.assertEqual(candidate.inner_diameter_mm, 70.0)
        self.assertEqual(candidate.outer_diameter_mm, 80.0)
        self.assertEqual(candidate.holder_height_mm, 60.0)
        self.assertEqual(candidate.wall_thickness_mm, 5.0)
        self.assertEqual(candidate.desk_gap_mm, 30.0)
        self.assertEqual(candidate.compatible_desk_min_mm, 20.0)
        self.assertEqual(candidate.compatible_desk_max_mm, 35.0)
        self.assertEqual(candidate.clamp_reach_mm, 45.0)
        self.assertEqual(candidate.vertex_count, 38032)
        self.assertEqual(candidate.face_count, 76068)
        self.assertTrue(candidate.watertight)
        self.assertEqual(candidate.connected_components, 1)
        self.assertEqual(candidate.provenance, "external generated concept geometry")
        self.assertFalse(candidate.is_mock)
        self.assertIsNotNone(candidate.bbox_min_mm)
        self.assertIsNotNone(candidate.bbox_max_mm)
        self.assertTrue(candidate.mesh_path.endswith(MESH_NAME))
        self.assertTrue(candidate.particle_path.endswith(PARTICLES_NAME))

    def test_payload_100_desk_20_fails_overall_and_blocks_structure(self) -> None:
        requirements = UserRequirements()
        requirements.object_geometry.bottle_diameter_mm = 100.0
        requirements.environment.desk_thickness_mm = 20.0
        candidate = load_imported_candidate()
        fit = check_candidate_fit(requirements, candidate)
        by_name = {check.name: check for check in fit.checks}
        self.assertEqual(by_name["payload_fit"].status, CandidateFitStatus.FAIL)
        self.assertEqual(by_name["payload_fit"].required_mm, 100.0)
        self.assertEqual(by_name["payload_fit"].available_mm, 70.0)
        self.assertIn("Payload does not fit holder opening", by_name["payload_fit"].message)
        self.assertEqual(by_name["desk_fit"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["desk_fit"].desk_mm, 20.0)
        self.assertEqual(by_name["desk_fit"].supported_range_mm, (20.0, 35.0))
        self.assertFalse(fit.fits)

        orch = _run(100.0, 20.0)
        self.assertIsNotNone(orch.state.candidate_fit)
        self.assertFalse(orch.state.candidate_fit.fits)
        self.assertIsNone(orch.state.structure)
        self.assertIsNone(orch.state.analysis)
        self.assertNotEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertIn("CANDIDATE GEOMETRY REJECTED", orch.state.notes)

    def test_payload_65_desk_25_passes_and_starts_structure(self) -> None:
        requirements = UserRequirements()
        requirements.object_geometry.bottle_diameter_mm = 65.0
        requirements.environment.desk_thickness_mm = 25.0
        requirements.design_envelope.max_protrusion_mm = 150.0
        candidate = load_imported_candidate()
        fit = check_candidate_fit(requirements, candidate)
        by_name = {check.name: check for check in fit.checks}
        self.assertEqual(by_name["payload_fit"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["desk_fit"].status, CandidateFitStatus.PASS)
        self.assertTrue(fit.fits)

        orch = _run(65.0, 25.0)
        self.assertIsNotNone(orch.state.candidate_fit)
        self.assertTrue(orch.state.candidate_fit.fits)
        self.assertIsNotNone(orch.state.structure)
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)

    def test_imported_mode_does_not_load_golden_geometry_fixture(self) -> None:
        fixtures = fixtures_for_geometry_mode(GEOM_IMPORTED)
        self.assertIsNone(fixtures.geometry)
        self.assertIsNone(fixtures.analysis)
        candidate = imported_candidate_for_mode(GEOM_IMPORTED)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.inner_diameter_mm, 70.0)


if __name__ == "__main__":
    unittest.main()


class UnknownFitTests(unittest.TestCase):
    """Missing evidence must read as unknown, never as a successful fit.

    The audit reproduced fits=true with all three checks marked n/a, followed by the
    message "fits the resolved user requirements". Every generated warm start hit this,
    because warmstart.py never populates the four fields the checks read.
    """

    def _candidate(self, **over):
        from schemas import ImportedCandidateGeometry

        data = {"mesh_path": "/tmp/x.stl", "candidate_name": "generated", "task": "generated"}
        data.update(over)
        return ImportedCandidateGeometry(**data)

    def test_all_checks_na_is_not_a_fit(self) -> None:
        result = check_candidate_fit(UserRequirements(), self._candidate())
        self.assertTrue(all(c.status == CandidateFitStatus.NA for c in result.checks))
        self.assertFalse(result.fits)
        self.assertIn("UNKNOWN", result.message)
        self.assertNotIn("fits the resolved user requirements", result.message)

    def test_unknown_checks_produce_answerable_questions(self) -> None:
        from imported_candidate import candidate_fit_questions

        result = check_candidate_fit(UserRequirements(), self._candidate())
        fields = {q.field for q in candidate_fit_questions(result)}
        self.assertEqual(
            fields, {"bottle_diameter_mm", "desk_thickness_mm", "max_protrusion_mm"}
        )

    def test_fully_checked_candidate_still_passes(self) -> None:
        req = UserRequirements()
        req.object_geometry.bottle_diameter_mm = 70.0
        req.environment.desk_thickness_mm = 25.0
        req.design_envelope.max_protrusion_mm = 120.0
        candidate = self._candidate(
            inner_diameter_mm=75.0,
            compatible_desk_min_mm=18.0,
            compatible_desk_max_mm=40.0,
            clamp_reach_mm=100.0,
        )
        result = check_candidate_fit(req, candidate)
        self.assertTrue(result.fits, result.message)
        # and it does not overclaim what was actually verified
        self.assertIn("not checked", result.message)
