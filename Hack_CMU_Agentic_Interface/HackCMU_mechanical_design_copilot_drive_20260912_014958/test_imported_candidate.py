"""Imported candidate geometry adapter and geometric fit checks."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from imported_candidate import (
    CANDIDATE_DIR,
    DIMENSIONS_NAME,
    MESH_NAME,
    PARTICLES_NAME,
    check_candidate_fit,
    fit_family_for,
    load_candidate,
    load_imported_candidate,
)
from orchestrator import Orchestrator
from schemas import (
    CandidateFitStatus,
    RequirementsUpdate,
    TopologySolverOptions,
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

        data = {"mesh_path": "/tmp/x.stl", "candidate_name": "cupholder", "task": "cupholder"}
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

    def test_generated_warm_start_does_not_require_bottle_or_desk(self) -> None:
        from imported_candidate import candidate_fit_questions
        from schemas import ImportedCandidateGeometry

        req = UserRequirements()
        req.task_kind = "bed_handle"
        req.design_envelope.max_protrusion_mm = 50.0
        candidate = ImportedCandidateGeometry(
            mesh_path="/tmp/x.stl",
            candidate_name="grok_warmstart",
            task="generated",
            watertight=True,
            bbox_min_mm=(0.0, -10.0, 0.0),
            bbox_max_mm=(40.0, 10.0, 30.0),
        )
        self.assertEqual(fit_family_for(candidate), "generated")
        result = check_candidate_fit(req, candidate)
        self.assertTrue(result.fits, result.message)
        fields = {q.field for q in candidate_fit_questions(result, family="generated")}
        self.assertNotIn("bottle_diameter_mm", fields)
        self.assertNotIn("desk_thickness_mm", fields)

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


class TaskAwareFitTests(unittest.TestCase):
    """Cupholder, hook, and shelf must not share a single inner-diameter check."""

    def _hook_requirements(self) -> UserRequirements:
        req = UserRequirements()
        req.object_geometry.kind = "strap"
        req.object_geometry.bottle_diameter_mm = 30.0
        req.payload.filled_mass_kg = 5.0
        req.environment.desk_thickness_mm = 20.0
        req.design_envelope.max_protrusion_mm = 110.0
        return req

    def test_desk_hook_does_not_use_inner_diameter(self) -> None:
        candidate = load_candidate("desk_bag_hook")
        self.assertEqual(fit_family_for(candidate), "desk_bag_hook")
        self.assertIsNone(candidate.inner_diameter_mm)
        self.assertEqual(candidate.dimensions.get("hook_opening"), 30.0)
        fit = check_candidate_fit(self._hook_requirements(), candidate)
        by_name = {c.name: c for c in fit.checks}
        self.assertNotIn("inner diameter", fit.message.lower())
        self.assertNotIn("holder opening", by_name["payload_fit"].message.lower())
        self.assertEqual(by_name["payload_fit"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["payload_fit"].required_mm, 30.0)
        self.assertEqual(by_name["payload_fit"].available_mm, 30.0)
        self.assertEqual(by_name["desk_fit"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["desk_fit"].desk_mm, 20.0)
        self.assertEqual(by_name["desk_fit"].supported_range_mm, (18.0, 22.0))
        self.assertEqual(by_name["envelope_fit"].status, CandidateFitStatus.PASS)
        self.assertAlmostEqual(by_name["envelope_fit"].available_mm, 106.85, places=1)
        self.assertEqual(by_name["load_rating"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["load_rating"].required_mm, 5.0)
        self.assertEqual(by_name["load_rating"].available_mm, 5.0)
        self.assertTrue(fit.fits, fit.message)
        from imported_candidate import candidate_fit_questions

        blob = " ".join(q.question.lower() for q in candidate_fit_questions(fit, family="desk_bag_hook"))
        self.assertNotIn("inner diameter", blob)

    def test_desk_hook_strap_wider_than_opening_fails(self) -> None:
        req = self._hook_requirements()
        req.object_geometry.bottle_diameter_mm = 40.0
        fit = check_candidate_fit(req, load_candidate("desk_bag_hook"))
        by_name = {c.name: c for c in fit.checks}
        self.assertEqual(by_name["payload_fit"].status, CandidateFitStatus.FAIL)
        self.assertFalse(fit.fits)

    def test_stapler_shelf_uses_platform_not_cup(self) -> None:
        candidate = load_candidate("stapler_shelf")
        self.assertEqual(fit_family_for(candidate), "stapler_shelf")
        self.assertIsNone(candidate.inner_diameter_mm)
        req = UserRequirements()
        req.object_geometry.kind = "box"
        req.object_geometry.bottle_diameter_mm = 60.0
        req.payload.filled_mass_kg = 0.5
        req.environment.desk_thickness_mm = 20.0
        req.design_envelope.max_protrusion_mm = 120.0
        fit = check_candidate_fit(req, candidate)
        names = {c.name for c in fit.checks}
        self.assertEqual(names, {"payload_fit", "envelope_fit"})
        self.assertNotIn("desk_fit", names)
        by_name = {c.name: c for c in fit.checks}
        self.assertEqual(by_name["payload_fit"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["payload_fit"].available_mm, 70.0)
        self.assertEqual(by_name["envelope_fit"].status, CandidateFitStatus.PASS)
        self.assertEqual(by_name["envelope_fit"].available_mm, 120.0)
        self.assertTrue(fit.fits, fit.message)
        self.assertNotIn("inner diameter", fit.message.lower())
        self.assertNotIn("holder opening", by_name["payload_fit"].message.lower())

    def test_stapler_shelf_footprint_too_wide_fails(self) -> None:
        candidate = load_candidate("stapler_shelf")
        req = UserRequirements()
        req.object_geometry.bottle_diameter_mm = 80.0
        req.design_envelope.max_protrusion_mm = 120.0
        fit = check_candidate_fit(req, candidate)
        self.assertEqual({c.name: c.status for c in fit.checks}["payload_fit"], CandidateFitStatus.FAIL)
        self.assertFalse(fit.fits)

    def test_cupholder_still_uses_inner_diameter(self) -> None:
        candidate = load_imported_candidate()
        self.assertEqual(fit_family_for(candidate), "cupholder")
        req = UserRequirements()
        req.object_geometry.bottle_diameter_mm = 65.0
        req.environment.desk_thickness_mm = 25.0
        req.design_envelope.max_protrusion_mm = 150.0
        fit = check_candidate_fit(req, candidate)
        by_name = {c.name: c for c in fit.checks}
        self.assertEqual(by_name["payload_fit"].available_mm, 70.0)
        self.assertIn("holder opening", by_name["payload_fit"].message.lower())
        self.assertTrue(fit.fits)

    def test_desk_hook_live_case_reaches_structure(self) -> None:
        orch = Orchestrator(
            fixtures=fixtures_for_geometry_mode(GEOM_IMPORTED),
            imported_candidate=load_candidate("desk_bag_hook"),
        )
        orch.ingest_user_request(
            "I want a hook clamped under my desk edge to hang a 5 kg bag about 100 mm out from the edge."
        )
        orch.apply_answers(
            RequirementsUpdate(
                filled_bottle_mass_kg=5.0,
                bottle_diameter_mm=30.0,
                bottle_height_mm=300.0,
                desk_thickness_mm=20.0,
                attachment_method="clamp",
                allowed_contact_region="desk_front_edge",
                max_protrusion_mm=110.0,
                manufacturing_method="3d_print",
                material="PLA",
                max_part_mass_kg=0.3,
            )
        )
        orch.run()
        self.assertIsNotNone(orch.state.candidate_fit)
        self.assertTrue(orch.state.candidate_fit.fits, orch.state.candidate_fit.message)
        self.assertNotIn("inner diameter", orch.state.candidate_fit.message.lower())
        self.assertIsNotNone(orch.state.structure)
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        loads = orch.state.structure.load_regions if orch.state.structure else []
        self.assertTrue(any(r.name == "strap_seat" for r in loads))
        self.assertFalse(any(r.name == "cup_cavity" for r in loads))


@unittest.skipUnless(os.environ.get("TO_AGENT_LIVE") == "1", "live to_agent smoke")
class DeskHookLiveTopologyTests(unittest.TestCase):
    """2-iter live SIMP: Candidate Fit → STRUCTURE → real to_agent → CAD STL."""

    def test_desk_hook_two_iter_smoke(self) -> None:
        orch = Orchestrator(
            fixtures=fixtures_for_geometry_mode(GEOM_IMPORTED, topology_live=True),
            imported_candidate=load_candidate("desk_bag_hook"),
            topology_options=TopologySolverOptions(
                element_size_mm=8.0,
                max_iters=2,
                time_budget_s=180.0,
                device="cpu",
            ),
        )
        orch.ingest_user_request(
            "I want a hook clamped under my desk edge to hang a 5 kg bag about 100 mm out from the edge."
        )
        orch.apply_answers(
            RequirementsUpdate(
                filled_bottle_mass_kg=5.0,
                bottle_diameter_mm=30.0,
                bottle_height_mm=300.0,
                desk_thickness_mm=20.0,
                attachment_method="clamp",
                allowed_contact_region="desk_front_edge",
                max_protrusion_mm=110.0,
                manufacturing_method="3d_print",
                material="PLA",
                max_part_mass_kg=0.3,
            )
        )
        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "live"}):
            orch.run()
        self.assertTrue(orch.state.candidate_fit and orch.state.candidate_fit.fits)
        self.assertTrue(any(r.name == "strap_seat" for r in orch.state.structure.load_regions))
        self.assertFalse(any(r.name == "cup_cavity" for r in orch.state.structure.load_regions))
        topo = orch.state.topology
        self.assertIsNotNone(topo)
        self.assertFalse(topo.is_mock)
        self.assertTrue((topo.optimized_geometry_ref or "").endswith("design.stl"))
        self.assertTrue(topo.optimized_geometry_ref and os.path.isfile(topo.optimized_geometry_ref))
        self.assertEqual(topo.acceptance.get("acceptance_status"), "unresolved_not_converged")
        self.assertIsNotNone(orch.state.cad)
        self.assertFalse(orch.state.cad.is_mock)
        self.assertEqual(orch.state.cad.filename, topo.optimized_geometry_ref)
        self.assertIn(
            orch.state.stage,
            {WorkflowStage.COMPLETE, WorkflowStage.VERIFICATION_FAILED},
        )
