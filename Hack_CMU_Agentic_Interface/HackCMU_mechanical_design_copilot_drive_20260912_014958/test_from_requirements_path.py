"""Deterministic From-requirements path reaches TO without Grok."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agents.interaction import TASK_CUPHOLDER, TASK_DESK_HOOK, TASK_GENERIC, TASK_WALL_SHELF
from geometry_sources import (
    GEOM_FROM_REQUIREMENTS,
    GEOM_GENERATED,
    fixtures_for_geometry_mode,
    uses_warm_start_generator,
)
from orchestrator import Orchestrator
from schemas import IntegrationFixtures, RequirementsUpdate, WorkflowStage
from ui_retry import PROGRESS_FROM_REQUIREMENTS, progress_caption


def _run(message: str, update: RequirementsUpdate, *, generator=None) -> tuple[Orchestrator, str]:
    grok = {"n": 0}

    def _forbidden(req):
        grok["n"] += 1
        raise AssertionError("generate_warm_start must not run on the From requirements path")

    orch = Orchestrator(
        fixtures=fixtures_for_geometry_mode(GEOM_FROM_REQUIREMENTS, topology_live=True),
        warm_start_generator=generator,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        orch.ingest_user_request(message)
        orch.apply_answers(update)
        orch.run()
    log = buf.getvalue()
    orch._grok_calls = grok["n"]  # type: ignore[attr-defined]
    return orch, log


class FromRequirementsPathTests(unittest.TestCase):
    def test_generator_is_off_for_from_requirements_mode(self) -> None:
        self.assertFalse(uses_warm_start_generator(GEOM_FROM_REQUIREMENTS))
        self.assertTrue(uses_warm_start_generator(GEOM_GENERATED))
        self.assertEqual(progress_caption("from_requirements"), PROGRESS_FROM_REQUIREMENTS)
        self.assertNotIn("mock", progress_caption("from_requirements").lower())

    def test_cup_holder_reaches_to_without_grok(self) -> None:
        orch, log = _run(
            "I want a cup holder attached to this desk",
            RequirementsUpdate(
                filled_bottle_mass_kg=1.1,
                bottle_diameter_mm=90.0,
                desk_thickness_mm=24.0,
                attachment_method="clamp",
                allowed_contact_region="desk_front_edge",
                max_protrusion_mm=140.0,
                manufacturing_method="3d_print",
            ),
        )
        self._assert_no_grok_to_reached(orch, log)
        self.assertEqual(orch.state.requirements.task_kind, TASK_CUPHOLDER)
        self.assertEqual(orch.state.geometry.payload_object.kind, "cylinder")

    def test_desk_hook_reaches_to_without_grok(self) -> None:
        orch, log = _run(
            "Design a hook for a bag on my desk",
            RequirementsUpdate(
                filled_bottle_mass_kg=5.0,
                desk_thickness_mm=20.0,
                attachment_method="clamp",
                allowed_contact_region="desk_front_edge",
                max_protrusion_mm=110.0,
                manufacturing_method="3d_print",
            ),
        )
        self._assert_no_grok_to_reached(orch, log)
        self.assertEqual(orch.state.requirements.task_kind, TASK_DESK_HOOK)
        self.assertEqual(orch.state.geometry.payload_object.kind, "strap")

    def test_wall_shelf_and_generic_support_load_reach_to_without_grok(self) -> None:
        shelf, shelf_log = _run(
            "Design a wall-mounted shelf for a router",
            RequirementsUpdate(
                supported_load_kg=3.0,
                payload_size_mm=200.0,
                attachment_method="screws",
                mounting_region="wall",
                wall_clearance_mm=80.0,
                max_protrusion_mm=180.0,
                manufacturing_method="3d_print",
            ),
        )
        self._assert_no_grok_to_reached(shelf, shelf_log)
        self.assertEqual(shelf.state.requirements.task_kind, TASK_WALL_SHELF)

        generic, generic_log = _run(
            "Design a phone stand for my nightstand",
            RequirementsUpdate(
                supported_load_kg=0.4,
                attachment_structure="nightstand",
                attachment_method="adhesive",
                required_reach_mm=40.0,
                max_protrusion_mm=90.0,
                manufacturing_method="3d_print",
            ),
        )
        self._assert_no_grok_to_reached(generic, generic_log)
        self.assertEqual(generic.state.requirements.task_kind, TASK_GENERIC)
        self.assertEqual(generic.state.geometry.payload_object.kind, "object")

    def test_same_requirements_are_reproducible(self) -> None:
        update = RequirementsUpdate(
            filled_bottle_mass_kg=5.0,
            desk_thickness_mm=20.0,
            attachment_method="clamp",
            allowed_contact_region="desk_front_edge",
            max_protrusion_mm=110.0,
            manufacturing_method="3d_print",
        )
        a, _ = _run("Design a hook for a bag on my desk", update)
        b, _ = _run("Design a hook for a bag on my desk", update)
        self.assertEqual(a.state.geometry.part.length_mm, b.state.geometry.part.length_mm)
        self.assertEqual(a.state.geometry.part.width_mm, b.state.geometry.part.width_mm)
        self.assertEqual(
            a.state.structure.load_regions[0].position_mm,
            b.state.structure.load_regions[0].position_mm,
        )
        self.assertEqual(
            [n.position_mm for n in a.state.structure.nodes],
            [n.position_mm for n in b.state.structure.nodes],
        )

    def test_forbidden_generator_is_never_invoked(self) -> None:
        def _gen(req):
            raise AssertionError("warm_start_generator must stay unused")

        orch, log = _run(
            "I want a cup holder attached to this desk",
            RequirementsUpdate(
                filled_bottle_mass_kg=1.1,
                bottle_diameter_mm=90.0,
                desk_thickness_mm=24.0,
                attachment_method="clamp",
                allowed_contact_region="desk_front_edge",
                max_protrusion_mm=140.0,
                manufacturing_method="3d_print",
            ),
            generator=None,
        )
        self.assertIsNone(orch.warm_start_generator)
        with patch("tools.warmstart.generate_warm_start") as generate:
            orch2 = Orchestrator(
                fixtures=IntegrationFixtures(
                    registration=None, geometry=None, analysis=None, topology=None, cad=None
                ),
                warm_start_generator=None,
            )
            orch2.ingest_user_request("Design a hook for a bag on my desk")
            orch2.apply_answers(
                RequirementsUpdate(
                    filled_bottle_mass_kg=5.0,
                    desk_thickness_mm=20.0,
                    attachment_method="clamp",
                    allowed_contact_region="desk_front_edge",
                    max_protrusion_mm=110.0,
                    manufacturing_method="3d_print",
                )
            )
            orch2.run()
            generate.assert_not_called()
        self.assertNotIn("GENERATION_STARTED", log)

    def _assert_no_grok_to_reached(self, orch: Orchestrator, log: str) -> None:
        self.assertIsNone(orch.warm_start_generator)
        self.assertIsNone(orch.imported_candidate)
        self.assertIsNone(orch.state.imported_candidate)
        self.assertIsNotNone(orch.state.geometry)
        self.assertFalse(orch.state.geometry.is_mock)
        self.assertIsNotNone(orch.state.structure)
        self.assertTrue(orch.state.structure.attachment_regions)
        self.assertTrue(orch.state.structure.load_regions)
        self.assertNotIn("[GROK] GENERATION_STARTED", log)
        self.assertNotIn("GROK_CALL_SITE_REACHED", log)
        self.assertIn("FROM_REQUIREMENTS", log)
        self.assertIn("[RUN] TO_PROBLEM_BUILD_STARTED", log)
        self.assertIn("[RUN] TO_SOLVER_STARTED", log)
        self.assertNotEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertIn(
            orch.state.stage,
            {
                WorkflowStage.TOPOLOGY_OPTIMIZATION,
                WorkflowStage.VERIFICATION,
                WorkflowStage.COMPLETE,
                WorkflowStage.TOPOLOGY_FAILED,
            },
        )


if __name__ == "__main__":
    unittest.main()
