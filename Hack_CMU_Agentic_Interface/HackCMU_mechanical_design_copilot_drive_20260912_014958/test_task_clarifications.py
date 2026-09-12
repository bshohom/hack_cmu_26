"""Task-relevant clarifications. Cup-holder fields are not a global default."""

from __future__ import annotations

import unittest

from agents.interaction import (
    InteractionAgent,
    TASK_BED_HANDLE,
    TASK_CUPHOLDER,
    TASK_DESK_HOOK,
    TASK_GENERIC,
    TASK_WALL_SHELF,
    classify_design_task,
    classify_hazard,
    clarification_specs_for_task,
)
from orchestrator import Orchestrator
from schemas import IntegrationFixtures, InteractionDecision, RequirementsUpdate, WorkflowStage
from ui_flow import action_spec, backend_missing_fields


class TaskClarificationTests(unittest.TestCase):
    def test_classifies_cross_task_requests(self) -> None:
        self.assertEqual(classify_design_task("Design a cup holder for my desk"), TASK_CUPHOLDER)
        self.assertEqual(classify_design_task("Design a hook for a bag on my desk"), TASK_DESK_HOOK)
        self.assertEqual(
            classify_design_task("Design a handle to help a person get up from bed"),
            TASK_BED_HANDLE,
        )
        self.assertEqual(
            classify_design_task("Design a wall-mounted shelf for a router"),
            TASK_WALL_SHELF,
        )
        self.assertEqual(
            classify_design_task("Design a phone stand for my nightstand"),
            TASK_GENERIC,
        )

    def test_bed_handle_is_not_auto_rejected(self) -> None:
        signal = classify_hazard("Design a handle to help a person get up from bed")
        self.assertNotEqual(signal.verdict, "refuse")

    def test_cup_holder_still_asks_bottle_and_desk(self) -> None:
        fields = {spec["field"] for spec in clarification_specs_for_task(TASK_CUPHOLDER)}
        for required in (
            "filled_bottle_mass_kg",
            "bottle_diameter_mm",
            "desk_thickness_mm",
        ):
            self.assertIn(required, fields)

    def test_desk_hook_asks_load_and_desk_not_bottle_size(self) -> None:
        fields = {spec["field"] for spec in clarification_specs_for_task(TASK_DESK_HOOK)}
        self.assertIn("filled_bottle_mass_kg", fields)
        self.assertIn("desk_thickness_mm", fields)
        self.assertNotIn("bottle_diameter_mm", fields)

    def test_bed_handle_does_not_ask_bottle_cup_or_desk(self) -> None:
        text = "Design a handle to help a person get up from bed"
        orch = Orchestrator()
        orch.ingest_user_request(text)
        self.assertEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertEqual(orch.state.interaction_decision, InteractionDecision.REQUEST_INFORMATION)
        fields = {item.field for item in orch.state.missing_information}
        blob = " ".join(
            f"{q.field} {q.question}" for q in orch.state.clarifications
        ).lower()
        self.assertNotIn("filled_bottle_mass_kg", fields)
        self.assertNotIn("bottle_diameter_mm", fields)
        self.assertNotIn("desk_thickness_mm", fields)
        self.assertNotIn("bottle", blob)
        self.assertNotIn("cup holder", blob)
        self.assertNotIn("desk thickness", blob)
        self.assertTrue({"supported_load_kg", "attachment_structure"} & fields)

    def test_wall_shelf_does_not_ask_desk_or_bottle(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request("Design a wall-mounted shelf for a router")
        fields = {item.field for item in orch.state.missing_information}
        blob = " ".join(q.question.lower() for q in orch.state.clarifications)
        self.assertNotIn("desk_thickness_mm", fields)
        self.assertNotIn("filled_bottle_mass_kg", fields)
        self.assertNotIn("bottle_diameter_mm", fields)
        self.assertNotIn("desk", blob)
        self.assertNotIn("bottle", blob)
        self.assertIn("supported_load_kg", fields)

    def test_agent_assess_matches_orchestrator(self) -> None:
        result = InteractionAgent().assess("Design a handle to help a person get up from bed")
        fields = {q.field for q in result.questions}
        self.assertNotIn("desk_thickness_mm", fields)
        self.assertNotIn("bottle_diameter_mm", fields)
        self.assertNotIn("filled_bottle_mass_kg", fields)

    def test_bed_handle_answers_map_and_geometry_does_not_bounce(self) -> None:
        grok_called = {"n": 0}

        def _gen(req):
            grok_called["n"] += 1
            return None

        orch = Orchestrator(
            fixtures=IntegrationFixtures(
                registration=None, geometry=None, analysis=None, topology=None, cad=None
            ),
            warm_start_generator=_gen,
        )
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(
            RequirementsUpdate(
                supported_load_kg=80.0,
                attachment_structure="wall",
                handle_location="bedside",
                mounting_region="wall",
                drilling_allowed=False,
                required_reach_mm=50.0,
                manufacturing_method="3d_print",
                attachment_method="adhesive",
            )
        )
        req = orch.state.requirements
        self.assertEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertEqual(req.payload.filled_mass_kg, 80.0)
        self.assertIsNone(req.design_envelope.max_protrusion_mm)
        self.assertEqual(req.attachment.method, "adhesive")
        self.assertEqual(req.attachment.allowed_contact_region, "wall")
        self.assertEqual(req.manufacturing.method, "3d_print")
        self.assertIsNone(req.object_geometry.bottle_diameter_mm)
        self.assertIsNone(req.environment.desk_thickness_mm)
        self.assertEqual(orch._requirements_gaps(), [])
        orch._handle_geometry()
        self.assertEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertIsNotNone(orch.state.geometry)
        self.assertIsNone(orch.state.topology)
        self.assertTrue(orch.warm_start_error)
        self.assertEqual(grok_called["n"], 1)
        self.assertEqual(req.task_answers.get("required_reach_mm"), 50.0)

    def test_required_reach_and_max_protrusion_stay_separate(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(
            RequirementsUpdate(
                supported_load_kg=80.0,
                attachment_structure="wall",
                handle_location="bedside",
                mounting_region="wall",
                drilling_allowed=False,
                required_reach_mm=50.0,
                max_protrusion_mm=150.0,
                manufacturing_method="3d_print",
                attachment_method="adhesive",
            )
        )
        req = orch.state.requirements
        assert req is not None
        self.assertEqual(req.task_answers.get("required_reach_mm"), 50.0)
        self.assertEqual(req.design_envelope.max_protrusion_mm, 150.0)
        self.assertEqual(orch.interaction._field_value(req, "required_reach_mm"), 50.0)
        self.assertEqual(orch.interaction._field_value(req, "max_protrusion_mm"), 150.0)

    def test_listed_bed_handle_answers_leave_handle_and_drilling_if_unset(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(
            RequirementsUpdate(
                supported_load_kg=80.0,
                attachment_structure="wall",
                required_reach_mm=50.0,
                manufacturing_method="3d_print",
                attachment_method="adhesive",
                mounting_region="wall",
            )
        )
        req = orch.state.requirements
        assert req is not None
        self.assertEqual(req.payload.filled_mass_kg, 80.0)
        self.assertIsNone(req.design_envelope.max_protrusion_mm)
        self.assertEqual(req.attachment.method, "adhesive")
        self.assertEqual(req.attachment.allowed_contact_region, "wall")
        self.assertEqual(req.manufacturing.method, "3d_print")
        gaps = set(orch._requirements_gaps())
        self.assertNotIn("bottle_diameter_mm", gaps)
        self.assertNotIn("desk_thickness_mm", gaps)
        self.assertTrue({"handle_location", "drilling_allowed"} <= gaps)
        spec = action_spec(orch.state, {
            "supported_load_kg": 80.0,
            "attachment_structure": "wall",
            "required_reach_mm": 50.0,
            "manufacturing_method": "3d_print",
            "attachment_method": "adhesive",
            "mounting_region": "wall",
        })
        self.assertNotEqual(spec.title, "Ready to optimize")

    def test_unknown_task_asks_generic_engineering_not_cupholder(self) -> None:
        fields = {spec["field"] for spec in clarification_specs_for_task(TASK_GENERIC)}
        self.assertIn("supported_load_kg", fields)
        self.assertIn("attachment_method", fields)
        self.assertIn("required_reach_mm", fields)
        self.assertIn("max_protrusion_mm", fields)
        self.assertNotIn("bottle_diameter_mm", fields)
        self.assertNotIn("desk_thickness_mm", fields)
        self.assertNotIn("filled_bottle_mass_kg", fields)

    def test_empty_clarifications_do_not_look_ready_when_backend_gaps_exist(self) -> None:
        orch = Orchestrator()
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.state.clarifications = []
        orch.state.stage = WorkflowStage.REQUEST_INFORMATION
        spec = action_spec(orch.state, {})
        self.assertNotEqual(spec.title, "Ready to optimize")
        self.assertTrue(backend_missing_fields(orch.state, {}))


class TaskClarificationAppTests(unittest.TestCase):
    def test_bed_handle_ui_does_not_ask_bottle_or_desk(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=180)
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        at.text_area(key="request_text").set_value(
            "Design a handle to help a person get up from bed"
        ).run()
        for button in at.button:
            if button.label == "Start design":
                button.click().run()
                break
        self.assertFalse(at.exception, msg=at.exception)
        page = " ".join(str(getattr(block, "value", "")) for block in at.markdown).lower()
        labels = " ".join(str(getattr(w, "label", "")) for w in list(at.number_input) + list(at.pills) + list(at.text_input)).lower()
        blob = f"{page} {labels}"
        self.assertNotIn("bottle", blob)
        self.assertNotIn("cup holder", blob)
        self.assertNotIn("desk thickness", blob)
        fields = {item.field for item in at.session_state.orch.state.clarifications}
        self.assertNotIn("desk_thickness_mm", fields)
        self.assertNotIn("bottle_diameter_mm", fields)
        self.assertNotIn("filled_bottle_mass_kg", fields)
