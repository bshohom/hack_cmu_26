"""Presentation-only phase mapping. Does not change backend stages."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from schemas import WorkflowStage
from ui_flow import (
    WORKSPACE_TAB_DESIGN,
    WORKSPACE_TAB_OPTIMIZATION,
    WORKSPACE_TAB_SCENE,
    action_spec,
    answers_look_complete,
    need_details_title,
    reconstructed_scene_status,
    stepper_states,
    ui_phase,
    workspace_tab_after_event,
)


def _state(**over):
    base = dict(
        stage=WorkflowStage.REQUIREMENTS,
        requirements=None,
        clarifications=[],
        geometry=None,
        structure=None,
        topology=None,
        verification=None,
        cad=None,
        feasibility=None,
        candidate_fit=None,
        contract_error=None,
        reject_reason=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class UiPhaseTests(unittest.TestCase):
    def test_empty_session_is_describe_start(self) -> None:
        state = _state()
        self.assertEqual(ui_phase(state), "Describe")
        spec = action_spec(state, {})
        self.assertEqual(spec.button, "Start design")
        self.assertEqual(spec.kind, "start")

    def test_missing_scene_fields_are_capture(self) -> None:
        q = SimpleNamespace(field="desk_thickness_mm")
        state = _state(stage=WorkflowStage.REQUEST_INFORMATION, clarifications=[q])
        self.assertEqual(ui_phase(state, {}), "Capture")
        spec = action_spec(state, {})
        self.assertEqual(spec.button, "Continue")
        self.assertEqual(spec.title, "Need 1 more detail")
        self.assertNotIn("not reconstructed", spec.subtitle.lower())

    def test_complete_answers_ready_to_optimize(self) -> None:
        q = SimpleNamespace(field="desk_thickness_mm")
        state = _state(
            stage=WorkflowStage.REQUEST_INFORMATION,
            clarifications=[q],
            requirements=object(),
        )
        answers = {"desk_thickness_mm": 20.0}
        self.assertTrue(answers_look_complete(state, answers))
        self.assertEqual(ui_phase(state, answers), "Design")
        spec = action_spec(state, answers)
        self.assertEqual(spec.button, "Run optimization")

    def test_complete_with_cad_is_export(self) -> None:
        state = _state(
            stage=WorkflowStage.COMPLETE,
            cad=SimpleNamespace(filename="design.stl"),
            topology=SimpleNamespace(
                is_mock=True, converged=True, acceptance={"acceptance_status": "pass", "accepted": True}
            ),
        )
        self.assertEqual(ui_phase(state), "Export")
        spec = action_spec(state, {})
        self.assertEqual(spec.kind, "download")

    def test_unconverged_is_preview_not_failure(self) -> None:
        state = _state(
            stage=WorkflowStage.COMPLETE,
            cad=SimpleNamespace(filename="mock://x"),
            topology=SimpleNamespace(
                is_mock=False,
                converged=False,
                acceptance={"acceptance_status": "unresolved_not_converged"},
            ),
        )
        spec = action_spec(state, {})
        self.assertEqual(spec.result_status, "preview")
        self.assertIn("not converged", spec.subtitle.lower())
        self.assertNotIn("failure", spec.title.lower())

    def test_stepper_marks_current(self) -> None:
        states = list(stepper_states("Design"))
        self.assertEqual([n for n, _ in states], ["Describe", "Capture", "Design", "Verify", "Export"])
        self.assertEqual(dict(states)["Describe"], "done")
        self.assertEqual(dict(states)["Design"], "current")
        self.assertEqual(dict(states)["Export"], "todo")

    def test_stepper_can_mark_current_phase_blocked(self) -> None:
        states = dict(stepper_states("Capture", blocked=True))
        self.assertEqual(states["Capture"], "blocked")
        self.assertEqual(states["Describe"], "done")

    def test_rejected_offers_revise_inputs(self) -> None:
        state = _state(stage=WorkflowStage.REJECTED, reject_reason="out of scope")
        spec = action_spec(state, {})
        self.assertEqual(spec.button, "Revise inputs")
        self.assertEqual(spec.kind, "revise")

    def test_need_details_title_is_terse(self) -> None:
        self.assertEqual(need_details_title(0), "Ready to optimize")
        self.assertEqual(need_details_title(1), "Need 1 more detail")
        self.assertEqual(need_details_title(2), "Need 2 more details")

    def test_workspace_tab_follows_newest_result(self) -> None:
        self.assertEqual(workspace_tab_after_event(reconstruction=True), WORKSPACE_TAB_SCENE)
        self.assertEqual(workspace_tab_after_event(design=True), WORKSPACE_TAB_DESIGN)
        self.assertEqual(
            workspace_tab_after_event(reconstruction=True, design=True, topology=True),
            WORKSPACE_TAB_OPTIMIZATION,
        )

    def test_reconstructed_scene_status_is_short(self) -> None:
        self.assertEqual(reconstructed_scene_status(reconstructed=True), "Scene reconstructed")
        self.assertEqual(reconstructed_scene_status(reconstructed=True, photo_count=8), "Scene reconstructed")
        self.assertEqual(reconstructed_scene_status(), "Add 8 more photos to reconstruct")
        self.assertEqual(reconstructed_scene_status(photo_count=2), "Add 6 more photos to reconstruct")
        self.assertEqual(reconstructed_scene_status(photo_count=7), "Add 1 more photo to reconstruct")
        self.assertEqual(reconstructed_scene_status(photo_count=8), "Scene not reconstructed yet")
        self.assertNotEqual(reconstructed_scene_status(photo_count=14), "Scene reconstructed")


if __name__ == "__main__":
    unittest.main()
