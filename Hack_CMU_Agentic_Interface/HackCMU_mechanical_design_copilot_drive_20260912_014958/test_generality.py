"""Cross-task generality: clarification, readiness, and warm-start prompts."""

from __future__ import annotations

import unittest

from agents.interaction import (
    TASK_BED_HANDLE,
    TASK_CUPHOLDER,
    TASK_DESK_HOOK,
    TASK_GENERIC,
    TASK_WALL_SHELF,
    classify_design_task,
    clarification_specs_for_task,
    missing_requirement_fields,
)
from imported_candidate import fit_family_for
from schemas import ImportedCandidateGeometry, UserRequirements
from tools.warmstart import combined_prompt
from ui_failure import STAGE_OPTIMIZATION
from ui_retry import RETRY_TOPOLOGY, retry_action


CASES = (
    ("I want a cup holder attached to this desk", TASK_CUPHOLDER, {"bottle_diameter_mm", "desk_thickness_mm"}),
    ("Design a hook for a bag on my desk", TASK_DESK_HOOK, {"desk_thickness_mm"}),
    ("Design a handle to help a person get up from bed", TASK_BED_HANDLE, set()),
    ("Design a wall-mounted shelf for a router", TASK_WALL_SHELF, set()),
    ("Design a phone stand for my nightstand", TASK_GENERIC, set()),
)

CUPHOLDER_ONLY = {"bottle_diameter_mm", "cup_radius_mm", "desk_thickness_mm"}


class CrossTaskGeneralityTests(unittest.TestCase):
    def test_classification_and_clarification_do_not_leak_unrelated_fields(self) -> None:
        for text, task, expected_extra in CASES:
            with self.subTest(task=task):
                self.assertEqual(classify_design_task(text), task)
                fields = {spec["field"] for spec in clarification_specs_for_task(task)}
                if task != TASK_CUPHOLDER:
                    self.assertNotIn("bottle_diameter_mm", fields)
                if task not in {TASK_CUPHOLDER, TASK_DESK_HOOK}:
                    self.assertNotIn("desk_thickness_mm", fields)
                for field in expected_extra:
                    self.assertIn(field, fields)
                self.assertTrue({"attachment_method", "manufacturing_method"} & fields or "attachment_structure" in fields)

    def test_readiness_is_the_same_backend_list(self) -> None:
        for text, task, _ in CASES:
            with self.subTest(task=task):
                req = UserRequirements(user_message=text, task_kind=task)
                self.assertEqual(
                    missing_requirement_fields(req),
                    [spec["field"] for spec in clarification_specs_for_task(task)],
                )

    def test_warm_start_prompt_is_task_appropriate(self) -> None:
        for text, task, _ in CASES:
            with self.subTest(task=task):
                req = UserRequirements(
                    description=text,
                    user_message=text,
                    task_kind=task,
                )
                prompt = combined_prompt(req, "warm_start")
                self.assertIn(task, prompt)
                if task != TASK_CUPHOLDER:
                    self.assertNotIn("cup_radius_mm", prompt)
                    self.assertNotIn("cup diameter", prompt.lower())
                if task not in {TASK_CUPHOLDER, TASK_DESK_HOOK}:
                    self.assertNotIn("desk_thickness_mm", prompt)
                self.assertIn("union_all", prompt)
                self.assertIn("concatenate", prompt)

    def test_unknown_imported_candidate_is_not_cupholder_family(self) -> None:
        cand = ImportedCandidateGeometry(mesh_path="/tmp/x.stl", task="custom_bracket")
        self.assertEqual(fit_family_for(cand), "generated")
        self.assertNotEqual(fit_family_for(cand), "cupholder")

    def test_retry_is_generic_across_tasks(self) -> None:
        for _, task, _ in CASES:
            with self.subTest(task=task):
                self.assertEqual(
                    retry_action(
                        failure_stage=STAGE_OPTIMIZATION,
                        has_valid_warm_start=True,
                        requirements_changed=False,
                    ),
                    RETRY_TOPOLOGY,
                )


if __name__ == "__main__":
    unittest.main()
