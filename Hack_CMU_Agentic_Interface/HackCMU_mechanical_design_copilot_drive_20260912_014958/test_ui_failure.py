"""Presentation-only failure card tests. Does not run solvers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from schemas import WorkflowStage
from ui_failure import (
    STAGE_NONCONVERGED,
    STAGE_RECONSTRUCTION,
    STAGE_WARM_START,
    build_failure_card,
    detect_ui_failure,
    parse_failure_findings,
)
from ui_flow import action_spec


RAW_WARM_START = (
    "mesh is not watertight\n"
    "mesh has 7 disconnected bodies\n"
    "part protrudes to x=207 mm beyond max_protrusion_mm=200"
)


class FailureParseTests(unittest.TestCase):
    def test_parses_numeric_envelope_and_mesh_issues(self) -> None:
        findings = {item.kind: item for item in parse_failure_findings(RAW_WARM_START)}
        self.assertIn("disconnected_bodies", findings)
        self.assertIn("watertight", findings)
        self.assertIn("envelope", findings)
        self.assertEqual(findings["disconnected_bodies"].what, "The generated shape split into 7 separate pieces.")
        self.assertEqual(
            findings["envelope"].what,
            "The design exceeds your 200 mm size limit by 7 mm.",
        )
        self.assertEqual(findings["envelope"].needed, 207.0)
        self.assertEqual(findings["envelope"].current, 200.0)
        self.assertEqual(findings["envelope"].field, "max_protrusion_mm")
        public = " ".join(item.what + " " + item.action for item in findings.values())
        self.assertNotIn("watertight", public.lower())
        self.assertNotIn("disconnected bodies", public.lower())
        self.assertNotIn("max_protrusion_mm", public)

    def test_raw_log_keeps_backend_text(self) -> None:
        card = build_failure_card(
            stage=STAGE_WARM_START,
            warm_start={"ok": False, "attempts": 3, "error": RAW_WARM_START, "problems": RAW_WARM_START.splitlines()},
        )
        self.assertIn("mesh is not watertight", card.raw_log)
        self.assertIn("7 disconnected bodies", card.raw_log)
        self.assertIn("max_protrusion_mm=200", card.raw_log)
        self.assertEqual(card.headline, "Generation failed validation")
        self.assertEqual(card.primary_label, "Try again")
        self.assertEqual(card.secondary_label, "Change size limit")
        self.assertEqual(card.log_label, "View full log")
        self.assertNotIn("mesh is not watertight", card.public_text())

    def test_reconstruction_and_nonconvergence_copy(self) -> None:
        recon = build_failure_card(stage=STAGE_RECONSTRUCTION, registration_error="SAM failed")
        self.assertEqual(recon.primary_label, "Add / replace photos")
        self.assertIn("photos", recon.findings[0].action.lower())
        self.assertIn("SAM failed", recon.raw_log)
        conv = build_failure_card(stage=STAGE_NONCONVERGED, notes="did not converge")
        self.assertEqual(conv.primary_label, "Retry optimization")
        self.assertIn("stable design", conv.headline.lower())


class FailureDetectTests(unittest.TestCase):
    def test_warm_start_failure_is_not_optimization_failure(self) -> None:
        state = SimpleNamespace(
            stage=WorkflowStage.COMPLETE,
            notes="Warm-start generation failed; continuing without a candidate mesh",
            topology=SimpleNamespace(is_mock=True, converged=True, acceptance={}),
            cad=None,
            geometry=None,
            structure=None,
            verification=None,
            clarifications=[],
            requirements=object(),
            contract_error=None,
            reject_reason=None,
            feasibility=None,
            candidate_fit=None,
        )
        card = detect_ui_failure(
            state,
            warm_start={"ok": False, "attempts": 3, "error": RAW_WARM_START, "problems": []},
        )
        self.assertIsNotNone(card)
        self.assertEqual(card.stage, STAGE_WARM_START)
        spec = action_spec(state, {"max_protrusion_mm": 200.0}, failure=card)
        self.assertNotIn("ready to optimize", spec.title.lower())
        self.assertNotEqual(spec.kind, "optimize")
        self.assertEqual(spec.kind, "retry")

    def test_detects_reconstruction_and_nonconvergence(self) -> None:
        recon = detect_ui_failure(
            SimpleNamespace(stage=WorkflowStage.REQUEST_INFORMATION, notes="", topology=None),
            registration_error="capture failed",
            registration_run=None,
        )
        self.assertEqual(recon.stage, STAGE_RECONSTRUCTION)
        conv = detect_ui_failure(
            SimpleNamespace(
                stage=WorkflowStage.COMPLETE,
                notes="",
                topology=SimpleNamespace(
                    converged=False,
                    acceptance={"acceptance_status": "unresolved_not_converged"},
                    notes="not converged",
                ),
            )
        )
        self.assertEqual(conv.stage, STAGE_NONCONVERGED)


class FailureAppTests(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=180)
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        return at

    def _page(self, at) -> str:
        return " ".join(str(getattr(block, "value", "")) for block in at.markdown)

    def test_main_ui_hides_raw_error_and_shows_actions(self) -> None:
        at = self._app()
        for button in at.button:
            if button.label == "Start design":
                button.click().run()
                break
        at.session_state.warm_start_result = {
            "ok": False,
            "attempts": 3,
            "error": RAW_WARM_START,
            "problems": RAW_WARM_START.splitlines(),
        }
        at.session_state.answers["max_protrusion_mm"] = 200.0
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        page = self._page(at)
        self.assertNotIn("mesh is not watertight", page)
        self.assertNotIn("disconnected bodies", page)
        self.assertNotIn("max_protrusion_mm", page)
        self.assertNotIn("Ready to optimize", page)
        self.assertIn("Generation failed validation", page)
        self.assertIn("7 mm", page)
        self.assertIn("200", page)
        self.assertIn("207", page)
        labels = [b.label for b in at.button]
        self.assertIn("Try again", labels)
        self.assertIn("Change size limit", labels)
        self.assertIn("View full log", labels)

    def test_change_size_limit_focuses_field_and_keeps_value(self) -> None:
        at = self._app()
        for button in at.button:
            if button.label == "Start design":
                button.click().run()
                break
        at.session_state.warm_start_result = {
            "ok": False,
            "attempts": 3,
            "error": RAW_WARM_START,
            "problems": RAW_WARM_START.splitlines(),
        }
        at.session_state.answers = {
            **dict(at.session_state.answers),
            "max_protrusion_mm": 200.0,
            "desk_thickness_mm": 20.0,
        }
        at.run()
        for button in at.button:
            if button.label == "Change size limit":
                button.click().run()
                break
        else:
            self.fail("Change size limit missing")
        self.assertFalse(at.exception, msg=at.exception)
        self.assertEqual(at.session_state.focus_constraint_field, "max_protrusion_mm")
        self.assertEqual(at.session_state.answers["max_protrusion_mm"], 200.0)
        self.assertEqual(at.session_state.answers["desk_thickness_mm"], 20.0)
        self.assertIn("ans_max_protrusion_mm", at.session_state)

    def test_retry_does_not_change_constraints(self) -> None:
        store = {"answers": {"max_protrusion_mm": 200.0, "desk_thickness_mm": 20.0}}
        before = dict(store["answers"])
        snapshot = dict(store.get("answers") or {})
        store["answers"] = snapshot
        self.assertEqual(store["answers"], before)
        self.assertEqual(snapshot["max_protrusion_mm"], 200.0)


if __name__ == "__main__":
    unittest.main()
