"""UI session-state helpers and Streamlit AppTest smoke for widget ownership."""

from __future__ import annotations

import io
import unittest

from app import (
    HAPPY_PATH_MESSAGE,
    MISSING_INFO_MESSAGE,
    REJECTED_MESSAGE,
    apply_pending_registration_path,
    apply_pending_request_prefill,
    sync_answer_widgets,
)


class PendingPrefillTests(unittest.TestCase):
    def test_pending_overwrites_widget_key_before_construction(self) -> None:
        store = {
            "request_text": "old custom text",
            "pending_request_prefill": "new scenario text",
        }
        apply_pending_request_prefill(store)
        self.assertEqual(store["request_text"], "new scenario text")
        self.assertIsNone(store["pending_request_prefill"])

    def test_no_pending_leaves_widget_key_alone(self) -> None:
        store = {"request_text": "typed by user", "pending_request_prefill": None}
        apply_pending_request_prefill(store)
        self.assertEqual(store["request_text"], "typed by user")
        self.assertIsNone(store["pending_request_prefill"])

    def test_completed_registration_path_moves_before_widget_construction(self) -> None:
        store = {
            "reg_target_path": "old.json",
            "pending_reg_target_path": "/tmp/new/target.json",
        }
        apply_pending_registration_path(store)
        self.assertEqual(store["reg_target_path"], "/tmp/new/target.json")
        self.assertIsNone(store["pending_reg_target_path"])

    def test_current_widget_values_win_when_continue_is_clicked(self) -> None:
        store = {
            "answers": {"desk_thickness_mm": 20.0},
            "ans_desk_thickness_mm": 17.32,
            "no_drill": True,
        }
        sync_answer_widgets(store)
        self.assertEqual(store["answers"]["desk_thickness_mm"], 17.32)
        self.assertEqual(
            store["answers"]["attachment_notes"], "clamp only, no drilling"
        )


class StreamlitWidgetOwnershipTests(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        # 60 s, not 12: app.py imports torch transitively, and under the full suite the
        # first AppTest run contends with an already-warm CUDA context. The script itself
        # takes well under a second in isolation.
        at = AppTest.from_file("app.py", default_timeout=60)
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        return at

    def _click(self, at, label: str):
        for button in at.button:
            if button.label == label:
                button.click().run()
                self.assertFalse(at.exception, msg=at.exception)
                return
        self.fail(f"No button labeled {label!r}")

    def test_custom_submit_does_not_raise(self) -> None:
        at = self._app()
        custom = "Please make a clamp-on holder for my travel mug."
        at.text_area(key="request_text").set_value(custom).run()
        self.assertFalse(at.exception, msg=at.exception)
        self._click(at, "Submit request")
        self.assertEqual(at.session_state.request_text, custom)
        self.assertEqual(at.session_state.submitted_request, custom)
        self.assertTrue(at.session_state.chat)

    def test_happy_path_updates_text_field(self) -> None:
        at = self._app()
        at.text_area(key="request_text").set_value("temporary custom text").run()
        self._click(at, "Load Happy Path")
        self.assertEqual(at.session_state.request_text, HAPPY_PATH_MESSAGE)
        self.assertIsNone(at.session_state.pending_request_prefill)

    def test_missing_information_updates_text_field(self) -> None:
        at = self._app()
        self._click(at, "Load Missing Information Case")
        self.assertEqual(at.session_state.request_text, MISSING_INFO_MESSAGE)

    def test_rejected_case_updates_text_field(self) -> None:
        at = self._app()
        self._click(at, "Load Rejected Case")
        self.assertEqual(at.session_state.request_text, REJECTED_MESSAGE)

    def test_upload_image_and_submit_custom_text(self) -> None:
        at = self._app()
        # AppTest has no file_uploader accessor; mimic the copy the UI makes.
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(200, 180, 150)).save(buf, format="PNG")
        at.session_state.image_bytes = buf.getvalue()
        at.session_state.image_name = "desk.png"
        custom = "Custom request with a desk photo attached."
        at.text_area(key="request_text").set_value(custom).run()
        self._click(at, "Submit request")
        self.assertEqual(at.session_state.request_text, custom)
        self.assertEqual(at.session_state.submitted_request, custom)
        self.assertEqual(at.session_state.image_name, "desk.png")
        self.assertTrue(at.session_state.image_bytes)

    def _fill_answers(self, at, **values):
        for key, value in values.items():
            widget_key = f"ans_{key}"
            if widget_key in at.session_state:
                at.session_state[widget_key] = value
            at.session_state.answers[key] = value
        at.run()

    def test_golden_happy_path_reaches_complete(self) -> None:
        at = self._app()
        self._click(at, "Load Happy Path")
        self._click(at, "Continue design")
        orch = at.session_state.orch
        self.assertEqual(orch.state.stage.value, "complete")
        self.assertEqual(orch.state.safety_status.value, "unverified")
        self.assertIsNone(orch.state.contract_error)

    def test_adaptive_custom_values_pass_geometry(self) -> None:
        from geometry_sources import GEOM_ADAPTIVE

        at = self._app()
        at.radio(key="mode_geom").set_value(GEOM_ADAPTIVE).run()
        self._click(at, "Submit request")
        self._fill_answers(
            at,
            filled_bottle_mass_kg=1.0,
            bottle_diameter_mm=100.0,
            desk_thickness_mm=20.0,
            max_protrusion_mm=150.0,
            attachment_method="clamp",
            allowed_contact_region="desk_front_edge",
            manufacturing_method="3d_print",
        )
        self._click(at, "Continue design")
        orch = at.session_state.orch
        self.assertIsNone(orch.state.contract_error)
        geom = orch.state.geometry
        self.assertIsNotNone(geom)
        self.assertEqual(geom.payload_object.bottle_diameter_mm, 100.0)
        self.assertEqual(geom.environment.desk_thickness_mm, 20.0)
        self.assertEqual(geom.design_envelope.max_protrusion_mm, 150.0)
        self.assertIsNotNone(orch.state.structure)

    def test_golden_custom_values_block_once(self) -> None:
        from geometry_sources import GEOM_GOLDEN

        at = self._app()
        at.radio(key="mode_geom").set_value(GEOM_GOLDEN).run()
        self._click(at, "Submit request")
        self._fill_answers(
            at,
            filled_bottle_mass_kg=1.0,
            bottle_diameter_mm=100.0,
            desk_thickness_mm=20.0,
            max_protrusion_mm=150.0,
            attachment_method="clamp",
            allowed_contact_region="desk_front_edge",
            manufacturing_method="3d_print",
        )
        self._click(at, "Continue design")
        orch = at.session_state.orch
        self.assertIsNotNone(orch.state.contract_error)
        refuse = "refuses to silently choose"
        first = sum(1 for item in at.session_state.chat if refuse in item["text"])
        self.assertEqual(first, 1)
        self._click(at, "Continue design")
        second = sum(1 for item in at.session_state.chat if refuse in item["text"])
        self.assertEqual(second, 1)
        self.assertIn("100", "".join(item["text"] for item in at.session_state.chat))
        self.assertIn("85", "".join(item["text"] for item in at.session_state.chat))

    def test_new_session_defaults_to_generated_warm_start(self) -> None:
        """Warm-start geometry is generated from the user's own measurements, so it is the
        default; without a generator configured the session falls back to the synthetic mock."""
        from app import _default_geometry_mode
        from geometry_sources import GEOM_ADAPTIVE, GEOM_GENERATED

        at = self._app()
        self.assertEqual(at.session_state.mode_geom, _default_geometry_mode())
        self.assertIn(at.session_state.mode_geom, (GEOM_GENERATED, GEOM_ADAPTIVE))
        self.assertEqual(at.session_state.mode_topo, "Live")

    def test_happy_path_switches_to_golden_geometry(self) -> None:
        from geometry_sources import GEOM_GOLDEN

        at = self._app()
        self._click(at, "Load Happy Path")
        self.assertEqual(at.session_state.mode_geom, GEOM_GOLDEN)

    def test_reset_session_restores_default_geometry(self) -> None:
        from app import _default_geometry_mode
        from geometry_sources import GEOM_GOLDEN

        at = self._app()
        at.radio(key="mode_geom").set_value(GEOM_GOLDEN).run()
        self._click(at, "Reset session")
        self.assertEqual(at.session_state.mode_geom, _default_geometry_mode())


if __name__ == "__main__":
    unittest.main()
