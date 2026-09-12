"""UI session-state helpers and Streamlit AppTest smoke for widget ownership."""

from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

from app import (
    HAPPY_PATH_MESSAGE,
    MISSING_INFO_MESSAGE,
    REJECTED_MESSAGE,
    apply_pending_registration_path,
    apply_pending_request_prefill,
    apply_pending_workspace_tab,
    apply_trusted_registration_answers,
    consume_scroll_to_action,
    field_display_label,
    field_widget_kind,
    mark_ui_transition,
    reconstruction_visuals,
    sync_answer_widgets,
)
from ui_flow import WORKSPACE_TAB_OPTIMIZATION, WORKSPACE_TAB_SCENE


class FieldWidgetTests(unittest.TestCase):
    def test_attach_region_uses_choices_and_short_label(self) -> None:
        from types import SimpleNamespace

        question = SimpleNamespace(
            field="allowed_contact_region",
            question="Where on the desk may it mount (e.g. front edge)?",
        )
        self.assertEqual(field_widget_kind(question), "categorical")
        self.assertEqual(field_display_label(question), "Where should it attach?")

    def test_reach_is_numeric_with_short_label(self) -> None:
        from types import SimpleNamespace

        question = SimpleNamespace(
            field="max_protrusion_mm",
            question="What is the maximum allowed protrusion from the desk (mm)?",
        )
        self.assertEqual(field_widget_kind(question), "numeric")
        self.assertEqual(field_display_label(question), "Maximum reach")

    def test_schema_options_win_over_field_name(self) -> None:
        from types import SimpleNamespace

        question = SimpleNamespace(
            field="custom_mount",
            question="Pick a side",
            options=["left", "right"],
        )
        self.assertEqual(field_widget_kind(question), "categorical")


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

    def test_scroll_flag_is_consumed_once(self) -> None:
        store: dict = {}
        self.assertFalse(consume_scroll_to_action(store))
        mark_ui_transition(store)
        self.assertTrue(consume_scroll_to_action(store))
        self.assertFalse(consume_scroll_to_action(store))

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

    def test_sync_maps_pill_labels_to_stored_values(self) -> None:
        store = {
            "answers": {},
            "ans_attachment_method": "Clamp",
            "ans_allowed_contact_region": "Front edge",
            "ans_manufacturing_method": "3D print",
        }
        sync_answer_widgets(store)
        self.assertEqual(store["answers"]["attachment_method"], "clamp")
        self.assertEqual(store["answers"]["allowed_contact_region"], "desk_front_edge")
        self.assertEqual(store["answers"]["manufacturing_method"], "3d_print")

    def test_sync_does_not_wipe_answers_when_pills_are_empty(self) -> None:
        store = {
            "answers": {
                "attachment_method": "clamp",
                "allowed_contact_region": "desk_front_edge",
            },
            "ans_attachment_method": None,
            "ans_allowed_contact_region": None,
        }
        sync_answer_widgets(store)
        self.assertEqual(store["answers"]["attachment_method"], "clamp")
        self.assertEqual(store["answers"]["allowed_contact_region"], "desk_front_edge")

    def test_trusted_registration_prefills_empty_thickness(self) -> None:
        store = {
            "answers": {},
            "request_text": "I want a cup holder attached to this desk",
            "registration_meas": {"prefill": True, "desk_thickness_mm": 17.32},
        }
        apply_trusted_registration_answers(store)
        self.assertEqual(store["answers"]["desk_thickness_mm"], 17.32)

    def test_trusted_registration_does_not_prefill_desk_for_non_desk_task(self) -> None:
        store = {
            "answers": {},
            "request_text": "Design a handle to help a person get up from bed",
            "registration_meas": {"prefill": True, "desk_thickness_mm": 17.32},
        }
        apply_trusted_registration_answers(store)
        self.assertNotIn("desk_thickness_mm", store["answers"])

    def test_pending_workspace_tab_moves_before_widget_construction(self) -> None:
        store = {
            "workspace_tab": WORKSPACE_TAB_SCENE,
            "pending_workspace_tab": WORKSPACE_TAB_OPTIMIZATION,
        }
        apply_pending_workspace_tab(store)
        self.assertEqual(store["workspace_tab"], WORKSPACE_TAB_OPTIMIZATION)
        self.assertIsNone(store["pending_workspace_tab"])

    def test_trusted_registration_does_not_overwrite_user_thickness(self) -> None:
        store = {
            "answers": {"desk_thickness_mm": 20.0},
            "registration_meas": {"prefill": True, "desk_thickness_mm": 17.32},
        }
        apply_trusted_registration_answers(store)
        self.assertEqual(store["answers"]["desk_thickness_mm"], 20.0)


class StreamlitWidgetOwnershipTests(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        # 60 s, not 12: app.py imports torch transitively, and under the full suite the
        # first AppTest run contends with an already-warm CUDA context. The script itself
        # takes well under a second in isolation.
        at = AppTest.from_file("app.py", default_timeout=180)
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

    def _load_example(self, at, label: str):
        at.selectbox(key="example_choice").set_value(label).run()
        self.assertFalse(at.exception, msg=at.exception)
        self._click(at, "Load example")

    def test_custom_submit_does_not_raise(self) -> None:
        at = self._app()
        custom = "Please make a clamp-on holder for my travel mug."
        at.text_area(key="request_text").set_value(custom).run()
        self.assertFalse(at.exception, msg=at.exception)
        self._click(at, "Start design")
        self.assertEqual(at.session_state.request_text, custom)
        self.assertEqual(at.session_state.submitted_request, custom)
        self.assertTrue(at.session_state.chat)

    def test_happy_path_updates_text_field(self) -> None:
        at = self._app()
        at.text_area(key="request_text").set_value("temporary custom text").run()
        self._load_example(at, "Happy Path")
        self.assertEqual(at.session_state.request_text, HAPPY_PATH_MESSAGE)
        self.assertIsNone(at.session_state.pending_request_prefill)

    def test_missing_information_updates_text_field(self) -> None:
        at = self._app()
        self._load_example(at, "Missing information")
        self.assertEqual(at.session_state.request_text, MISSING_INFO_MESSAGE)

    def test_missing_information_uses_attach_choices(self) -> None:
        at = self._app()
        self._load_example(at, "Missing information")
        pill_labels = [getattr(widget, "label", "") for widget in at.pills]
        self.assertTrue(
            any("attach" in label.lower() for label in pill_labels),
            pill_labels,
        )
        page = " ".join(str(getattr(block, "value", "")) for block in at.markdown).lower()
        self.assertIn("need", page)
        self.assertIn("detail", page)
        self.assertNotIn("not reconstructed", page)
        self.assertNotIn("grok geometry generation is not configured", page)

    def test_rejected_case_updates_text_field(self) -> None:
        at = self._app()
        self._load_example(at, "Rejected")
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
        self._click(at, "Start design")
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
        self._load_example(at, "Happy Path")
        self._click(at, "Run optimization")
        orch = at.session_state.orch
        self.assertEqual(orch.state.stage.value, "complete")
        self.assertEqual(orch.state.safety_status.value, "unverified")
        self.assertIsNone(orch.state.contract_error)

    def test_adaptive_custom_values_pass_geometry(self) -> None:
        from geometry_sources import GEOM_ADAPTIVE

        at = self._app()
        at.radio(key="mode_geom").set_value(GEOM_ADAPTIVE).run()
        self._click(at, "Start design")
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
        self._click(at, "Run optimization")
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
        self._click(at, "Start design")
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
        self._click(at, "Run optimization")
        orch = at.session_state.orch
        self.assertIsNotNone(orch.state.contract_error)
        refuse = "refuses to silently choose"
        first = sum(1 for item in at.session_state.chat if refuse in item["text"])
        self.assertEqual(first, 1)
        self._click(at, "Continue")
        second = sum(1 for item in at.session_state.chat if refuse in item["text"])
        self.assertEqual(second, 1)
        self.assertIn("100", "".join(item["text"] for item in at.session_state.chat))
        self.assertIn("85", "".join(item["text"] for item in at.session_state.chat))

    def test_new_session_defaults_to_from_requirements(self) -> None:
        """Product defaults: deterministic From requirements + live topology. Grok is optional."""
        from app import _default_geometry_mode
        from geometry_sources import GEOM_FROM_REQUIREMENTS

        at = self._app()
        self.assertEqual(at.session_state.mode_geom, _default_geometry_mode())
        self.assertEqual(at.session_state.mode_geom, GEOM_FROM_REQUIREMENTS)
        self.assertEqual(at.session_state.mode_topo, "Live")
        self.assertFalse(at.session_state.enable_mock_fixtures)
        self.assertEqual(at.session_state.mode_reason, "Grok")
        self.assertNotIn("Topology", [w.label for w in at.radio])
        self.assertTrue(any(b.label == "Load example" for b in at.sidebar.button))

    def test_right_workspace_defaults_to_scene_not_engineering(self) -> None:
        at = self._app()
        self.assertEqual(at.session_state.workspace_tab, WORKSPACE_TAB_SCENE)
        values = [str(getattr(block, "value", "")) for block in at.markdown]
        self.assertTrue(
            any(
                "Scene not reconstructed yet" in value or "Add 8 more photos to reconstruct" in value
                for value in values
            ),
            values,
        )
        self.assertFalse(any("Scene reconstructed" in value for value in values))
        self.assertFalse(any("**Engineering**" in value for value in values))
        self.assertFalse(any("Structural concept" in value for value in values))

    def test_ok_run_without_artifacts_is_not_reconstructed(self) -> None:
        from types import SimpleNamespace

        at = self._app()
        at.session_state.registration_run = SimpleNamespace(
            ok=True,
            artifacts={},
            capture_dir="",
            measurements={},
        )
        at.session_state.pending_workspace_tab = WORKSPACE_TAB_SCENE
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        self.assertEqual(at.session_state.workspace_tab, WORKSPACE_TAB_SCENE)
        page = " ".join(str(getattr(block, "value", "")) for block in at.markdown)
        captions = " ".join(str(getattr(block, "value", "")) for block in at.caption)
        blob = f"{page} {captions}"
        self.assertNotIn("Scene reconstructed", blob)
        self.assertTrue(
            "Scene not reconstructed yet" in blob or "more photos to reconstruct" in blob,
            blob,
        )
        self.assertNotIn("Structural concept", page)

    def _png_bytes(self) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(200, 180, 150)).save(buf, format="PNG")
        return buf.getvalue()

    def test_uploaded_photo_is_preview_not_reconstruction(self) -> None:
        at = self._app()
        at.session_state.image_bytes = self._png_bytes()
        at.session_state.image_name = "desk.png"
        at.session_state.pending_workspace_tab = WORKSPACE_TAB_SCENE
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        captions = " ".join(str(getattr(block, "value", "")) for block in at.caption)
        page = " ".join(str(getattr(block, "value", "")) for block in at.markdown)
        blob = f"{page} {captions}"
        self.assertIn("Photo preview", blob)
        self.assertNotIn("Scene reconstructed", blob)
        self.assertIn("Add 7 more photos to reconstruct", blob)

    def test_reconstruction_failure_uses_failure_ux_not_photo(self) -> None:
        at = self._app()
        at.session_state.image_bytes = self._png_bytes()
        at.session_state.image_name = "desk.png"
        at.session_state.registration_error = "SAM failed"
        at.session_state.registration_run = None
        at.session_state.pending_workspace_tab = WORKSPACE_TAB_SCENE
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        page = " ".join(str(getattr(block, "value", "")) for block in at.markdown)
        captions = " ".join(str(getattr(block, "value", "")) for block in at.caption)
        blob = f"{page} {captions}".lower()
        self.assertIn("reconstruct the scene", blob)
        self.assertIn("source photo", blob)
        self.assertNotIn("scene reconstructed", blob)
        self.assertNotIn("photo preview", blob)

    def test_real_artifact_claims_reconstructed(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        at = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            glb = Path(tmp) / "target_mesh_hybrid.glb"
            glb.write_bytes(b"glb")
            at.session_state.registration_run = SimpleNamespace(
                ok=True,
                artifacts={"hybrid_mesh_viewer": str(glb)},
                capture_dir="",
                out_dir=tmp,
                measurements={},
            )
            at.session_state.pending_workspace_tab = WORKSPACE_TAB_SCENE
            at.run()
            self.assertFalse(at.exception, msg=at.exception)
            page = " ".join(str(getattr(block, "value", "")) for block in at.markdown)
            self.assertIn("Scene reconstructed", page)
            self.assertNotIn("Scene not reconstructed yet", page)

    def test_desk_hook_example_keeps_live_topology(self) -> None:
        from geometry_sources import GEOM_IMPORTED

        at = self._app()
        self._load_example(at, "Desk hook (live TO)")
        self.assertEqual(at.session_state.mode_geom, GEOM_IMPORTED)
        self.assertEqual(at.session_state.mode_topo, "Live")
        self.assertFalse(at.session_state.enable_mock_fixtures)
        self.assertEqual(at.session_state.orch.imported_candidate.candidate_name, "desk_bag_hook")
        self.assertTrue(
            any(b.label == "Run optimization" for b in at.button),
            [b.label for b in at.button],
        )

    def test_happy_path_switches_to_golden_geometry(self) -> None:
        from geometry_sources import GEOM_GOLDEN

        at = self._app()
        self._load_example(at, "Happy Path")
        self.assertEqual(at.session_state.mode_geom, GEOM_GOLDEN)

    def test_reset_session_restores_default_geometry(self) -> None:
        from app import _default_geometry_mode
        from geometry_sources import GEOM_GOLDEN

        at = self._app()
        at.radio(key="mode_geom").set_value(GEOM_GOLDEN).run()
        self._click(at, "Reset session")
        self.assertEqual(at.session_state.mode_geom, _default_geometry_mode())


class ReconstructionVisualTests(unittest.TestCase):
    def test_photos_and_empty_ok_run_are_not_reconstruction(self) -> None:
        self.assertFalse(reconstruction_visuals(None)["reconstructed"])
        self.assertFalse(
            reconstruction_visuals(SimpleNamespace(ok=False, artifacts={}, out_dir=""))["reconstructed"]
        )
        self.assertFalse(
            reconstruction_visuals(SimpleNamespace(ok=True, artifacts={}, out_dir=""))["reconstructed"]
        )

    def test_hybrid_glb_counts_as_reconstruction(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            glb = Path(tmp) / "target_mesh_hybrid.glb"
            glb.write_bytes(b"glb")
            visuals = reconstruction_visuals(
                SimpleNamespace(ok=True, artifacts={"hybrid_mesh_viewer": str(glb)}, out_dir=tmp)
            )
            self.assertTrue(visuals["reconstructed"])
            self.assertEqual(visuals["glb"], glb)

    def test_target_glb_and_views_count(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            glb = Path(tmp) / "target.glb"
            top = Path(tmp) / "debug"
            top.mkdir()
            view = top / "world_top.png"
            glb.write_bytes(b"glb")
            view.write_bytes(b"png")
            visuals = reconstruction_visuals(
                SimpleNamespace(ok=True, artifacts={"viewer": str(glb), "top_view": str(view)}, out_dir=tmp)
            )
            self.assertTrue(visuals["reconstructed"])
            self.assertEqual(visuals["glb"], glb)
            self.assertEqual(visuals["views"], [view])


if __name__ == "__main__":
    unittest.main()
