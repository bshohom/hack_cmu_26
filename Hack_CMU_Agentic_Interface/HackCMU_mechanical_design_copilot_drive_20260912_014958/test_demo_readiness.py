"""Demo-readiness checks: capture contract, branding, and Grok status copy."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app import (
    API_NOT_CONNECTED,
    BRAND_SUB,
    BRAND_TAGLINE,
    BRAND_TITLE,
    GENERATION_FAILED_VALIDATION,
    SCENE_PHOTO_HELP,
    generator_status_message,
    reconstruction_photo_status,
)
from tools.registration import MAX_IMAGES, MIN_IMAGES, RECOMMENDED_IMAGES
from ui_failure import detect_ui_failure


class ReconstructionContractTests(unittest.TestCase):
    def test_photo_status_matches_capture_minimum(self) -> None:
        two = reconstruction_photo_status(2)
        self.assertEqual(two["count"], 2)
        self.assertEqual(two["min_images"], 8)
        self.assertEqual(two["min_images"], MIN_IMAGES)
        self.assertFalse(two["ready"])
        self.assertEqual(two["count_label"], "2 photos")
        self.assertEqual(two["action"], "Add 6 more photos")
        self.assertNotIn("14", two["action"])
        self.assertNotIn("14", two["count_label"])

        eight = reconstruction_photo_status(8)
        self.assertTrue(eight["ready"])
        self.assertEqual(eight["action"], "Ready to reconstruct")

    def test_recommended_range_is_help_only(self) -> None:
        status = reconstruction_photo_status(0)
        main = " ".join([status["count_label"], status["min_label"], status["action"]])
        self.assertIn("0 photos", main)
        self.assertIn("minimum 8", main)
        self.assertIn("Add 8 more photos", main)
        self.assertNotIn("14–18", main)
        self.assertNotIn("14-18", main)
        self.assertIn("14–18", SCENE_PHOTO_HELP)
        self.assertEqual(RECOMMENDED_IMAGES, 14)
        self.assertEqual(MAX_IMAGES, 18)


class BrandingTests(unittest.TestCase):
    def test_final_branding_copy(self) -> None:
        self.assertEqual(BRAND_TITLE, "On TOP of the World")
        self.assertEqual(BRAND_TAGLINE, "Snap it. TOPtimize it. Print it.")
        self.assertEqual(
            BRAND_SUB,
            "Take a photo. Get a lightweight custom part designed to fit your space, "
            "powered by TOPology optimization.",
        )


class GrokStatusTests(unittest.TestCase):
    def test_api_not_connected_is_not_validation_failure(self) -> None:
        self.assertEqual(generator_status_message(False, None), API_NOT_CONNECTED)
        self.assertEqual(
            generator_status_message(False, {"ok": False, "error": "mesh is not watertight"}),
            API_NOT_CONNECTED,
        )
        self.assertEqual(
            generator_status_message(True, {"ok": False, "error": "mesh is not watertight"}),
            GENERATION_FAILED_VALIDATION,
        )
        self.assertIsNone(generator_status_message(True, None))
        self.assertIsNone(generator_status_message(True, {"ok": True}))

        from types import SimpleNamespace

        card = detect_ui_failure(
            SimpleNamespace(stage="request_information", notes="", topology=None),
            warm_start=None,
            warm_start_error="",
        )
        self.assertIsNone(card)

    def test_grok_provider_reads_env_keys(self) -> None:
        from providers import GrokReasoningProvider

        with patch("providers.load_dotenv"):
            with patch.dict(os.environ, {"GROK_API_KEY": "", "XAI_API_KEY": ""}, clear=False):
                os.environ.pop("GROK_API_KEY", None)
                os.environ.pop("XAI_API_KEY", None)
                missing = GrokReasoningProvider()
                self.assertFalse(missing.configured)
                self.assertIn("GROK_API_KEY", missing.not_connected_reason)

            with patch.dict(os.environ, {"GROK_API_KEY": "test-key", "XAI_API_KEY": ""}, clear=False):
                ready = GrokReasoningProvider()
                self.assertTrue(ready.configured)
                self.assertEqual(ready.not_connected_reason, "")


class DemoAppTests(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=180)
        at.run()
        self.assertFalse(at.exception, msg=at.exception)
        return at

    def _page(self, at) -> str:
        parts = [str(getattr(block, "value", "")) for block in at.markdown]
        parts.extend(str(getattr(block, "value", "")) for block in at.caption)
        return " ".join(parts)

    def test_landing_photo_gate_and_branding(self) -> None:
        at = self._app()
        page = self._page(at)
        self.assertIn(BRAND_TITLE, page)
        self.assertIn(BRAND_TAGLINE, page)
        self.assertIn(BRAND_SUB, page)
        self.assertIn("0 photos", page)
        self.assertIn("minimum 8", page)
        self.assertIn("Add 8 more photos", page)
        reconstruct = [b for b in at.button if b.label == "Reconstruct scene"]
        self.assertTrue(reconstruct)
        self.assertTrue(reconstruct[0].disabled)
        main_copy = " ".join(
            str(getattr(block, "value", ""))
            for block in at.markdown
            if "Take 14–18" not in str(getattr(block, "value", ""))
        )
        self.assertIn("Add 8 more photos", main_copy)
        self.assertNotIn("14–18", main_copy)
        help_text = " ".join(
            str(getattr(uploader, "help", "") or "") for uploader in at.file_uploader
        )
        if help_text.strip():
            self.assertIn("14–18", help_text)
