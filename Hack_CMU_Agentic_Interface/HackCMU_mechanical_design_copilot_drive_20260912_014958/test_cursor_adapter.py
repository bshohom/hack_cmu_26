"""Tests for Cursor SceneObservation adapter. Does not call the live SDK."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cursor_adapter import (
    NOT_CONNECTED,
    extract_json_object,
    is_cursor_configured,
    not_connected_reason,
    observation_to_requirements_update,
    parse_scene_observation,
)
from providers import CursorReasoningProvider, get_provider
from schemas import SceneObservation


SAMPLE = {
    "detected_payload_type": "cylindrical bottle",
    "detected_support_type": "desk edge",
    "likely_attachment_regions": ["desk front edge"],
    "likely_attachment_methods": ["clamp"],
    "visible_constraints": ["no obvious screw holes"],
    "inferred_values": {"bottle_diameter_mm": 85},
    "missing_measurements": [
        "bottle_diameter_mm",
        "filled_bottle_mass_kg",
        "desk_thickness_mm",
    ],
    "uncertainties": [
        "desk thickness cannot be measured reliably from a single image"
    ],
    "assumptions": ["payload is a bottle"],
    "confidence": 0.82,
    "source": "cursor_live",
}


class SceneObservationSchemaTests(unittest.TestCase):
    def test_parses_example_payload(self) -> None:
        obs = parse_scene_observation(SAMPLE)
        self.assertEqual(obs.detected_payload_type, "cylindrical bottle")
        self.assertEqual(obs.source, "cursor_live")
        self.assertAlmostEqual(obs.confidence, 0.82)
        self.assertIn("bottle_diameter_mm", obs.missing_measurements)

    def test_parses_fenced_json(self) -> None:
        text = "Here you go:\n```json\n" + __import__("json").dumps(SAMPLE) + "\n```\n"
        obs = parse_scene_observation(text)
        self.assertEqual(obs.detected_support_type, "desk edge")

    def test_confidence_percent_is_scaled(self) -> None:
        payload = dict(SAMPLE)
        payload["confidence"] = 82
        obs = parse_scene_observation(payload)
        self.assertAlmostEqual(obs.confidence, 0.82)

    def test_forces_source_cursor_live(self) -> None:
        payload = dict(SAMPLE)
        payload["source"] = "something_else"
        obs = parse_scene_observation(payload)
        self.assertEqual(obs.source, "cursor_live")

    def test_extract_json_object_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("   ")

    def test_inferred_values_are_not_requirements(self) -> None:
        obs = SceneObservation.model_validate(SAMPLE)
        self.assertEqual(observation_to_requirements_update(obs), {})


class CursorProviderFallbackTests(unittest.TestCase):
    def test_missing_key_does_not_crash(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "CURSOR_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(is_cursor_configured())
            self.assertIn(NOT_CONNECTED, not_connected_reason())
            provider = CursorReasoningProvider()
            self.assertFalse(provider.configured)
            obs = provider.observe_scene(
                b"not-an-image",
                "Here is a cup and I want to design a cup holder.",
                model_id="unused",
                image_name="desk.png",
            )
            self.assertIsNone(obs.observation)
            self.assertIn(NOT_CONNECTED, obs.error)
            self.assertEqual(obs.latency_s, 0.0)
            self.assertEqual(obs.model_id, "unused")

    def test_get_provider_cursor_without_key(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "CURSOR_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider("cursor")
            self.assertEqual(provider.name, "cursor")
            self.assertFalse(provider.configured)

    def test_observation_does_not_fill_orchestrator_requirements(self) -> None:
        from orchestrator import Orchestrator
        from schemas import WorkflowStage

        obs = parse_scene_observation(SAMPLE)
        self.assertEqual(observation_to_requirements_update(obs), {})
        orch = Orchestrator()
        orch.ingest_user_request(
            "Here is a cup and I want to design a cup holder for it, "
            "which should be attached to the desk in the image."
        )
        self.assertEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        fields = {item.field for item in orch.state.missing_information}
        self.assertIn("bottle_diameter_mm", fields)
        self.assertIn("filled_bottle_mass_kg", fields)
        self.assertIn("desk_thickness_mm", fields)
        self.assertIsNone(orch.state.requirements.object_geometry.bottle_diameter_mm)
        self.assertIsNone(orch.state.requirements.payload.filled_mass_kg)
        self.assertIsNone(orch.state.requirements.environment.desk_thickness_mm)

    def test_complete_does_not_use_live_agent(self) -> None:
        from schemas import ReasoningRole
        from state import DesignState

        env = {k: v for k, v in os.environ.items() if k != "CURSOR_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = CursorReasoningProvider()
            result = provider.complete(ReasoningRole.INTERACTION, DesignState())
            self.assertIn("does not mutate DesignState", result.notes)
            self.assertTrue(result.is_mock)


class CatalogParamResolutionTests(unittest.TestCase):
    def _gpt(self):
        return {
            "id": "gpt-5.6-sol",
            "parameters": [
                {
                    "id": "context",
                    "display_name": "Context",
                    "values": [
                        {"value": "272k", "display_name": "272k"},
                        {"value": "1M", "display_name": "1M"},
                    ],
                },
                {
                    "id": "reasoning",
                    "display_name": "Reasoning",
                    "values": [
                        {"value": "low", "display_name": "Low"},
                        {"value": "high", "display_name": "High"},
                    ],
                },
                {
                    "id": "fast",
                    "display_name": "Fast",
                    "values": [
                        {"value": "true", "display_name": "On"},
                        {"value": "false", "display_name": "Off"},
                    ],
                },
            ],
        }

    def _sonnet(self):
        return {
            "id": "claude-sonnet-5",
            "parameters": [
                {
                    "id": "thinking",
                    "display_name": "Thinking",
                    "values": [
                        {"value": "true", "display_name": "On"},
                        {"value": "false", "display_name": "Off"},
                    ],
                },
                {
                    "id": "effort",
                    "display_name": "Effort",
                    "values": [
                        {"value": "low", "display_name": "Low"},
                        {"value": "high", "display_name": "High"},
                    ],
                },
            ],
        }

    def test_default_gpt_params_skip_1m(self) -> None:
        from cursor_adapter import DEFAULT_PARAM_HINTS, resolve_param_values

        resolved = resolve_param_values(self._gpt(), DEFAULT_PARAM_HINTS)
        self.assertEqual(resolved["context"], "272k")
        self.assertEqual(resolved["reasoning"], "high")
        self.assertEqual(resolved["fast"], "false")
        self.assertNotIn("1M", resolved.values())

    def test_drops_unknown_parameter_names(self) -> None:
        from cursor_adapter import resolve_param_values

        resolved = resolve_param_values(
            self._gpt(),
            {"context": "272k", "not_a_real_param": "yes", "reasoning": "extreme"},
        )
        self.assertEqual(resolved.get("context"), "272k")
        self.assertNotIn("not_a_real_param", resolved)
        self.assertNotIn("reasoning", resolved)

    def test_compare_b_params(self) -> None:
        from cursor_adapter import COMPARE_B_PARAM_HINTS, resolve_param_values

        resolved = resolve_param_values(self._sonnet(), COMPARE_B_PARAM_HINTS)
        self.assertEqual(resolved["thinking"], "true")
        self.assertEqual(resolved["effort"], "high")

    def test_pick_prefers_confirmed_id(self) -> None:
        from cursor_adapter import pick_catalog_model_id

        catalog = [self._sonnet(), self._gpt()]
        self.assertEqual(pick_catalog_model_id(catalog, "gpt-5.6-sol"), "gpt-5.6-sol")
        self.assertEqual(pick_catalog_model_id(catalog, "missing"), "claude-sonnet-5")


if __name__ == "__main__":
    unittest.main()
