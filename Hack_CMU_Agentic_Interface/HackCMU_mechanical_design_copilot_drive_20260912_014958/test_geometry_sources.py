"""Geometry source modes: golden fixture vs adaptive synthetic mock."""

from __future__ import annotations

import unittest

from fixtures import load_clarifications, load_user_request
from geometry_sources import (
    GEOM_ADAPTIVE,
    GEOM_GOLDEN,
    chat_error_key,
    fixtures_for_geometry_mode,
    format_mismatch_message,
    requirement_geometry_rows,
)
from orchestrator import Orchestrator
from schemas import RequirementsUpdate, SafetyStatus, WorkflowStage

CUSTOM_UPDATE = RequirementsUpdate(
    filled_bottle_mass_kg=1.0,
    bottle_diameter_mm=100.0,
    bottle_height_mm=250.0,
    desk_thickness_mm=20.0,
    attachment_method="clamp",
    allowed_contact_region="desk_front_edge",
    attachment_notes="clamp only, no drilling",
    max_protrusion_mm=150.0,
    manufacturing_method="3d_print",
    material="PLA",
)


def _run(message: str, update: RequirementsUpdate, mode: str) -> Orchestrator:
    orch = Orchestrator(fixtures=fixtures_for_geometry_mode(mode))
    orch.ingest_user_request(message)
    orch.apply_answers(update)
    orch.run()
    return orch


class GeometrySourceTests(unittest.TestCase):
    def test_golden_happy_path_reaches_complete_unverified(self) -> None:
        orch = _run(load_user_request(), load_clarifications(), GEOM_GOLDEN)
        self.assertEqual(orch.state.stage, WorkflowStage.COMPLETE)
        self.assertEqual(orch.state.safety_status, SafetyStatus.UNVERIFIED)
        self.assertIsNone(orch.state.contract_error)
        geom = orch.state.geometry
        assert geom is not None
        self.assertEqual(geom.payload_object.bottle_diameter_mm, 85.0)
        self.assertEqual(geom.environment.desk_thickness_mm, 25.0)
        self.assertEqual(geom.design_envelope.max_protrusion_mm, 130.0)

    def test_adaptive_uses_user_entered_dimensions(self) -> None:
        orch = _run(load_user_request(), CUSTOM_UPDATE, GEOM_ADAPTIVE)
        self.assertIsNone(orch.state.contract_error)
        geom = orch.state.geometry
        assert geom is not None
        self.assertFalse(geom.is_mock)
        self.assertEqual(geom.payload_object.bottle_diameter_mm, 100.0)
        self.assertEqual(geom.payload_object.filled_mass_kg, 1.0)
        self.assertEqual(geom.environment.desk_thickness_mm, 20.0)
        self.assertEqual(geom.design_envelope.max_protrusion_mm, 150.0)
        self.assertIsNotNone(orch.state.structure)
        self.assertNotEqual(orch.state.stage, WorkflowStage.GEOMETRY)
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)

    def test_golden_custom_values_block_with_comparison(self) -> None:
        orch = _run(load_user_request(), CUSTOM_UPDATE, GEOM_GOLDEN)
        self.assertIsNotNone(orch.state.contract_error)
        self.assertIn("disagree", orch.state.contract_error or "")
        rows = requirement_geometry_rows(orch.state)
        by_label = {row["label"]: row for row in rows}
        diameter = by_label["Bottle / payload diameter (mm)"]
        self.assertTrue(diameter["disagree"])
        self.assertEqual(diameter["requirement"], 100.0)
        self.assertEqual(diameter["geometry"], 85.0)
        desk = by_label["Desk / mounting-surface thickness (mm)"]
        self.assertTrue(desk["disagree"])
        self.assertEqual(desk["requirement"], 20.0)
        self.assertEqual(desk["geometry"], 25.0)
        text = format_mismatch_message(orch.state)
        self.assertIn("100", text)
        self.assertIn("85", text)
        self.assertIn("refuses to silently choose", text)
        first = chat_error_key(orch.state)
        orch.run()
        self.assertEqual(chat_error_key(orch.state), first)


if __name__ == "__main__":
    unittest.main()
