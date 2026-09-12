"""Live topology tool wiring: mock fallback paths and the live path with a fake adapter."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from schemas import ImportedCandidateGeometry, LoadCase, PartGeometry, TopologyInput, TopologyOutput
from tools import topology as topology_tool


def _inp(candidate=None) -> TopologyInput:
    return TopologyInput(
        design_domain=PartGeometry(length_mm=120.0, width_mm=100.0, height_mm=45.0, volume_mm3=1.0),
        loads=[LoadCase(load_case_id="static_gravity", name="static_gravity", region_name="cup_cavity", force_N=(0.0, 0.0, -10.8))],
        candidate=candidate,
        target_volume_fraction=0.4,
    )


def _candidate() -> ImportedCandidateGeometry:
    return ImportedCandidateGeometry(mesh_path="/nonexistent/c.stl", task="cupholder", candidate_name="cupholder")


class TopologyToolTests(unittest.TestCase):
    def test_no_candidate_designs_from_scratch(self) -> None:
        """Without a candidate mesh the tool still runs live (from requirements), not a mock."""
        seen = {}

        def fake(inp, out_root=None, log=None, progress=None):
            seen["candidate"] = inp.get("candidate")
            return {"is_mock": False, "optimized_geometry_ref": "/tmp/d.stl", "solver_status": "converged",
                    "model": "from scratch", "notes": "designed from the requirements"}

        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "auto"}), mock.patch.object(
            topology_tool, "_import_adapter", return_value=fake
        ):
            out = topology_tool.run_topology_optimization(_inp(None))
        self.assertIsNone(seen["candidate"])
        self.assertFalse(out.is_mock)
        self.assertIn("requirements", out.notes)

    def test_mode_off_is_mock(self) -> None:
        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "off"}):
            out = topology_tool.run_topology_optimization(_inp(_candidate()))
        self.assertTrue(out.is_mock)
        self.assertIn("TO_AGENT_MODE=off", out.notes)

    def test_adapter_failure_degrades_to_mock(self) -> None:
        def boom(_inp, out_root=None, log=None, progress=None):
            raise ValueError("problem too big")

        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "auto"}), mock.patch.object(
            topology_tool, "_import_adapter", return_value=boom
        ):
            out = topology_tool.run_topology_optimization(_inp(_candidate()))
        self.assertTrue(out.is_mock)
        self.assertIn("problem too big", out.notes)

    def test_import_error_degrades_to_mock(self) -> None:
        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "auto"}), mock.patch.object(
            topology_tool, "_import_adapter", side_effect=ImportError("no torch")
        ):
            out = topology_tool.run_topology_optimization(_inp(_candidate()))
        self.assertTrue(out.is_mock)
        self.assertIn("not importable", out.notes)

    def test_live_result_is_validated(self) -> None:
        def fake(inp, out_root=None, log=None, progress=None):
            self.assertEqual(inp["candidate"]["task"], "cupholder")
            return {
                "is_mock": False,
                "compliance": 0.46,
                "volume_fraction": 0.2,
                "mass_reduction_pct": 30.5,
                "optimized_geometry_ref": "/tmp/design.stl",
                "solver_status": "converged",
                "model": "to_agent SIMP/OC",
                "artifacts": {"design_stl": "/tmp/design.stl"},
                "iterations": 40,
                "wall_time_s": 41.9,
                "converged": True,
                "notes": "ok",
            }

        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "auto"}), mock.patch.object(
            topology_tool, "_import_adapter", return_value=fake
        ):
            out = topology_tool.run_topology_optimization(_inp(_candidate()))
        self.assertIsInstance(out, TopologyOutput)
        self.assertFalse(out.is_mock)
        self.assertEqual(out.iterations, 40)
        self.assertEqual(out.optimized_geometry_ref, "/tmp/design.stl")


if __name__ == "__main__":
    unittest.main()
