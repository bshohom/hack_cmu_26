"""Live topology tool wiring: mock fallback paths and the live path with a fake adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from schemas import ImportedCandidateGeometry, LoadCase, PartGeometry, TopologyInput, TopologyOutput
from tools import topology as topology_tool


def _real_stl(tmpdir: str, name: str = "design.stl") -> str:
    """A non-empty file on disk: the tool rejects a result naming a mesh that isn't there."""
    path = Path(tmpdir) / name
    path.write_text("solid x\nendsolid x\n")
    return str(path)


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

        with tempfile.TemporaryDirectory() as tmp:
            stl = _real_stl(tmp, "d.stl")

            def fake(inp, out_root=None, log=None, progress=None):
                seen["candidate"] = inp.get("candidate")
                return {"is_mock": False, "optimized_geometry_ref": stl, "solver_status": "converged",
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
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        stl = _real_stl(tmpdir.name)

        def fake(inp, out_root=None, log=None, progress=None):
            self.assertEqual(inp["candidate"]["task"], "cupholder")
            return {
                "is_mock": False,
                "compliance": 0.46,
                "volume_fraction": 0.2,
                "mass_reduction_pct": 30.5,
                "optimized_geometry_ref": stl,
                "solver_status": "converged",
                "model": "to_agent SIMP/OC",
                "artifacts": {"design_stl": stl},
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
        self.assertEqual(out.optimized_geometry_ref, stl)


    def test_result_naming_a_missing_mesh_is_not_a_success(self) -> None:
        """A run that reports success but wrote no file must not read as a live result."""
        def fake(inp, out_root=None, log=None, progress=None):
            return {"is_mock": False, "optimized_geometry_ref": "/tmp/never_written.stl",
                    "solver_status": "converged", "model": "x", "notes": "ok"}

        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "auto"}), mock.patch.object(
            topology_tool, "_import_adapter", return_value=fake
        ):
            out = topology_tool.run_topology_optimization(_inp(_candidate()))
        self.assertTrue(out.is_mock)
        self.assertIn("produced no mesh file", out.notes)

    def test_live_mode_raises_instead_of_returning_a_placeholder(self) -> None:
        """TO_AGENT_MODE=live: the workflow must fail, not continue on a mock."""
        def boom(_inp, out_root=None, log=None, progress=None):
            raise ValueError("problem too big")

        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "live"}), mock.patch.object(
            topology_tool, "_import_adapter", return_value=boom
        ):
            with self.assertRaises(topology_tool.TopologyUnavailable) as ctx:
                topology_tool.run_topology_optimization(_inp(_candidate()))
        self.assertIn("problem too big", str(ctx.exception))

    def test_live_mode_raises_when_the_mesh_file_is_missing(self) -> None:
        def fake(inp, out_root=None, log=None, progress=None):
            return {"is_mock": False, "optimized_geometry_ref": "/tmp/never_written.stl",
                    "solver_status": "converged", "model": "x", "notes": "ok"}

        with mock.patch.dict(os.environ, {"TO_AGENT_MODE": "live"}), mock.patch.object(
            topology_tool, "_import_adapter", return_value=fake
        ):
            with self.assertRaises(topology_tool.TopologyUnavailable):
                topology_tool.run_topology_optimization(_inp(_candidate()))


if __name__ == "__main__":
    unittest.main()
