"""Run-optimization terminal instrumentation. Does not exercise solvers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ui_run_log import optimize_click_decision, should_reuse_warm_start


class OptimizePathDecisionTests(unittest.TestCase):
    def test_optimize_without_request_is_blocked(self) -> None:
        reason = optimize_click_decision(
            kind="optimize", has_request=False, answers_complete=True
        )
        self.assertIsNotNone(reason)
        self.assertIn("design request", reason)

    def test_optimize_with_request_proceeds(self) -> None:
        self.assertIsNone(
            optimize_click_decision(kind="optimize", has_request=True, answers_complete=True)
        )

    def test_reuse_valid_warm_start(self) -> None:
        self.assertTrue(
            should_reuse_warm_start(
                warm_start_ok=True,
                has_candidate=True,
                fingerprint="a",
                current_fingerprint="a",
                force_regen=False,
            )
        )
        self.assertFalse(
            should_reuse_warm_start(
                warm_start_ok=True,
                has_candidate=True,
                fingerprint="a",
                current_fingerprint="b",
                force_regen=False,
            )
        )


class OrchestratorTracePrintTests(unittest.TestCase):
    def test_emit_prints_run_prefix(self) -> None:
        from orchestrator import Orchestrator

        orch = Orchestrator()
        buf = io.StringIO()
        with redirect_stdout(buf):
            orch._emit("TO_SOLVER_STARTED")
            orch._emit("BLOCKED: missing analysis")
        text = buf.getvalue()
        self.assertIn("[RUN] TO_SOLVER_STARTED", text)
        self.assertIn("[RUN] BLOCKED: missing analysis", text)


class WarmStartReuseHelperTests(unittest.TestCase):
    def test_invalid_without_mesh(self) -> None:
        from app import _warm_start_is_valid, _warm_start_mesh_path

        self.assertEqual(_warm_start_mesh_path(SimpleNamespace(mesh_path="")), "")
        self.assertFalse(_warm_start_is_valid(None))


if __name__ == "__main__":
    unittest.main()
