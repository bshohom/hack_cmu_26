"""Retry optimization reuses a validated warm-start and does not call Grok."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.geometry import GeometryAgent
from agents.structure import StructureAgent
from orchestrator import Orchestrator
from schemas import (
    AnalysisOutput,
    GeometryInput,
    ImportedCandidateGeometry,
    IntegrationFixtures,
    MassProvenance,
    StructureInput,
    UserRequirements,
    WorkflowStage,
)
from tools.topology import TopologyUnavailable
from ui_failure import STAGE_OPTIMIZATION, STAGE_WARM_START, build_failure_card
from ui_retry import (
    PROGRESS_RETRY_TOPOLOGY,
    REGENERATE_DESIGN,
    RETRY_TOPOLOGY,
    progress_caption,
    retry_action,
)


def _candidate(path: str) -> ImportedCandidateGeometry:
    return ImportedCandidateGeometry(
        mesh_path=path,
        task="generated",
        candidate_name="warm_start",
        watertight=True,
        bbox_min_mm=(0.0, -10.0, 0.0),
        bbox_max_mm=(40.0, 10.0, 30.0),
    )


class RetryActionTests(unittest.TestCase):
    def test_topology_failure_reuses_valid_warm_start(self) -> None:
        self.assertEqual(
            retry_action(
                failure_stage=STAGE_OPTIMIZATION,
                has_valid_warm_start=True,
                requirements_changed=False,
            ),
            RETRY_TOPOLOGY,
        )

    def test_changed_requirements_or_missing_mesh_regenerates(self) -> None:
        self.assertEqual(
            retry_action(
                failure_stage=STAGE_OPTIMIZATION,
                has_valid_warm_start=True,
                requirements_changed=True,
            ),
            REGENERATE_DESIGN,
        )
        self.assertEqual(
            retry_action(
                failure_stage=STAGE_OPTIMIZATION,
                has_valid_warm_start=False,
            ),
            REGENERATE_DESIGN,
        )

    def test_warm_start_failure_regenerates(self) -> None:
        self.assertEqual(
            retry_action(
                failure_stage=STAGE_WARM_START,
                has_valid_warm_start=False,
            ),
            REGENERATE_DESIGN,
        )

    def test_retry_progress_copy_is_not_generation_copy(self) -> None:
        self.assertEqual(progress_caption(RETRY_TOPOLOGY), PROGRESS_RETRY_TOPOLOGY)
        self.assertNotIn("preliminary optimized design", progress_caption(RETRY_TOPOLOGY).lower())
        self.assertNotIn("generating", progress_caption(RETRY_TOPOLOGY).lower())

    def test_optimization_card_exposes_both_actions(self) -> None:
        card = build_failure_card(stage=STAGE_OPTIMIZATION, notes="solver failed")
        self.assertEqual(card.primary_label, "Retry optimization")
        self.assertEqual(card.secondary_label, "Regenerate design")


class RetryTopologyOrchestratorTests(unittest.TestCase):
    def test_retry_optimization_reuses_path_and_skips_generator(self) -> None:
        grok = {"n": 0}
        to = {"n": 0}

        def _gen(req):
            grok["n"] += 1
            raise AssertionError("Grok must not run on topology retry")

        def _to(inp, log=None, progress=None):
            to["n"] += 1
            raise TopologyUnavailable("synthetic topology failure")

        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "warm.stl"
            mesh.write_bytes(b"solid ok\nendsolid ok\n")
            candidate = _candidate(str(mesh))
            orch = Orchestrator(
                fixtures=IntegrationFixtures(
                    registration=None, geometry=None, analysis=None, topology=None, cad=None
                ),
                imported_candidate=candidate,
                warm_start_generator=_gen,
            )
            req = UserRequirements(
                description="custom bracket",
                user_message="Design a bracket for this load",
                task_kind="generic",
            )
            req.payload.filled_mass_kg = 2.0
            req.design_envelope.max_protrusion_mm = 80.0
            geom = GeometryAgent().run(GeometryInput(requirements=req))
            structure = StructureAgent().run(
                StructureInput(
                    requirements=req,
                    geometry=geom,
                    payload_mass_kg=2.0,
                    payload_mass_provenance=MassProvenance.USER_REQUIREMENTS,
                )
            )
            orch.state.requirements = req
            orch.state.geometry = geom
            orch.state.structure = structure
            orch.state.analysis = AnalysisOutput(
                load_case_id="static_gravity",
                load_force_N=(0.0, 0.0, -20.0),
                is_mock=True,
            )
            orch.state.imported_candidate = candidate
            orch.state.stage = WorkflowStage.TOPOLOGY_FAILED
            with patch("orchestrator.run_topology_optimization", side_effect=_to):
                orch.retry_topology()
            self.assertEqual(grok["n"], 0)
            self.assertEqual(to["n"], 1)
            self.assertEqual(orch.imported_candidate.mesh_path, str(mesh))
            self.assertEqual(orch.state.imported_candidate.mesh_path, str(mesh))
            self.assertEqual(orch.state.stage, WorkflowStage.TOPOLOGY_FAILED)


if __name__ == "__main__":
    unittest.main()
