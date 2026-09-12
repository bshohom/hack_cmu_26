"""Warm-start generation contract, task-aware feasibility, and failure state."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.feasibility import check_design_feasibility
from agents.geometry import GeometryAgent
from agents.interaction import (
    TASK_BED_HANDLE,
    TASK_CUPHOLDER,
    TASK_DESK_HOOK,
    TASK_WALL_SHELF,
    missing_requirement_fields,
)
from imported_candidate import fit_family_for
from orchestrator import Orchestrator
from schemas import (
    GeometryInput,
    ImportedCandidateGeometry,
    IntegrationFixtures,
    RequirementsUpdate,
    UserRequirements,
    WorkflowStage,
)
from tools.warmstart import (
    SYSTEM_CONTRACT,
    combined_prompt,
    script_failure_feedback,
    script_uses_concatenate_as_final_assembly,
)
from ui_failure import STAGE_WARM_START, detect_ui_failure


def _bed_handle_update() -> RequirementsUpdate:
    return RequirementsUpdate(
        supported_load_kg=80.0,
        attachment_structure="wall",
        handle_location="bedside",
        mounting_region="wall",
        drilling_allowed=False,
        required_reach_mm=50.0,
        max_protrusion_mm=50.0,
        manufacturing_method="3d_print",
        attachment_method="adhesive",
    )


def _cupholder_req(**over) -> UserRequirements:
    req = UserRequirements(
        description="desk cup holder",
        user_message="I want a cup holder attached to this desk",
        task_kind=TASK_CUPHOLDER,
    )
    req.payload.filled_mass_kg = over.get("mass", 1.1)
    req.object_geometry.bottle_diameter_mm = over.get("diameter", 100.0)
    req.environment.desk_thickness_mm = over.get("desk", 24.0)
    req.design_envelope.max_protrusion_mm = over.get("protrusion", 150.0)
    req.attachment.method = "clamp"
    return req


def _geom(req: UserRequirements):
    return GeometryAgent().run(GeometryInput(requirements=req))


class TrimeshContractTests(unittest.TestCase):
    def test_prompt_forbids_remove_duplicate_faces(self) -> None:
        self.assertIn("remove_duplicate_faces", SYSTEM_CONTRACT)
        self.assertIn("FORBIDDEN", SYSTEM_CONTRACT)
        self.assertIn("cleanup_mesh", SYSTEM_CONTRACT)
        self.assertIn("unique_faces", SYSTEM_CONTRACT)
        self.assertIn("union_all", SYSTEM_CONTRACT)
        self.assertIn("concatenate", SYSTEM_CONTRACT)

    def test_unsupported_api_feedback_names_method_and_traceback(self) -> None:
        traceback = (
            "Traceback (most recent call last):\n"
            "  File \"script.py\", line 54, in build\n"
            "    body.remove_duplicate_faces()\n"
            "AttributeError: 'Trimesh' object has no attribute 'remove_duplicate_faces'\n"
        )
        feedback = script_failure_feedback({"error": traceback, "problems": []})
        self.assertIn("UNSUPPORTED API", feedback)
        self.assertIn("remove_duplicate_faces", feedback)
        self.assertIn("AttributeError", feedback)
        self.assertIn("cleanup_mesh", feedback)

    def test_runner_cleanup_does_not_need_remove_duplicate_faces(self) -> None:
        from tools.warmstart_runner import cleanup_mesh
        import trimesh

        mesh = trimesh.creation.box(extents=(10, 10, 10))
        cleaned = cleanup_mesh(mesh)
        self.assertIsInstance(cleaned, trimesh.Trimesh)
        self.assertGreater(len(cleaned.faces), 0)

    def test_generated_script_can_use_cleanup_helper(self) -> None:
        from tools.warmstart import run_script

        script = """
import trimesh
PARAMS = {"w": 20.0}
def build(params):
    mesh = trimesh.creation.box(extents=(params["w"], 10, 10))
    return cleanup_mesh(mesh)
DIMENSIONS = {"w": 20.0}
REGIONS = {
    "load": {"min": [0, -5, -5], "max": [20, 5, 5]},
    "mounts": [{"name": "pad", "min": [0, -5, -5], "max": [2, 5, 5]}],
}
NOTES = []
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.py"
            path.write_text(script)
            result = run_script(path, Path(tmp), "part", {"max_protrusion_mm": 40.0})
        self.assertNotIn("remove_duplicate_faces", script)
        self.assertIn("cleanup_mesh", script)
        self.assertTrue(result.get("ok"), result)

    def test_concatenate_is_rejected_and_reports_components(self) -> None:
        from tools.warmstart import run_script

        script = """
import trimesh
from trimesh.transformations import translation_matrix
PARAMS = {}
def build(params):
    a = trimesh.creation.box(extents=(10, 10, 10), transform=translation_matrix([0, 0, 0]))
    b = trimesh.creation.box(extents=(10, 10, 10), transform=translation_matrix([20, 0, 0]))
    return trimesh.util.concatenate([a, b])
DIMENSIONS = {}
REGIONS = {"load": {"min": [-5, -5, -5], "max": [5, 5, 5]}, "mounts": [{"name": "a", "min": [-5, -5, -5], "max": [5, 5, 5]}]}
NOTES = []
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.py"
            path.write_text(script)
            result = run_script(path, Path(tmp), "part", {"max_protrusion_mm": 40.0})
        self.assertFalse(result.get("ok"))
        blob = "\n".join(result.get("problems") or [])
        self.assertIn("concatenate", blob.lower())
        self.assertIn("disconnected", blob.lower())
        self.assertIn("component[0]", blob)
        self.assertIn("nearest gap", blob.lower())

    def test_union_all_of_overlapping_boxes_is_one_solid(self) -> None:
        from tools.warmstart import run_script

        script = """
import trimesh
from trimesh.transformations import translation_matrix
PARAMS = {}
def build(params):
    a = trimesh.creation.box(extents=(10, 10, 10), transform=translation_matrix([0, 0, 0]))
    b = trimesh.creation.box(extents=(10, 10, 10), transform=translation_matrix([8, 0, 0]))
    return union_all([a, b])
DIMENSIONS = {}
REGIONS = {"load": {"min": [-5, -5, -5], "max": [5, 5, 5]}, "mounts": [{"name": "a", "min": [-5, -5, -5], "max": [5, 5, 5]}]}
NOTES = []
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.py"
            path.write_text(script)
            result = run_script(path, Path(tmp), "part", {"max_protrusion_mm": 40.0})
        self.assertTrue(result.get("ok"), result)
        self.assertTrue(result.get("watertight"))
        self.assertEqual(result.get("connected_components"), 1)


class BedHandlePromptTests(unittest.TestCase):
    def test_bed_handle_prompt_has_no_cupholder_template(self) -> None:
        req = UserRequirements(
            description="bed handle",
            user_message="Design a bed handle that attaches to the wall on 3 points",
            task_kind=TASK_BED_HANDLE,
        )
        req.payload.filled_mass_kg = 80.0
        req.attachment.method = "adhesive"
        req.attachment.allowed_contact_region = "wall"
        req.task_answers = {"required_reach_mm": 50.0, "handle_location": "bedside"}
        req.design_envelope.max_protrusion_mm = 150.0
        prompt = combined_prompt(req, "grok_warmstart")
        self.assertIn("bed_handle", prompt)
        self.assertNotIn("cup_radius_mm", prompt)
        self.assertNotIn("desk_thickness_mm", prompt)
        self.assertNotIn("cup diameter", prompt.lower())
        self.assertIn("union_all", prompt)
        self.assertIn("concatenate", prompt)
        self.assertIn("mounting block", prompt.lower())
        self.assertTrue(script_uses_concatenate_as_final_assembly(
            "part = trimesh.util.concatenate(meshes)\nreturn part\n"
        ))
        self.assertFalse(script_uses_concatenate_as_final_assembly(
            "part = union_all(meshes)\nreturn part\n"
        ))


class TaskAwareFeasibilityTests(unittest.TestCase):
    def test_bed_handle_does_not_require_bottle_or_desk(self) -> None:
        req = UserRequirements(
            description="bed handle",
            user_message="Design a handle to help a person get up from bed",
            task_kind=TASK_BED_HANDLE,
        )
        req.payload.filled_mass_kg = 80.0
        req.attachment.allowed_contact_region = "wall"
        req.design_envelope.max_protrusion_mm = 50.0
        geom = _geom(req)
        result = check_design_feasibility(req, geom)
        self.assertTrue(result.feasible, result.message)
        blob = result.message.lower()
        self.assertNotIn("payload diameter", blob)
        self.assertNotIn("desk thickness", blob)

    def test_cup_holder_legacy_messages_still_used_when_values_are_zero(self) -> None:
        req = _cupholder_req(diameter=0.0, desk=0.0)
        geom = _geom(req)
        result = check_design_feasibility(req, geom)
        self.assertFalse(result.feasible)
        self.assertIn("Payload diameter must be positive and physically nonzero", result.message)
        self.assertIn("Desk thickness must be positive and physically nonzero", result.message)

    def test_cup_holder_still_requires_diameter_and_desk(self) -> None:
        req = _cupholder_req(diameter=None, desk=None)
        req.object_geometry.bottle_diameter_mm = None
        req.environment.desk_thickness_mm = None
        geom = _geom(req)
        result = check_design_feasibility(req, geom)
        self.assertFalse(result.feasible)
        codes = {v.code for v in result.violations}
        self.assertTrue(
            {"payload_diameter_unavailable", "nonpositive_payload_diameter"} & codes
        )
        self.assertTrue(
            {"desk_thickness_unavailable", "nonpositive_desk_thickness"} & codes
        )

    def test_desk_hook_requires_desk_not_bottle(self) -> None:
        req = UserRequirements(
            user_message="Design a hook for a bag on my desk",
            task_kind=TASK_DESK_HOOK,
        )
        req.payload.filled_mass_kg = 2.0
        req.environment.desk_thickness_mm = 24.0
        req.design_envelope.max_protrusion_mm = 80.0
        geom = _geom(req)
        result = check_design_feasibility(req, geom)
        self.assertTrue(result.feasible, result.message)
        req.environment.desk_thickness_mm = None
        geom2 = _geom(req)
        bad = check_design_feasibility(req, geom2)
        self.assertFalse(bad.feasible)
        self.assertFalse(any("diameter" in (v.message or "").lower() for v in bad.violations))

    def test_wall_shelf_does_not_require_desk_or_bottle_diameter(self) -> None:
        req = UserRequirements(
            user_message="Design a wall-mounted shelf for a router",
            task_kind=TASK_WALL_SHELF,
        )
        req.payload.filled_mass_kg = 3.0
        req.task_answers = {"payload_size_mm": 200.0}
        req.design_envelope.max_protrusion_mm = 180.0
        geom = _geom(req)
        result = check_design_feasibility(req, geom)
        self.assertTrue(result.feasible, result.message)
        self.assertFalse(any("desk" in (v.message or "").lower() for v in result.violations))
        self.assertFalse(any("bottle" in (v.message or "").lower() for v in result.violations))


class WarmStartFailureStateTests(unittest.TestCase):
    def test_failed_generation_is_not_request_information(self) -> None:
        calls = {"n": 0, "topology": 0}

        def _gen(req):
            calls["n"] += 1
            raise RuntimeError("AttributeError: 'Trimesh' object has no attribute 'remove_duplicate_faces'")

        orch = Orchestrator(
            fixtures=IntegrationFixtures(
                registration=None, geometry=None, analysis=None, topology=None, cad=None
            ),
            warm_start_generator=_gen,
        )
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(_bed_handle_update())
        self.assertEqual(missing_requirement_fields(orch.state.requirements), [])
        orch.run()
        self.assertEqual(missing_requirement_fields(orch.state.requirements), [])
        self.assertNotEqual(orch.state.stage, WorkflowStage.REQUEST_INFORMATION)
        self.assertTrue(orch.warm_start_error)
        self.assertIsNone(orch.state.topology)
        self.assertIsNotNone(orch.state.geometry)
        self.assertIsNone(orch.imported_candidate)
        card = detect_ui_failure(orch.state, warm_start_error=orch.warm_start_error)
        self.assertIsNotNone(card)
        self.assertEqual(card.stage, STAGE_WARM_START)
        self.assertEqual(card.headline, "Starting design needs revision")
        self.assertEqual(card.primary_label, "Try again")
        self.assertEqual(card.log_label, "View full log")
        self.assertIsNone(card.secondary_label)

    def test_answers_preserved_across_generation_failure(self) -> None:
        orch = Orchestrator(
            fixtures=IntegrationFixtures(
                registration=None, geometry=None, analysis=None, topology=None, cad=None
            ),
            warm_start_generator=lambda req: None,
        )
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(_bed_handle_update())
        req_before = orch.state.requirements.model_copy(deep=True)
        orch.run()
        req = orch.state.requirements
        self.assertEqual(req.payload.filled_mass_kg, req_before.payload.filled_mass_kg)
        self.assertEqual(req.attachment.method, "adhesive")
        self.assertEqual(req.attachment.allowed_contact_region, "wall")
        self.assertEqual(req.design_envelope.max_protrusion_mm, 50.0)
        self.assertEqual((req.task_answers or {}).get("handle_location"), "bedside")

    def test_topology_not_called_without_validated_warm_start(self) -> None:
        orch = Orchestrator(
            fixtures=IntegrationFixtures(
                registration=None, geometry=None, analysis=None, topology=None, cad=None
            ),
            warm_start_generator=lambda req: None,
        )
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(_bed_handle_update())
        orch.run()
        self.assertIsNone(orch.state.topology)
        self.assertIsNone(orch.imported_candidate)
        self.assertNotEqual(orch.state.stage, WorkflowStage.TOPOLOGY_OPTIMIZATION)

    def test_validated_warm_start_advances_past_geometry(self) -> None:
        candidate = ImportedCandidateGeometry(
            mesh_path="/tmp/ok.stl",
            task="generated",
            candidate_name="grok_warmstart",
            watertight=True,
            bbox_min_mm=(0.0, -10.0, 0.0),
            bbox_max_mm=(40.0, 10.0, 30.0),
        )
        orch = Orchestrator(
            fixtures=IntegrationFixtures(
                registration=None, geometry=None, analysis=None, topology=None, cad=None
            ),
            warm_start_generator=lambda req: candidate,
        )
        orch.ingest_user_request("Design a handle to help a person get up from bed")
        orch.apply_answers(_bed_handle_update())
        orch._handle_geometry()
        self.assertEqual(orch.imported_candidate, candidate)
        self.assertEqual(fit_family_for(candidate), "generated")
        self.assertEqual(orch.state.stage, WorkflowStage.CANDIDATE_FIT)
        self.assertIsNone(orch.state.topology)


if __name__ == "__main__":
    unittest.main()
