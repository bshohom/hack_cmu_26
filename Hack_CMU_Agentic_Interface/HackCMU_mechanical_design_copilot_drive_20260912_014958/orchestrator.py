"""Deterministic state machine. Owns DesignState transitions.

LLM reasoning happens inside a stage. Tools perform numerical/geometric work.
Agents never pass natural-language messages to each other.

Optional IntegrationFixtures stand in for teammate tool outputs. They are not
implementations of registration, FEM, or topology optimization.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from agents.design_review import review_design
from agents.feasibility import check_design_feasibility, feasibility_questions
from agents.geometry import GeometryAgent
from agents.interaction import InteractionAgent, missing_requirement_fields
from agents.structure import StructureAgent
from reasoning import run_reasoning_agent
from imported_candidate import candidate_fit_questions, check_candidate_fit, fit_family_for
from schemas import (
    AnalysisInput,
    AnalysisOutput,
    CadInput,
    DesignIteration,
    GeometryInput,
    ImportedCandidateGeometry,
    IntegrationFixtures,
    InteractionDecision,
    MassProvenance,
    MissingInformation,
    ReasoningRole,
    RequirementsUpdate,
    ResolvedPayloadMass,
    ReviewDecision,
    SafetyStatus,
    StructureInput,
    TopologyInput,
    TopologySolverOptions,
    TraceEvent,
    UserRequirements,
    VerificationResult,
    WorkflowStage,
)
from state import DesignState
from tools.analysis import run_analysis
from tools.cad import generate_cad
from tools.topology import TopologyUnavailable, run_topology_optimization

MAX_STRUCTURE_ITERATIONS = 4

# Used when the requirements do not state one. FDM parts are printed with layer-adhesion
# weakness already folded into the material yield, so this is a design margin, not a code.
DEFAULT_REQUIRED_FOS = 1.5


@dataclass
class _Verdict:
    passed: bool
    summary: str

_STOP_STAGES: Set[WorkflowStage] = {
    WorkflowStage.COMPLETE,
    WorkflowStage.REJECTED,
    WorkflowStage.REQUEST_INFORMATION,
    WorkflowStage.DESIGN_REVIEW_FAILED,
    WorkflowStage.TOPOLOGY_FAILED,
    WorkflowStage.VERIFICATION_FAILED,
}

_TRACE_FIELDS = [
    "stage",
    "safety_status",
    "interaction_decision",
    "requirements",
    "missing_information",
    "registration",
    "geometry",
    "imported_candidate",
    "candidate_fit",
    "feasibility",
    "structure",
    "analysis",
    "design_review",
    "design_iterations",
    "topology",
    "cad",
    "verification",
    "resolved_payload_mass",
    "reject_reason",
    "contract_error",
]

_FLOAT_ABS_TOL = 1e-3


class Orchestrator:
    """Drives REQUIREMENTS -> GEOMETRY -> FEASIBILITY_CHECK -> STRUCTURE ->
    ANALYSIS -> DESIGN_REVIEW -> (STRUCTURE refinement)* ->
    TOPOLOGY_OPTIMIZATION -> VERIFICATION -> COMPLETE.
    """

    def __init__(
        self,
        fixtures: Optional[IntegrationFixtures] = None,
        max_structure_iterations: int = MAX_STRUCTURE_ITERATIONS,
        analysis_never_pass: bool = False,
        imported_candidate: Optional[ImportedCandidateGeometry] = None,
        topology_options: Optional[TopologySolverOptions] = None,
        topology_log=None,
        topology_progress=None,
        warm_start_generator=None,
        run_trace=None,
    ) -> None:
        self.state = DesignState()
        self.fixtures = fixtures or IntegrationFixtures()
        self.trace: List[TraceEvent] = []
        self.interaction = InteractionAgent()
        self.geometry = GeometryAgent()
        self.structure = StructureAgent()
        self.max_structure_iterations = max_structure_iterations
        self.analysis_never_pass = analysis_never_pass
        self.imported_candidate = imported_candidate
        self.topology_options = topology_options
        self.topology_log = topology_log
        self.topology_progress = topology_progress
        # Callable[[UserRequirements], Optional[ImportedCandidateGeometry]]: generates the
        # warm-start candidate (e.g. Grok-written trimesh script) when none was imported.
        self.warm_start_generator = warm_start_generator
        self.warm_start_error: Optional[str] = None
        self.run_trace = run_trace
        self._handlers = {
            WorkflowStage.REQUIREMENTS: self._handle_requirements,
            WorkflowStage.REQUEST_INFORMATION: self._handle_request_information,
            WorkflowStage.GEOMETRY: self._handle_geometry,
            WorkflowStage.CANDIDATE_FIT: self._handle_candidate_fit,
            WorkflowStage.FEASIBILITY_CHECK: self._handle_feasibility,
            WorkflowStage.STRUCTURE: self._handle_structure,
            WorkflowStage.ANALYSIS: self._handle_analysis,
            WorkflowStage.DESIGN_REVIEW: self._handle_design_review,
            WorkflowStage.TOPOLOGY_OPTIMIZATION: self._handle_topology,
            WorkflowStage.VERIFICATION: self._handle_verification,
            WorkflowStage.COMPLETE: self._handle_complete,
            WorkflowStage.REJECTED: self._handle_complete,
            WorkflowStage.DESIGN_REVIEW_FAILED: self._handle_complete,
            WorkflowStage.TOPOLOGY_FAILED: self._handle_complete,
            WorkflowStage.VERIFICATION_FAILED: self._handle_complete,
        }

    def _emit(self, event: str, detail: str = "") -> None:
        line = f"[RUN] {event}" + (f" {detail}" if detail else "")
        print(line, flush=True)
        cb = self.run_trace
        if cb is not None:
            cb(event, detail)

    def ingest_user_request(self, message: str) -> DesignState:
        before = self._snapshot()
        executed = self.state.stage
        run_reasoning_agent(ReasoningRole.INTERACTION, self.state)
        result = self.interaction.assess(message, self.state.requirements)
        self._apply_interaction(result)
        self._record(
            executed,
            "InteractionAgent",
            "ingest_user_request",
            before,
            notes=result.reject_reason or "",
        )
        return self.state

    def apply_answers(self, update: RequirementsUpdate) -> DesignState:
        before = self._snapshot()
        executed = self.state.stage
        run_reasoning_agent(ReasoningRole.INTERACTION, self.state)
        req = self.state.requirements
        if req is None:
            req = UserRequirements()
        result = self.interaction.apply_update(req, update)
        self._apply_interaction(result)
        self._record(executed, "InteractionAgent", "apply_answers", before)
        return self.state

    def step(self) -> DesignState:
        before = self._snapshot()
        executed = self.state.stage
        handler = self._handlers[executed]
        handler()
        component, action, notes = self._step_trace_meta(executed)
        self._record(executed, component, action, before, notes=notes)
        return self.state

    def run(self) -> DesignState:
        """Advance until complete, rejected, blocked, or a contract error."""
        for _ in range(32):
            if self.state.stage in _STOP_STAGES:
                return self.state
            before = self.state.stage
            self.step()
            if self.state.stage == before:
                return self.state
            if self.state.contract_error:
                return self.state
        return self.state

    def retry_topology(self) -> DesignState:
        """Rebuild/retry topology from the existing validated candidate.

        Does not clear geometry, regions, or the warm-start path. Does not call
        the warm-start generator when a candidate is already attached.
        """
        print("[RUN] RETRY_OPTIMIZATION", flush=True)
        if self.imported_candidate is None:
            existing = getattr(self.state, "imported_candidate", None)
            if existing is not None:
                self.imported_candidate = existing
        if self.imported_candidate is not None:
            self.state.imported_candidate = self.imported_candidate
            path = getattr(self.imported_candidate, "mesh_path", "") or ""
            print(f"[RUN] reusing warm start path = {path}", flush=True)
        self.state.topology = None
        self.state.notes = ""
        self.state.contract_error = None
        if (
            self.state.geometry is not None
            and self.state.structure is not None
            and self.state.analysis is not None
        ):
            self.state.stage = WorkflowStage.TOPOLOGY_OPTIMIZATION
        elif self.state.geometry is None:
            self.state.stage = WorkflowStage.GEOMETRY
        elif self.state.structure is None:
            self.state.stage = WorkflowStage.STRUCTURE
        else:
            self.state.stage = WorkflowStage.ANALYSIS
        return self.run()

    def _apply_interaction(self, result) -> None:
        self.state.requirements = result.requirements
        self.state.interaction_decision = result.decision
        self.state.clarifications = result.questions
        self.state.missing_information = [
            MissingInformation(field=q.field, reason=q.question, priority=q.priority)
            for q in result.questions
        ]
        self.state.vision_request = result.vision_request

        if result.decision == InteractionDecision.REJECT_OR_ESCALATE:
            self.state.stage = WorkflowStage.REJECTED
            self.state.safety_status = SafetyStatus.REJECTED
            self.state.reject_reason = result.reject_reason
            return
        if result.decision == InteractionDecision.REQUEST_INFORMATION:
            self.state.stage = WorkflowStage.REQUEST_INFORMATION
            return
        self.state.stage = WorkflowStage.GEOMETRY
        self.state.clarifications = []
        self.state.missing_information = []
        self.state.contract_error = None

    def _request_info(self, note: str) -> None:
        self.state.stage = WorkflowStage.REQUEST_INFORMATION
        self.state.interaction_decision = InteractionDecision.REQUEST_INFORMATION
        self.state.notes = note
        req = self.state.requirements
        if req is None:
            print(
                f"[GEOMETRY] returning request_information because = {note} "
                "(no requirements)",
                flush=True,
            )
            return
        result = self.interaction._decide(req)
        if result.questions:
            self.state.clarifications = result.questions
            self.state.missing_information = [
                MissingInformation(field=q.field, reason=q.question, priority=q.priority)
                for q in result.questions
            ]
        fields = [getattr(q, "field", None) for q in (self.state.clarifications or [])]
        print(
            f"[GEOMETRY] returning request_information because = {note} "
            f"missing_fields = {fields}",
            flush=True,
        )

    def _block_contract(self, message: str) -> None:
        self.state.contract_error = message
        self.state.safety_status = SafetyStatus.NEEDS_REVIEW
        self.state.notes = message

    def _handle_requirements(self) -> None:
        if self.state.requirements is None:
            self._request_info("No user request yet.")
            return
        result = self.interaction.assess(
            self.state.requirements.user_message or self.state.requirements.description,
            self.state.requirements,
        )
        self._apply_interaction(result)

    def _handle_request_information(self) -> None:
        return

    def _handle_geometry(self) -> None:
        req = self.state.requirements
        gaps = self._requirements_gaps()
        print("[GEOMETRY] ENTER", flush=True)
        print(
            "[GEOMETRY] requirements = "
            f"task={getattr(req, 'task_kind', '')!r} "
            f"mass={getattr(getattr(req, 'payload', None), 'filled_mass_kg', None)} "
            f"diameter={getattr(getattr(req, 'object_geometry', None), 'bottle_diameter_mm', None)} "
            f"desk={getattr(getattr(req, 'environment', None), 'desk_thickness_mm', None)} "
            f"method={getattr(getattr(req, 'attachment', None), 'method', None)!r} "
            f"region={getattr(getattr(req, 'attachment', None), 'allowed_contact_region', None)!r} "
            f"reach={((getattr(req, 'task_answers', None) or {}).get('required_reach_mm'))} "
            f"protrusion={getattr(getattr(req, 'design_envelope', None), 'max_protrusion_mm', None)} "
            f"mfg={getattr(getattr(req, 'manufacturing', None), 'method', None)!r} "
            f"answers={getattr(req, 'task_answers', None)}",
            flush=True,
        )
        print(f"[GEOMETRY] missing_fields = {gaps}", flush=True)
        print(
            "[GEOMETRY] clarification_questions = "
            f"{[getattr(q, 'field', None) for q in (self.state.clarifications or [])]}",
            flush=True,
        )
        if gaps:
            print(f"[GEOMETRY] blocker = requirements_gaps {gaps}", flush=True)
            print(
                "[GEOMETRY] returning request_information because = "
                "Geometry needs complete requirements",
                flush=True,
            )
            self._request_info("Geometry needs complete requirements.")
            return
        run_reasoning_agent(ReasoningRole.GEOMETRY, self.state)
        assert self.state.requirements is not None
        if self.fixtures.registration is not None:
            self.state.registration = self.fixtures.registration
        if self.fixtures.geometry is not None:
            self.state.geometry = self.fixtures.geometry
            if self.state.registration is None:
                print(
                    "[GEOMETRY] blocker = fixture geometry requires registration",
                    flush=True,
                )
                self._block_contract(
                    "Contract/integration error: RegistrationOutput is required "
                    "before external GeometryOutput can advance to STRUCTURE."
                )
                return
        else:
            self.state.geometry = self.geometry.run(
                GeometryInput(
                    requirements=self.state.requirements,
                    registration=self.state.registration,
                    registration_frame=(
                        self.state.registration.frame_id
                        if self.state.registration
                        else None
                    ),
                )
            )
        mismatch = self._requirement_geometry_mismatch()
        if mismatch:
            self._block_contract(mismatch)
            return
        frame_error = self._frame_mismatch()
        if frame_error:
            self._block_contract(frame_error)
            return
        if self.imported_candidate is not None:
            self._emit("WARM_START_PRESENT", "existing candidate attached")
        if self.imported_candidate is None and self.warm_start_generator is not None:
            print("[GEOMETRY] GROK_CALL_SITE_REACHED", flush=True)
            self.warm_start_error = None
            try:
                generated = self.warm_start_generator(self.state.requirements)
            except Exception as exc:  # noqa: BLE001 — generation failure never blocks the workflow
                generated = None
                self.warm_start_error = f"{type(exc).__name__}: {exc}"
            if generated is not None:
                self.imported_candidate = generated
                self._emit("WARM_START_PRESENT", "generated this run")
            else:
                self.warm_start_error = (
                    self.warm_start_error
                    or "Warm-start generation returned no validated mesh"
                )
                self.state.notes = (
                    "Warm-start generation failed. "
                    f"{self.warm_start_error}"
                ).strip()
                print("[RUN] failure stage = warm_start_generation_failed", flush=True)
                return
        if self.imported_candidate is not None:
            self.state.stage = WorkflowStage.CANDIDATE_FIT
            return
        self.state.stage = WorkflowStage.FEASIBILITY_CHECK

    def _handle_candidate_fit(self) -> None:
        if self.state.requirements is None:
            self._request_info("Candidate fit needs complete requirements.")
            return
        candidate = self.imported_candidate
        if candidate is None:
            self.state.stage = WorkflowStage.FEASIBILITY_CHECK
            return
        self.state.imported_candidate = candidate
        result = check_candidate_fit(self.state.requirements, candidate)
        self.state.candidate_fit = result
        if result.fits:
            self.state.stage = WorkflowStage.FEASIBILITY_CHECK
            return
        questions = candidate_fit_questions(result, family=fit_family_for(candidate))
        print(
            "[GEOMETRY] returning request_information because = candidate does not fit "
            f"fields={[getattr(q, 'field', None) for q in questions]}",
            flush=True,
        )
        self.state.stage = WorkflowStage.REQUEST_INFORMATION
        self.state.interaction_decision = InteractionDecision.REQUEST_INFORMATION
        self.state.clarifications = questions
        self.state.missing_information = [
            MissingInformation(field=q.field, reason=q.question, priority=q.priority)
            for q in questions
        ]
        self.state.notes = result.message
        self.state.structure = None
        self.state.analysis = None
        self.state.design_review = None
        self.state.topology = None

    def _handle_feasibility(self) -> None:
        if self.warm_start_generator is not None and self.imported_candidate is None:
            print("[RUN] failure stage = warm_start_generation_failed", flush=True)
            return
        if self.state.geometry is None or self.state.requirements is None:
            self._request_info("Feasibility check needs geometry and requirements.")
            return
        result = check_design_feasibility(self.state.requirements, self.state.geometry)
        self.state.feasibility = result
        if result.feasible:
            self.state.stage = WorkflowStage.STRUCTURE
            return
        questions = feasibility_questions(result)
        self.state.stage = WorkflowStage.REQUEST_INFORMATION
        self.state.interaction_decision = InteractionDecision.REQUEST_INFORMATION
        self.state.clarifications = questions
        self.state.missing_information = [
            MissingInformation(field=q.field, reason=q.question, priority=q.priority)
            for q in questions
        ]
        self.state.notes = result.message
        self.state.structure = None
        self.state.analysis = None
        self.state.design_review = None
        self.state.topology = None

    def _handle_structure(self) -> None:
        if self.state.geometry is None or self.state.requirements is None:
            self._request_info("Structure needs geometry and requirements.")
            return
        frame_error = self._frame_mismatch()
        if frame_error:
            self._block_contract(frame_error)
            return
        resolved = self._resolve_payload_mass()
        if resolved is None:
            return
        self.state.resolved_payload_mass = resolved
        run_reasoning_agent(ReasoningRole.STRUCTURE, self.state)
        previous = None
        review = None
        if (
            self.state.design_review is not None
            and self.state.design_review.decision == ReviewDecision.REVISE
            and self.state.structure is not None
        ):
            previous = self.state.structure
            review = self.state.design_review
        self.state.structure = self.structure.run(
            StructureInput(
                requirements=self.state.requirements,
                geometry=self.state.geometry,
                payload_mass_kg=resolved.mass_kg,
                payload_mass_provenance=resolved.provenance,
                previous_structure=previous,
                review=review,
            )
        )
        self.state.stage = WorkflowStage.ANALYSIS

    def _handle_analysis(self) -> None:
        if self.state.geometry is None or self.state.structure is None:
            self._request_info("Analysis needs geometry and structure.")
            return
        req = self.state.requirements
        material = "PLA"
        if req and req.manufacturing.material:
            material = req.manufacturing.material
        if self.state.geometry.material and req and req.manufacturing.material:
            if self.state.geometry.material != req.manufacturing.material:
                self._block_contract(
                    "Contract/integration error: material disagreement "
                    f"requirements={req.manufacturing.material!r} "
                    f"geometry={self.state.geometry.material!r}."
                )
                return

        run_reasoning_agent(ReasoningRole.ANALYSIS_INTERPRETATION, self.state)
        primary = self.state.structure.load_cases[0]
        if self.fixtures.analysis is not None:
            mismatch = self._analysis_load_case_mismatch(self.fixtures.analysis)
            if mismatch:
                self._block_contract(mismatch)
                return
            self.state.analysis = self.fixtures.analysis
        else:
            inp = AnalysisInput(
                geometry=self.state.geometry,
                structure=self.state.structure,
                load_case_id=primary.load_case_id,
                load_force_N=primary.force_N,
                material=material,
                boundary_conditions=self.state.structure.boundary_conditions,
                load_cases=self.state.structure.load_cases,
            )
            self.state.analysis = run_analysis(inp, never_pass=self.analysis_never_pass)
        self.state.safety_status = SafetyStatus.UNVERIFIED
        self.state.notes = self.state.analysis.disclaimer
        self.state.stage = WorkflowStage.DESIGN_REVIEW

    def _handle_design_review(self) -> None:
        if self.state.structure is None or self.state.analysis is None:
            self._block_contract(
                "Contract/integration error: DESIGN_REVIEW requires "
                "StructureOutput and AnalysisOutput."
            )
            return
        run_reasoning_agent(ReasoningRole.DESIGN_REVIEW, self.state)
        review = review_design(self.state.structure, self.state.analysis)
        self.state.design_review = review
        self.state.design_iterations.append(
            DesignIteration(
                iteration=self.state.structure.iteration,
                structure=self.state.structure.model_copy(deep=True),
                analysis=self.state.analysis.model_copy(deep=True),
                review=review,
            )
        )
        if review.decision == ReviewDecision.PASS:
            self.state.stage = WorkflowStage.TOPOLOGY_OPTIMIZATION
            return
        if self.state.structure.iteration + 1 >= self.max_structure_iterations:
            self.state.stage = WorkflowStage.DESIGN_REVIEW_FAILED
            self.state.notes = (
                "DESIGN_REVIEW_FAILED: iteration budget exhausted without PASS. "
                "Topology optimization was not started. "
                "Not physical safety certification."
            )
            return
        self.state.stage = WorkflowStage.STRUCTURE

    def _handle_topology(self) -> None:
        if self.warm_start_generator is not None and self.imported_candidate is None:
            self._emit("BLOCKED: topology needs a validated warm-start mesh")
            print("[RUN] failure stage = warm_start_generation_failed", flush=True)
            return
        if self.state.analysis is None:
            self._emit("BLOCKED: TOPOLOGY_OPTIMIZATION requires AnalysisOutput")
            self._block_contract(
                "Contract/integration error: TOPOLOGY_OPTIMIZATION requires "
                "a valid AnalysisOutput."
            )
            return
        if self.state.geometry is None or self.state.structure is None:
            self._emit("BLOCKED: topology needs geometry and structure")
            self._request_info("Topology optimization needs geometry and structure.")
            return
        req = self.state.requirements
        material = "PLA"
        max_mass: Optional[float] = None
        if req:
            material = req.manufacturing.material or "PLA"
            max_mass = req.part_mass.max_part_mass_kg
        if self.fixtures.topology is not None:
            self._emit("TO_PROBLEM_BUILD_STARTED", "fixture topology")
            self.state.topology = self.fixtures.topology
            self._emit("TO_PROBLEM_BUILD_FINISHED")
            self._emit("TO_SOLVER_STARTED", "fixture")
            self._emit("TO_SOLVER_FINISHED")
            self._emit("TO_RESULT_ACCEPTED", "fixture topology")
        else:
            self._emit("TO_PROBLEM_BUILD_STARTED")
            inp = TopologyInput(
                design_domain=self.state.geometry.part,
                fixed_regions=self.state.structure.attachment_regions,
                load_regions=self.state.structure.load_regions,
                loads=self.state.structure.load_cases,
                material=material,
                target_volume_fraction=0.4,
                max_part_mass_kg=max_mass,
                candidate=self.state.imported_candidate,
                desk_thickness_mm=self.state.geometry.environment.desk_thickness_mm,
                solver_options=self.topology_options,
                envelope=self.state.geometry.design_envelope,
                attachment_method=(req.attachment.method if req else None),
                payload_kind=self.state.geometry.payload_object.kind,
                payload_size_mm=self.state.geometry.payload_object.bottle_diameter_mm,
                structure=self.state.structure,
            )
            self._emit("TO_PROBLEM_BUILD_FINISHED")
            self._emit("TO_SOLVER_STARTED")
            try:
                self.state.topology = run_topology_optimization(
                    inp, log=self.topology_log, progress=self.topology_progress
                )
            except TopologyUnavailable as exc:
                print("[RUN] EXCEPTION at TO_SOLVER", flush=True)
                traceback.print_exc()
                self.state.stage = WorkflowStage.TOPOLOGY_FAILED
                self.state.safety_status = SafetyStatus.NEEDS_REVIEW
                self.state.notes = (
                    f"TOPOLOGY_FAILED: live optimization did not produce a result. {exc} "
                    "No CAD artifact was written and the run did not continue on a placeholder."
                )
                return
            self._emit("TO_SOLVER_FINISHED")
            self._emit("RESULT_STORED")
        if self.state.analysis.is_mock or self.state.topology.is_mock:
            self.state.safety_status = SafetyStatus.UNVERIFIED
        self.state.stage = WorkflowStage.VERIFICATION

    def _handle_verification(self) -> None:
        run_reasoning_agent(ReasoningRole.DESIGN_REVIEW, self.state)
        analysis_mock = self.state.analysis.is_mock if self.state.analysis is not None else True
        topology_mock = self.state.topology.is_mock if self.state.topology is not None else True
        artifacts_ok = all(
            [
                self.state.geometry is not None,
                self.state.structure is not None,
                self.state.analysis is not None,
                self.state.topology is not None,
            ]
        )
        if artifacts_ok and self.state.geometry is not None and self.state.topology is not None:
            if self.fixtures.cad is not None:
                self.state.cad = self.fixtures.cad
            else:
                self.state.cad = generate_cad(
                    CadInput(geometry=self.state.geometry, topology=self.state.topology)
                )
        # "Artifacts complete" means files exist on disk, not that Python objects exist.
        missing = self._missing_artifact_files()
        artifacts_ok = artifacts_ok and self.state.cad is not None and not missing

        mocked = analysis_mock or topology_mock or (
            self.state.cad.is_mock if self.state.cad is not None else True
        )
        if mocked:
            reason = "mock analysis data" if analysis_mock else "analysis is not a safety certification"
            if topology_mock:
                reason += "; mock topology data"
        else:
            reason = "this prototype does not certify physical safety"

        if not artifacts_ok:
            detail = (
                f"Artifact file(s) named by the run are not on disk: {'; '.join(missing)}. "
                if missing
                else "Required prototype artifacts are missing. "
            )
            self.state.verification = VerificationResult(
                artifacts_complete=False,
                analysis_is_mock=analysis_mock,
                topology_is_mock=topology_mock,
                safety_validated=False,
                notes=detail + "stage remains VERIFICATION. NOT validated for physical use.",
            )
            self.state.safety_status = SafetyStatus.NEEDS_REVIEW
            return

        # The post-optimization FE check and the mesh acceptance checks are the real gate.
        gate = self._acceptance_verdict()
        reason += "; " + gate.summary

        if not gate.passed and not topology_mock:
            self.state.verification = VerificationResult(
                artifacts_complete=True,
                analysis_is_mock=analysis_mock,
                topology_is_mock=topology_mock,
                safety_validated=False,
                notes=(
                    f"VERIFICATION_FAILED: {gate.summary}. "
                    "An artifact was produced but does not meet the accepted requirements. "
                    "NOT validated for physical use."
                ),
            )
            self.state.safety_status = SafetyStatus.NEEDS_REVIEW
            self.state.stage = WorkflowStage.VERIFICATION_FAILED
            self.state.notes = gate.summary
            return

        self.state.verification = VerificationResult(
            artifacts_complete=True,
            analysis_is_mock=analysis_mock,
            topology_is_mock=topology_mock,
            safety_validated=False,
            notes=(
                "engineering validation: UNVERIFIED. "
                f"reason: {reason}. "
                "COMPLETE means the artifact exists and passed the acceptance checks above "
                "under idealized boundary conditions, not physical safety. "
                "NOT validated for physical use."
            ),
        )
        self.state.safety_status = SafetyStatus.UNVERIFIED
        self.state.stage = WorkflowStage.COMPLETE

    def _missing_artifact_files(self) -> List[str]:
        """Artifact paths this run claims but that are not present and non-empty on disk."""
        missing: List[str] = []
        topology = self.state.topology
        if topology is not None and not topology.is_mock:
            ref = Path(topology.optimized_geometry_ref or "")
            if not (ref.is_file() and ref.stat().st_size > 0):
                missing.append(f"optimized geometry {topology.optimized_geometry_ref!r}")
        cad = self.state.cad
        if cad is not None and not cad.is_mock:
            path = Path(cad.filename or "")
            if not (path.is_file() and path.stat().st_size > 0):
                missing.append(f"CAD export {cad.filename!r}")
        return missing

    def _acceptance_verdict(self) -> "_Verdict":
        """Combine the mesh acceptance checks with the post-TO FE check.

        Unknown is not a pass: a check that could not run blocks completion just as a failed
        one does, because "we did not look" and "we looked and it was fine" are different
        claims and only one of them justifies handing someone a part to print.
        """
        topology = self.state.topology
        if topology is None:
            return _Verdict(False, "no topology result")
        if topology.is_mock:
            return _Verdict(True, "mock topology: no acceptance checks apply")

        problems: List[str] = []
        acceptance = topology.acceptance or {}
        if not acceptance:
            problems.append("mesh acceptance checks did not run")
        else:
            if acceptance.get("failed"):
                problems.append("failed mesh checks: " + ", ".join(acceptance["failed"]))
            if acceptance.get("unknown"):
                problems.append("unverifiable mesh checks: " + ", ".join(acceptance["unknown"]))

        check = topology.post_check
        if check is None:
            problems.append("post-TO FE check did not run, so strength is unknown")
            fe = ""
        else:
            fos = check.factor_of_safety
            required = self._required_factor_of_safety()
            fos_text = f"{fos:.2f}" if fos is not None else "n/a"
            fe = (
                f"post-TO linear FE at nominal load: max displacement "
                f"{check.max_displacement_mm:.3f} mm, max stress {check.max_stress_pa / 1e6:.2f} MPa, "
                f"factor of safety {fos_text} (required {required})"
            )
            if fos is None:
                problems.append("factor of safety could not be computed (no yield strength)")
            elif fos < required:
                problems.append(f"factor of safety {fos:.2f} is below the required {required}")

        if topology.unsupported_requirements:
            names = ", ".join(str(u.get("requirement")) for u in topology.unsupported_requirements)
            problems.append(f"requirements accepted but not applied: {names}")

        idealizations = (
            "FE idealizations: thresholded voxels with residual void stiffness, supports fully "
            "fixed in x/y/z, peak stress smoothed over element size"
        )
        if problems:
            return _Verdict(False, "; ".join(problems) + (f". {fe}" if fe else ""))
        return _Verdict(True, f"{fe}. {idealizations}" if fe else idealizations)

    def _required_factor_of_safety(self) -> float:
        req = self.state.requirements
        value = getattr(getattr(req, "safety", None), "min_factor_of_safety", None) if req else None
        return float(value) if value else DEFAULT_REQUIRED_FOS

    def _handle_complete(self) -> None:
        return

    def _requirements_gaps(self) -> List[str]:
        """Same missing-field list the InteractionAgent uses. One readiness contract."""
        req = self.state.requirements
        if req is None:
            return ["requirements"]
        return missing_requirement_fields(req)

    def _requirements_ready(self) -> bool:
        return not self._requirements_gaps()

    def _requirement_geometry_mismatch(self) -> Optional[str]:
        req = self.state.requirements
        geom = self.state.geometry
        if req is None or geom is None:
            return None
        checks: List[Tuple[str, Optional[float], Optional[float]]] = [
            (
                "payload diameter_mm",
                req.object_geometry.bottle_diameter_mm,
                geom.payload_object.bottle_diameter_mm,
            ),
            (
                "payload filled_mass_kg",
                req.payload.filled_mass_kg,
                geom.payload_object.filled_mass_kg,
            ),
            (
                "desk/support thickness_mm",
                req.environment.desk_thickness_mm,
                geom.environment.desk_thickness_mm,
            ),
            (
                "design envelope max_protrusion_mm",
                req.design_envelope.max_protrusion_mm,
                geom.design_envelope.max_protrusion_mm,
            ),
            (
                "design envelope max_width_mm",
                req.design_envelope.max_width_mm,
                geom.design_envelope.max_width_mm,
            ),
            (
                "design envelope max_height_mm",
                req.design_envelope.max_height_mm,
                geom.design_envelope.max_height_mm,
            ),
        ]
        for label, left, right in checks:
            if left is None or right is None:
                continue
            if not self._numbers_agree(left, right):
                return (
                    "Contract/integration error: "
                    f"{label} disagree (requirements={left}, geometry={right}). "
                    "No value was chosen silently."
                )
        if req.manufacturing.material and geom.material:
            if req.manufacturing.material != geom.material:
                return (
                    "Contract/integration error: material disagreement "
                    f"requirements={req.manufacturing.material!r} "
                    f"geometry={geom.material!r}."
                )
        return None

    def _resolve_payload_mass(self) -> Optional[ResolvedPayloadMass]:
        req = self.state.requirements
        geom = self.state.geometry
        req_mass = req.payload.filled_mass_kg if req else None
        geom_mass = geom.payload_object.filled_mass_kg if geom else None
        if req_mass is not None and geom_mass is not None:
            if not self._numbers_agree(req_mass, geom_mass):
                self._block_contract(
                    "Contract/integration error: payload filled_mass_kg disagree "
                    f"(requirements={req_mass}, geometry={geom_mass}). "
                    "No value was chosen silently."
                )
                return None
            return ResolvedPayloadMass(
                mass_kg=geom_mass,
                provenance=MassProvenance.GEOMETRY_OUTPUT,
                notes="requirements and geometry agree; geometry is authoritative after GEOMETRY",
            )
        if geom_mass is not None:
            return ResolvedPayloadMass(
                mass_kg=geom_mass,
                provenance=MassProvenance.GEOMETRY_OUTPUT,
                notes="payload mass taken from GeometryOutput",
            )
        if req_mass is not None:
            return ResolvedPayloadMass(
                mass_kg=req_mass,
                provenance=MassProvenance.USER_REQUIREMENTS,
                notes="geometry omitted payload mass; used user_requirements",
            )
        self._block_contract(
            "Contract/integration error: payload.filled_mass_kg is required "
            "to construct gravity loads. Do not use part.estimated_mass_kg "
            "or part_mass.max_part_mass_kg."
        )
        return None

    def _analysis_load_case_mismatch(self, analysis: AnalysisOutput) -> Optional[str]:
        structure = self.state.structure
        if structure is None or not structure.load_cases:
            return (
                "Contract/integration error: analysis fixture has no "
                "StructureOutput load case to match."
            )
        by_id = {lc.load_case_id: lc for lc in structure.load_cases}
        if analysis.load_case_id not in by_id:
            return (
                "Contract/integration error: analysis load_case_id="
                f"{analysis.load_case_id!r} does not match StructureAgent load cases "
                f"{sorted(by_id)}."
            )
        expected = by_id[analysis.load_case_id]
        if not self._vectors_agree(analysis.load_force_N, expected.force_N):
            return (
                "Contract/integration error: analysis load_force_N "
                f"{analysis.load_force_N} does not match StructureAgent load case "
                f"{expected.load_case_id} force_N={expected.force_N}."
            )
        return None

    def _vectors_agree(self, left, right) -> bool:
        if len(left) != len(right):
            return False
        return all(self._numbers_agree(a, b) for a, b in zip(left, right))

    def _frame_mismatch(self) -> Optional[str]:
        registration = self.state.registration
        geometry = self.state.geometry
        if registration is None or geometry is None:
            return None
        if registration.frame_id != geometry.coordinate_frame:
            return (
                "Contract/integration error: incompatible coordinate frames "
                f"registration.frame_id={registration.frame_id!r} "
                f"geometry.coordinate_frame={geometry.coordinate_frame!r}. "
                "No transform was applied."
            )
        return None

    @staticmethod
    def _numbers_agree(left: float, right: float) -> bool:
        return abs(left - right) <= _FLOAT_ABS_TOL

    def _snapshot(self) -> dict:
        dumped = self.state.model_dump()
        return {key: dumped.get(key) for key in _TRACE_FIELDS}

    def _record(
        self,
        stage_executed: WorkflowStage,
        component: str,
        action: str,
        before: dict,
        notes: str = "",
    ) -> None:
        after = self._snapshot()
        changed = [key for key in _TRACE_FIELDS if before.get(key) != after.get(key)]
        decision = None
        if self.state.interaction_decision is not None:
            decision = self.state.interaction_decision.value
        self.trace.append(
            TraceEvent(
                stage_executed=stage_executed,
                next_stage=self.state.stage,
                component=component,
                action=action,
                decision=decision,
                fields_changed=changed,
                notes=notes,
            )
        )

    def _step_trace_meta(self, started: WorkflowStage) -> tuple:
        if started == WorkflowStage.GEOMETRY:
            if self.state.contract_error:
                return "Orchestrator", "geometry_contract_check", self.state.contract_error
            if self.fixtures.geometry is not None:
                return "Yujie fixture", "load_geometry", "loaded Yujie mock fixture"
            if self.warm_start_generator is not None:
                if self.warm_start_error:
                    return "warm-start generator", "generate_warm_start", f"failed: {self.warm_start_error}"
                if self.imported_candidate is not None and self.imported_candidate.task == "generated":
                    return "warm-start generator", "generate_warm_start", self.imported_candidate.provenance
            return "GeometryAgent", "run_geometry", "placeholder geometry agent"
        if started == WorkflowStage.CANDIDATE_FIT:
            result = self.state.candidate_fit
            if result is None:
                return "imported candidate adapter", "candidate_fit", ""
            if result.fits:
                return "imported candidate adapter", "candidate_fit", "candidate fits"
            return "imported candidate adapter", "candidate_fit", result.message
        if started == WorkflowStage.FEASIBILITY_CHECK:
            result = self.state.feasibility
            if result is None:
                return "Orchestrator", "feasibility_check", ""
            if result.feasible:
                return "Orchestrator", "feasibility_check", "feasible"
            return "Orchestrator", "feasibility_check", result.message
        if started == WorkflowStage.STRUCTURE:
            if self.state.contract_error or self.state.structure is None:
                return "Orchestrator", "structure_contract_check", self.state.contract_error or ""
            n = len(self.state.structure.nodes)
            m = len(self.state.structure.members)
            return "StructureAgent", "run_structure", f"generated {n} nodes / {m} members"
        if started == WorkflowStage.ANALYSIS:
            if self.state.contract_error:
                return "Orchestrator", "analysis_load_case_check", self.state.contract_error
            if self.fixtures.analysis is not None:
                return "analysis fixture", "load_analysis", "MOCK RESULT"
            return "analysis tool", "run_analysis", "DETERMINISTIC MOCK RESPONSE FOR LOOP TESTING"
        if started == WorkflowStage.DESIGN_REVIEW:
            review = self.state.design_review
            if self.state.stage == WorkflowStage.DESIGN_REVIEW_FAILED:
                return "DesignReviewer", "review_design", "DESIGN_REVIEW_FAILED"
            if review is None:
                return "DesignReviewer", "review_design", ""
            return "DesignReviewer", "review_design", review.decision.value
        if started == WorkflowStage.TOPOLOGY_OPTIMIZATION:
            if self.state.contract_error:
                return "Orchestrator", "topology_prerequisite_check", self.state.contract_error
            if self.fixtures.topology is not None:
                return "Shohom fixture", "load_topology", "loaded Shohom mock fixture"
            topo = self.state.topology
            if topo is not None and not topo.is_mock:
                return "to_agent", "run_topology", topo.model
            if topo is not None and topo.notes:
                return "topology tool", "run_topology", f"placeholder topology tool ({topo.notes})"
            return "topology tool", "run_topology", "placeholder topology tool"
        if started == WorkflowStage.VERIFICATION:
            if self.state.stage != WorkflowStage.COMPLETE:
                return "verification", "run_verification", "artifacts incomplete; not COMPLETE"
            return "verification", "run_verification", "artifacts complete; safety UNVERIFIED"
        return "Orchestrator", f"handle_{started.value}", ""
