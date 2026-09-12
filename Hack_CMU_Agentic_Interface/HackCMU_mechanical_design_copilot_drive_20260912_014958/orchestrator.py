"""Deterministic state machine. Owns DesignState transitions.

LLM reasoning happens inside a stage. Tools perform numerical/geometric work.
Agents never pass natural-language messages to each other.

Optional IntegrationFixtures stand in for teammate tool outputs. They are not
implementations of registration, FEM, or topology optimization.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from agents.design_review import review_design
from agents.feasibility import check_design_feasibility, feasibility_questions
from agents.geometry import GeometryAgent
from agents.interaction import InteractionAgent
from agents.structure import StructureAgent
from reasoning import run_reasoning_agent
from imported_candidate import candidate_fit_questions, check_candidate_fit
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
from tools.topology import run_topology_optimization

MAX_STRUCTURE_ITERATIONS = 4

_STOP_STAGES: Set[WorkflowStage] = {
    WorkflowStage.COMPLETE,
    WorkflowStage.REJECTED,
    WorkflowStage.REQUEST_INFORMATION,
    WorkflowStage.DESIGN_REVIEW_FAILED,
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
        warm_start_generator=None,
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
        # Callable[[UserRequirements], Optional[ImportedCandidateGeometry]]: generates the
        # warm-start candidate (e.g. Grok-written trimesh script) when none was imported.
        self.warm_start_generator = warm_start_generator
        self.warm_start_error: Optional[str] = None
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
        }

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
        if not self._requirements_ready():
            self._request_info("Geometry needs complete requirements.")
            return
        run_reasoning_agent(ReasoningRole.GEOMETRY, self.state)
        assert self.state.requirements is not None
        if self.fixtures.registration is not None:
            self.state.registration = self.fixtures.registration
        if self.fixtures.geometry is not None:
            self.state.geometry = self.fixtures.geometry
            if self.state.registration is None:
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
        if self.imported_candidate is None and self.warm_start_generator is not None:
            self.warm_start_error = None
            try:
                generated = self.warm_start_generator(self.state.requirements)
            except Exception as exc:  # noqa: BLE001 — generation failure never blocks the workflow
                generated = None
                self.warm_start_error = f"{type(exc).__name__}: {exc}"
            if generated is not None:
                self.imported_candidate = generated
            else:
                self.state.notes = (
                    "Warm-start generation failed; continuing without a candidate mesh "
                    f"(topology optimization will be mocked). {self.warm_start_error or ''}"
                ).strip()
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
        questions = candidate_fit_questions(result)
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
        if self.state.analysis is None:
            self._block_contract(
                "Contract/integration error: TOPOLOGY_OPTIMIZATION requires "
                "a valid AnalysisOutput."
            )
            return
        if self.state.geometry is None or self.state.structure is None:
            self._request_info("Topology optimization needs geometry and structure.")
            return
        req = self.state.requirements
        material = "PLA"
        max_mass: Optional[float] = None
        if req:
            material = req.manufacturing.material or "PLA"
            max_mass = req.part_mass.max_part_mass_kg
        if self.fixtures.topology is not None:
            self.state.topology = self.fixtures.topology
        else:
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
            )
            self.state.topology = run_topology_optimization(inp, log=self.topology_log)
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
        artifacts_ok = artifacts_ok and self.state.cad is not None

        mocked = analysis_mock or topology_mock or (
            self.state.cad.is_mock if self.state.cad is not None else True
        )
        if mocked:
            safety_validated = False
            safety_status = SafetyStatus.UNVERIFIED
            reason = "mock analysis data" if analysis_mock else "analysis is not a safety certification"
            if topology_mock:
                reason += "; mock topology data"
        else:
            safety_validated = False
            safety_status = SafetyStatus.UNVERIFIED
            reason = "this prototype does not certify physical safety"
        check = self.state.topology.post_check if self.state.topology is not None else None
        if check is not None:
            fos = f"{check.factor_of_safety:.2f}" if check.factor_of_safety is not None else "n/a"
            reason += (
                f"; post-TO linear FE check at nominal load: max displacement "
                f"{check.max_displacement_mm:.3f} mm, max stress {check.max_stress_pa / 1e6:.2f} MPa, "
                f"factor of safety {fos} (not a certification)"
            )

        if not artifacts_ok:
            self.state.verification = VerificationResult(
                artifacts_complete=False,
                analysis_is_mock=analysis_mock,
                topology_is_mock=topology_mock,
                safety_validated=False,
                notes=(
                    "Required prototype artifacts are missing. "
                    "stage remains VERIFICATION. NOT validated for physical use."
                ),
            )
            self.state.safety_status = SafetyStatus.NEEDS_REVIEW
            return

        self.state.verification = VerificationResult(
            artifacts_complete=True,
            analysis_is_mock=analysis_mock,
            topology_is_mock=topology_mock,
            safety_validated=safety_validated,
            notes=(
                "engineering validation: UNVERIFIED. "
                f"reason: {reason}. "
                "COMPLETE means artifacts exist, not physical safety. "
                "NOT validated for physical use."
            ),
        )
        self.state.safety_status = safety_status
        self.state.stage = WorkflowStage.COMPLETE

    def _handle_complete(self) -> None:
        return

    def _requirements_ready(self) -> bool:
        req = self.state.requirements
        if req is None:
            return False
        return all(
            [
                req.payload.filled_mass_kg is not None,
                req.object_geometry.bottle_diameter_mm is not None,
                req.environment.desk_thickness_mm is not None,
                bool(req.attachment.method),
                bool(req.attachment.allowed_contact_region),
                req.design_envelope.max_protrusion_mm is not None,
                bool(req.manufacturing.method),
            ]
        )

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
