"""Format DesignState for the workflow inspect panels. Display only."""

from __future__ import annotations

from typing import Any, Dict, List

from schemas import ReviewDecision, WorkflowStage
from state import DesignState

PIPELINE = [
    "REQUIREMENTS",
    "REGISTRATION",
    "GEOMETRY",
    "CANDIDATE FIT",
    "FEASIBILITY",
    "STRUCTURE",
    "ANALYSIS",
    "DESIGN REVIEW",
    "TOPOLOGY OPTIMIZATION",
    "VERIFICATION",
    "CAD OUTPUT",
]


def _artifact_badge(is_mock: bool) -> str:
    return "MOCK" if is_mock else "LIVE"


def highlighted_stage(state: DesignState) -> str:
    """Which pipeline card to emphasize. Display-only; not a transition."""
    stage = state.stage
    if stage in (WorkflowStage.REQUIREMENTS, WorkflowStage.REJECTED):
        return "REQUIREMENTS"
    if stage == WorkflowStage.REQUEST_INFORMATION:
        if state.candidate_fit is not None and not state.candidate_fit.fits:
            return "CANDIDATE FIT"
        if state.feasibility is not None and not state.feasibility.feasible:
            return "FEASIBILITY"
        return "REQUIREMENTS"
    if stage == WorkflowStage.GEOMETRY:
        return "REGISTRATION" if state.registration is None else "GEOMETRY"
    if stage == WorkflowStage.CANDIDATE_FIT:
        return "CANDIDATE FIT"
    if stage == WorkflowStage.FEASIBILITY_CHECK:
        return "FEASIBILITY"
    if stage == WorkflowStage.STRUCTURE:
        return "STRUCTURE"
    if stage == WorkflowStage.ANALYSIS:
        return "ANALYSIS"
    if stage in (WorkflowStage.DESIGN_REVIEW, WorkflowStage.DESIGN_REVIEW_FAILED):
        return "DESIGN REVIEW"
    if stage == WorkflowStage.TOPOLOGY_OPTIMIZATION:
        return "TOPOLOGY OPTIMIZATION"
    if stage == WorkflowStage.VERIFICATION:
        return "VERIFICATION"
    if stage == WorkflowStage.COMPLETE:
        return "CAD OUTPUT"
    return "REQUIREMENTS"


def pipeline_status(state: DesignState) -> Dict[str, str]:
    """Derive display badges from backend state. Does not decide transitions."""
    status = {name: "NOT STARTED" for name in PIPELINE}
    stage = state.stage

    if stage == WorkflowStage.REJECTED:
        status["REQUIREMENTS"] = "REJECTED"
        return status

    missing_before_geometry = (
        stage == WorkflowStage.REQUEST_INFORMATION
        and state.geometry is None
        and (state.feasibility is None or state.feasibility.feasible)
        and (state.candidate_fit is None or state.candidate_fit.fits)
    )
    if missing_before_geometry:
        status["REQUIREMENTS"] = "WAITING FOR INPUT"
        return status

    if state.requirements is not None:
        status["REQUIREMENTS"] = "COMPLETE"
    if stage == WorkflowStage.REQUEST_INFORMATION and state.feasibility is not None and not state.feasibility.feasible:
        status["REQUIREMENTS"] = "NEEDS REVISION"

    if state.registration is not None:
        status["REGISTRATION"] = _artifact_badge(state.registration.is_mock)
    if state.geometry is not None:
        status["GEOMETRY"] = _artifact_badge(state.geometry.is_mock)
    if state.candidate_fit is not None:
        status["CANDIDATE FIT"] = "PASS" if state.candidate_fit.fits else "REJECTED"
    elif state.imported_candidate is not None:
        status["CANDIDATE FIT"] = "IMPORTED"
    if state.feasibility is not None:
        status["FEASIBILITY"] = "FEASIBLE" if state.feasibility.feasible else "INFEASIBLE"
    if state.structure is not None:
        status["STRUCTURE"] = f"ITERATION {state.structure.iteration}"
    if state.analysis is not None:
        status["ANALYSIS"] = _artifact_badge(state.analysis.is_mock)
    if state.design_review is not None:
        if stage == WorkflowStage.DESIGN_REVIEW_FAILED:
            status["DESIGN REVIEW"] = "FAILED"
        else:
            status["DESIGN REVIEW"] = state.design_review.decision.value.upper()
    elif stage == WorkflowStage.DESIGN_REVIEW_FAILED:
        status["DESIGN REVIEW"] = "FAILED"
    if state.topology is not None:
        status["TOPOLOGY OPTIMIZATION"] = _artifact_badge(state.topology.is_mock)
    if state.verification is not None:
        status["VERIFICATION"] = (
            "COMPLETE" if state.verification.artifacts_complete else "BLOCKED"
        )
    if state.cad is not None:
        status["CAD OUTPUT"] = _artifact_badge(state.cad.is_mock)
    elif stage == WorkflowStage.COMPLETE:
        status["CAD OUTPUT"] = "COMPLETE"

    if state.contract_error:
        current = highlighted_stage(state)
        status[current] = "BLOCKED"
    return status


def design_loop_timeline(state: DesignState) -> List[Dict[str, Any]]:
    """Visible STRUCTURE → ANALYSIS → REVIEW loop derived from history."""
    steps: List[Dict[str, Any]] = []
    for item in state.design_iterations:
        review = item.review
        analysis = item.analysis
        steps.append(
            {
                "title": f"STRUCTURE — ITERATION {item.iteration}",
                "badge": f"ITERATION {item.iteration}",
            }
        )
        steps.append(
            {
                "title": "ANALYSIS — MOCK",
                "badge": "MOCK",
                "detail": (
                    f"disp {analysis.max_displacement_mm:.2f} mm, "
                    f"stress {analysis.max_stress_pa / 1e6:.2f} MPa"
                ),
            }
        )
        if state.stage == WorkflowStage.DESIGN_REVIEW_FAILED and item is state.design_iterations[-1]:
            badge = "FAILED"
        elif review.decision == ReviewDecision.PASS:
            badge = "PASS"
        else:
            badge = "REVISE"
        steps.append(
            {
                "title": f"DESIGN REVIEW — {badge}",
                "badge": badge,
            }
        )
    if state.topology is not None:
        steps.append(
            {
                "title": "TOPOLOGY OPTIMIZATION",
                "badge": _artifact_badge(state.topology.is_mock),
            }
        )
    elif state.stage == WorkflowStage.DESIGN_REVIEW_FAILED:
        steps.append(
            {
                "title": "TOPOLOGY OPTIMIZATION",
                "badge": "NOT STARTED",
            }
        )
    return steps


def iteration_cards(state: DesignState) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for item in state.design_iterations:
        params = item.structure.parameters
        cards.append(
            {
                "iteration": item.iteration,
                "support_thickness_mm": params.support_thickness_mm,
                "brace_count": params.brace_count,
                "brace_thickness_mm": params.brace_thickness_mm,
                "ring_support_width_mm": params.ring_support_width_mm,
                "max_displacement_mm": item.analysis.max_displacement_mm,
                "max_stress_mpa": item.analysis.max_stress_pa / 1e6,
                "review": item.review.decision.value.upper(),
                "requested_changes": [
                    {
                        "parameter": change.parameter,
                        "action": change.action,
                        "reason": change.reason,
                    }
                    for change in item.review.requested_changes
                ],
                "violations": [
                    {
                        "metric": violation.metric,
                        "observed": violation.observed,
                        "limit": violation.limit,
                    }
                    for violation in item.review.violations
                ],
            }
        )
    return cards


def engineering_evidence_is_mocked(state: DesignState) -> bool:
    if state.analysis is not None and state.analysis.is_mock:
        return True
    if state.topology is not None and state.topology.is_mock:
        return True
    if state.cad is not None and state.cad.is_mock:
        return True
    if state.verification is not None and (
        state.verification.analysis_is_mock or state.verification.topology_is_mock
    ):
        return True
    return False


def inspect_cards(state: DesignState) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    if state.requirements is not None:
        req = state.requirements
        cards.append(
            {
                "title": "REQUIREMENTS",
                "owner": "Our system — InteractionAgent",
                "input_schema": "user message (str)",
                "input": {"user_message": req.user_message},
                "output_schema": "UserRequirements + InteractionDecision",
                "output": {
                    "decision": state.interaction_decision.value
                    if state.interaction_decision
                    else None,
                    "payload_mass_kg": req.payload.filled_mass_kg,
                    "bottle_diameter_mm": req.object_geometry.bottle_diameter_mm,
                    "desk_thickness_mm": req.environment.desk_thickness_mm,
                    "attachment": req.attachment.method,
                    "max_protrusion_mm": req.design_envelope.max_protrusion_mm,
                    "manufacturing": req.manufacturing.method,
                },
                "is_mock": False,
                "provenance": None,
            }
        )
    if state.registration is not None:
        reg = state.registration
        cards.append(
            {
                "title": "REGISTRATION",
                "owner": "Aman",
                "input_schema": "GeometryInput.image_paths / point clouds",
                "input": {"photo": "uploaded image is not consumed by registration yet"},
                "output_schema": "RegistrationOutput",
                "output": {
                    "frame_id": reg.frame_id,
                    "confidence": reg.confidence,
                    "is_mock": reg.is_mock,
                },
                "is_mock": reg.is_mock,
                "provenance": "mock fixture" if reg.is_mock else "live",
            }
        )
    if state.geometry is not None:
        geom = state.geometry
        cards.append(
            {
                "title": "GEOMETRY",
                "owner": (
                    "Yujie"
                    if "Yujie" in (geom.notes or "")
                    else "Our system — GeometryAgent"
                ),
                "input_schema": "GeometryInput",
                "input": {
                    "coordinate_frame": geom.coordinate_frame,
                    "from_requirements": True,
                },
                "output_schema": "GeometryOutput",
                "output": {
                    "payload": geom.payload_object.kind,
                    "diameter_mm": geom.payload_object.bottle_diameter_mm,
                    "desk_thickness_mm": geom.environment.desk_thickness_mm,
                    "protrusion_mm": geom.design_envelope.max_protrusion_mm,
                    "part_estimated_mass_kg": geom.part.estimated_mass_kg,
                    "is_mock": geom.is_mock,
                },
                "is_mock": geom.is_mock,
                "provenance": geom.notes or ("mock fixture" if geom.is_mock else "live"),
            }
        )
    if state.imported_candidate is not None:
        cand = state.imported_candidate
        fit = state.candidate_fit
        cards.append(
            {
                "title": "CANDIDATE FIT",
                "owner": "Experimental imported-candidate adapter",
                "input_schema": "UserRequirements + ImportedCandidateGeometry",
                "input": {
                    "mesh_path": cand.mesh_path,
                    "inner_diameter_mm": cand.inner_diameter_mm,
                    "desk_range": [cand.compatible_desk_min_mm, cand.compatible_desk_max_mm],
                },
                "output_schema": "CandidateFitResult",
                "output": {
                    "fits": fit.fits if fit is not None else None,
                    "checks": [c.model_dump() for c in fit.checks] if fit is not None else [],
                    "provenance": cand.provenance,
                    "vertex_count": cand.vertex_count,
                    "face_count": cand.face_count,
                },
                "is_mock": cand.is_mock,
                "provenance": cand.provenance,
            }
        )
    if state.feasibility is not None:
        feas = state.feasibility
        cards.append(
            {
                "title": "FEASIBILITY",
                "owner": "Our system — feasibility gate",
                "input_schema": "UserRequirements + GeometryOutput",
                "input": {"geometry": True, "requirements": True},
                "output_schema": "DesignFeasibilityResult",
                "output": {
                    "feasible": feas.feasible,
                    "required_user_revision": feas.required_user_revision,
                    "violations": [v.code for v in feas.violations],
                    "message": feas.message,
                },
                "is_mock": False,
                "provenance": None,
            }
        )
    if state.structure is not None:
        structure = state.structure
        load = structure.load_cases[0] if structure.load_cases else None
        cards.append(
            {
                "title": "STRUCTURE",
                "owner": "Our system — StructureAgent",
                "input_schema": "StructureInput",
                "input": {
                    "payload_mass_kg": (
                        state.resolved_payload_mass.mass_kg
                        if state.resolved_payload_mass
                        else None
                    ),
                    "provenance": (
                        state.resolved_payload_mass.provenance.value
                        if state.resolved_payload_mass
                        else None
                    ),
                    "iteration": structure.iteration,
                },
                "output_schema": "StructureOutput",
                "output": {
                    "concept": structure.concept,
                    "iteration": structure.iteration,
                    "support_thickness_mm": structure.parameters.support_thickness_mm,
                    "brace_count": structure.parameters.brace_count,
                    "nodes": len(structure.nodes),
                    "members": len(structure.members),
                    "load_case_id": load.load_case_id if load else None,
                    "force_N": load.force_N if load else None,
                },
                "is_mock": False,
                "provenance": (
                    state.resolved_payload_mass.notes
                    if state.resolved_payload_mass
                    else None
                ),
            }
        )
    if state.analysis is not None:
        analysis = state.analysis
        cards.append(
            {
                "title": "ANALYSIS",
                "owner": "Analysis tool / fixture",
                "input_schema": "AnalysisInput",
                "input": {
                    "load_case_id": analysis.load_case_id,
                    "load_force_N": analysis.load_force_N,
                },
                "output_schema": "AnalysisOutput",
                "output": {
                    "is_mock": analysis.is_mock,
                    "max_stress_pa": analysis.max_stress_pa,
                    "max_displacement_mm": analysis.max_displacement_mm,
                    "factor_of_safety": analysis.factor_of_safety,
                    "is_safety_validation": analysis.is_safety_validation,
                    "disclaimer": analysis.disclaimer,
                },
                "is_mock": analysis.is_mock,
                "provenance": analysis.solver,
            }
        )
    if state.design_review is not None:
        review = state.design_review
        cards.append(
            {
                "title": "DESIGN REVIEW",
                "owner": "Our system — DesignReviewer",
                "input_schema": "StructureOutput + AnalysisOutput",
                "input": {"iteration": review.iteration},
                "output_schema": "DesignReviewOutput",
                "output": {
                    "decision": review.decision.value,
                    "violations": [v.model_dump() for v in review.violations],
                    "requested_changes": [c.model_dump() for c in review.requested_changes],
                },
                "is_mock": True,
                "provenance": review.notes,
            }
        )
    if state.topology is not None:
        topology = state.topology
        cards.append(
            {
                "title": "TOPOLOGY OPTIMIZATION",
                "owner": "Shohom",
                "input_schema": "TopologyInput",
                "input": {"design_domain": "GeometryOutput.part"},
                "output_schema": "TopologyOutput",
                "output": {
                    "is_mock": topology.is_mock,
                    "volume_fraction": topology.volume_fraction,
                    "mass_reduction_pct": topology.mass_reduction_pct,
                    "geometry_ref": topology.optimized_geometry_ref,
                    "model": topology.model,
                },
                "is_mock": topology.is_mock,
                "provenance": "mock fixture" if topology.is_mock else "live",
            }
        )
    if state.verification is not None:
        ver = state.verification
        cards.append(
            {
                "title": "VERIFICATION",
                "owner": "Our system",
                "input_schema": "DesignState artifacts",
                "input": {
                    "geometry": state.geometry is not None,
                    "structure": state.structure is not None,
                    "analysis": state.analysis is not None,
                    "topology": state.topology is not None,
                },
                "output_schema": "VerificationResult",
                "output": {
                    "artifacts_complete": ver.artifacts_complete,
                    "safety_validated": ver.safety_validated,
                    "safety_status": state.safety_status.value,
                    "notes": ver.notes,
                },
                "is_mock": ver.analysis_is_mock or ver.topology_is_mock,
                "provenance": None,
            }
        )
    if state.cad is not None:
        cad = state.cad
        cards.append(
            {
                "title": "CAD OUTPUT",
                "owner": "CAD tool / fixture",
                "input_schema": "CadInput",
                "input": {"geometry": True, "topology": True},
                "output_schema": "CadOutput",
                "output": {
                    "filename": cad.filename,
                    "format": cad.format,
                    "is_mock": cad.is_mock,
                    "stl_hook": "Display STL/OBJ here when a real path exists.",
                },
                "is_mock": cad.is_mock,
                "provenance": cad.notes,
            }
        )
    return cards
