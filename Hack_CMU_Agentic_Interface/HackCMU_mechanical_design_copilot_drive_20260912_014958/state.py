"""Shared DesignState: the single source of truth for the workflow."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from schemas import (
    AnalysisOutput,
    CadOutput,
    CandidateFitResult,
    ClarificationQuestion,
    DesignFeasibilityResult,
    DesignIteration,
    DesignReviewOutput,
    GeometryOutput,
    ImportedCandidateGeometry,
    InteractionDecision,
    RegistrationOutput,
    MissingInformation,
    SafetyStatus,
    StructureOutput,
    TopologyOutput,
    UserRequirements,
    VerificationResult,
    VisionRequest,
    WorkflowStage,
    ResolvedPayloadMass,
)


class DesignState(BaseModel):
    """All agents and tools read/write this object. No chat messages between agents."""

    stage: WorkflowStage = WorkflowStage.REQUIREMENTS
    safety_status: SafetyStatus = SafetyStatus.UNKNOWN
    interaction_decision: Optional[InteractionDecision] = None
    requirements: Optional[UserRequirements] = None
    missing_information: List[MissingInformation] = Field(default_factory=list)
    clarifications: List[ClarificationQuestion] = Field(default_factory=list)
    vision_request: VisionRequest = Field(default_factory=VisionRequest)
    registration: Optional[RegistrationOutput] = None
    geometry: Optional[GeometryOutput] = None
    imported_candidate: Optional[ImportedCandidateGeometry] = None
    candidate_fit: Optional[CandidateFitResult] = None
    feasibility: Optional[DesignFeasibilityResult] = None
    structure: Optional[StructureOutput] = None
    analysis: Optional[AnalysisOutput] = None
    design_review: Optional[DesignReviewOutput] = None
    design_iterations: List[DesignIteration] = Field(default_factory=list)
    topology: Optional[TopologyOutput] = None
    cad: Optional[CadOutput] = None
    verification: Optional[VerificationResult] = None
    resolved_payload_mass: Optional[ResolvedPayloadMass] = None
    reject_reason: Optional[str] = None
    contract_error: Optional[str] = None
    notes: str = ""
