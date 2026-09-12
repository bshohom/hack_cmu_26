"""Typed contracts for the mechanical-design workflow.

Agents and tools communicate only through these models and DesignState.
They do not exchange free-form natural-language messages.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

Vec3 = Tuple[float, float, float]


class WorkflowStage(str, Enum):
    REQUIREMENTS = "requirements"
    REQUEST_INFORMATION = "request_information"
    GEOMETRY = "geometry"
    CANDIDATE_FIT = "candidate_fit"
    FEASIBILITY_CHECK = "feasibility_check"
    STRUCTURE = "structure"
    ANALYSIS = "analysis"
    DESIGN_REVIEW = "design_review"
    TOPOLOGY_OPTIMIZATION = "topology_optimization"
    VERIFICATION = "verification"
    COMPLETE = "complete"
    REJECTED = "rejected"
    DESIGN_REVIEW_FAILED = "design_review_failed"


class InteractionDecision(str, Enum):
    PROCEED = "proceed"
    REQUEST_INFORMATION = "request_information"
    REJECT_OR_ESCALATE = "reject_or_escalate"


class SafetyStatus(str, Enum):
    UNKNOWN = "unknown"
    UNVERIFIED = "unverified"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ReasoningRole(str, Enum):
    INTERACTION = "interaction"
    GEOMETRY = "geometry"
    STRUCTURE = "structure"
    ANALYSIS_INTERPRETATION = "analysis_interpretation"
    DESIGN_REVIEW = "design_review"


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


class Payload(BaseModel):
    """The supported object (payload mass), not the printed part."""

    description: str = "bottle"
    filled_mass_kg: Optional[float] = None
    volume_l: Optional[float] = None


class ObjectGeometry(BaseModel):
    """Payload/object geometry. Bottle is a cylinder, not the design envelope.

    filled_mass_kg, if present, is the same payload mass as
    UserRequirements.payload.filled_mass_kg. It is never part mass.
    """

    kind: str = "cylinder"
    bottle_diameter_mm: Optional[float] = None
    bottle_height_mm: Optional[float] = None
    filled_mass_kg: Optional[float] = None


class EnvironmentGeometry(BaseModel):
    """Support / environment geometry (desk as plane + thickness)."""

    kind: str = "desk_plane"
    desk_thickness_mm: Optional[float] = None
    surface_normal: Vec3 = (0.0, 0.0, 1.0)


class DesignEnvelope(BaseModel):
    """Allowed occupancy of the printed part. Not object geometry."""

    max_protrusion_mm: Optional[float] = None
    max_width_mm: Optional[float] = None
    max_height_mm: Optional[float] = None


class AttachmentConstraints(BaseModel):
    method: Optional[str] = None  # clamp / screws / adhesive
    allowed_contact_region: Optional[str] = None
    notes: str = ""


class ManufacturingConstraints(BaseModel):
    """How the part will be made. A requirement, not a workflow stage."""

    method: Optional[str] = None  # 3d_print
    material: Optional[str] = None
    min_wall_thickness_mm: Optional[float] = None


class PartMassConstraint(BaseModel):
    """Mass limit of the printed part, distinct from payload mass."""

    max_part_mass_kg: Optional[float] = None


class VisionRequest(BaseModel):
    """Future photo / reference-object request. Vision is not implemented."""

    want_environment_photo: bool = False
    want_reference_object: bool = False
    requested_measurements: List[str] = Field(default_factory=list)
    notes: str = "Vision is not implemented. This interface is reserved for later."


class SceneObservation(BaseModel):
    """Live image + text observation. Not an engineering input and not CAD.

    Uncertain or inferred dimensions must not be copied into UserRequirements
    unless a later, explicit policy decides they are sufficiently grounded.
    """

    detected_payload_type: str = ""
    detected_support_type: str = ""
    likely_attachment_regions: List[str] = Field(default_factory=list)
    likely_attachment_methods: List[str] = Field(default_factory=list)
    visible_constraints: List[str] = Field(default_factory=list)
    inferred_values: Dict[str, Any] = Field(default_factory=dict)
    missing_measurements: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "cursor_live"


class UserRequirements(BaseModel):
    description: str = ""
    user_message: str = ""
    payload: Payload = Field(default_factory=Payload)
    object_geometry: ObjectGeometry = Field(default_factory=ObjectGeometry)
    environment: EnvironmentGeometry = Field(default_factory=EnvironmentGeometry)
    design_envelope: DesignEnvelope = Field(default_factory=DesignEnvelope)
    attachment: AttachmentConstraints = Field(default_factory=AttachmentConstraints)
    manufacturing: ManufacturingConstraints = Field(default_factory=ManufacturingConstraints)
    part_mass: PartMassConstraint = Field(default_factory=PartMassConstraint)
    vision: VisionRequest = Field(default_factory=VisionRequest)


class RequirementsUpdate(BaseModel):
    """Structured user answers. Not a free-form dict."""

    filled_bottle_mass_kg: Optional[float] = None
    bottle_diameter_mm: Optional[float] = None
    bottle_height_mm: Optional[float] = None
    desk_thickness_mm: Optional[float] = None
    attachment_method: Optional[str] = None
    allowed_contact_region: Optional[str] = None
    attachment_notes: Optional[str] = None
    max_protrusion_mm: Optional[float] = None
    manufacturing_method: Optional[str] = None
    material: Optional[str] = None
    max_part_mass_kg: Optional[float] = None


class MassProvenance(str, Enum):
    USER_REQUIREMENTS = "user_requirements"
    GEOMETRY_OUTPUT = "geometry_output"
    OTHER = "other"


class ResolvedPayloadMass(BaseModel):
    """Payload mass actually used to build gravity loads. Never part mass."""

    mass_kg: float
    provenance: MassProvenance
    notes: str = ""


class MissingInformation(BaseModel):
    field: str
    reason: str
    priority: str = "medium"


class ClarificationQuestion(BaseModel):
    field: str
    question: str
    priority: str = "medium"


class InteractionResult(BaseModel):
    decision: InteractionDecision
    requirements: UserRequirements
    questions: List[ClarificationQuestion] = Field(default_factory=list)
    vision_request: VisionRequest = Field(default_factory=VisionRequest)
    reject_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class AttachmentRegion(BaseModel):
    name: str
    position_mm: Vec3 = (0.0, 0.0, 0.0)
    normal: Vec3 = (0.0, 0.0, 1.0)
    area_mm2: Optional[float] = None
    notes: str = ""


class LoadRegion(BaseModel):
    name: str
    position_mm: Vec3 = (0.0, 0.0, 0.0)
    direction: Vec3 = (0.0, 0.0, -1.0)
    notes: str = ""


class PartGeometry(BaseModel):
    """Printed-part occupancy. Distinct from payload geometry and envelope.

    estimated_mass_kg is predicted printed-part mass, never payload mass.
    """

    shape: str = "clamp_arm_ring"
    length_mm: float = 0.0
    width_mm: float = 0.0
    height_mm: float = 0.0
    volume_mm3: float = 0.0
    estimated_mass_kg: Optional[float] = None


class RegistrationOutput(BaseModel):
    """Common coordinate frame from registration. Not a real CV pipeline.

    TODO(Aman): replace fixture data with the registration pipeline.
    """

    is_mock: bool
    frame_id: str = "desk_edge_frame"
    origin_mm: Vec3 = (0.0, 0.0, 0.0)
    x_axis: Vec3 = (1.0, 0.0, 0.0)
    y_axis: Vec3 = (0.0, 1.0, 0.0)
    z_axis: Vec3 = (0.0, 0.0, 1.0)
    desk_plane_point_mm: Vec3 = (0.0, 0.0, 0.0)
    desk_plane_normal: Vec3 = (0.0, 0.0, 1.0)
    confidence: float = 0.0
    notes: str = ""


class GeometryInput(BaseModel):
    requirements: UserRequirements
    # TODO(Aman): image paths / point clouds / registration into a common frame
    image_paths: List[str] = Field(default_factory=list)
    point_cloud_paths: List[str] = Field(default_factory=list)
    registration_frame: Optional[str] = None
    registration: Optional[RegistrationOutput] = None


class GeometryOutput(BaseModel):
    """Simplified engineering geometry. An image does not become exact CAD.

    After GEOMETRY, this object is the authoritative engineering representation
    for downstream stages. Coordinates are in coordinate_frame.
    """

    is_mock: bool
    coordinate_frame: str
    environment: EnvironmentGeometry
    payload_object: ObjectGeometry
    design_envelope: DesignEnvelope
    part: PartGeometry
    material: Optional[str] = None
    attachment_regions: List[AttachmentRegion] = Field(default_factory=list)
    load_regions: List[LoadRegion] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class Node(BaseModel):
    id: str
    position_mm: Vec3
    role: str = ""


class Member(BaseModel):
    id: str
    start_node_id: str
    end_node_id: str
    kind: str = "beam"
    notes: str = ""


class BoundaryCondition(BaseModel):
    node_id: str
    constrained_dof: List[str] = Field(
        default_factory=lambda: ["ux", "uy", "uz", "rx", "ry", "rz"]
    )
    notes: str = ""


class LoadPath(BaseModel):
    name: str
    node_ids: List[str] = Field(default_factory=list)
    description: str = ""


class LoadCase(BaseModel):
    load_case_id: str
    name: str
    region_name: str
    force_N: Vec3 = (0.0, 0.0, 0.0)
    notes: str = "quasi-static gravity"


class StructureDesignParameters(BaseModel):
    """Explicit, refinable mock-structure knobs. Not CAD."""

    support_thickness_mm: float = 4.0
    brace_count: int = 1
    brace_thickness_mm: float = 3.0
    ring_support_width_mm: float = 8.0


class ReviewDecision(str, Enum):
    PASS = "pass"
    REVISE = "revise"


class ReviewViolation(BaseModel):
    metric: str
    observed: float
    limit: float


class RequestedChange(BaseModel):
    parameter: str
    action: str
    reason: str


class DesignReviewOutput(BaseModel):
    """Deterministic mock design review. Not physical safety certification."""

    decision: ReviewDecision
    violations: List[ReviewViolation] = Field(default_factory=list)
    requested_changes: List[RequestedChange] = Field(default_factory=list)
    iteration: int = 0
    notes: str = (
        "Deterministic mock design review for loop testing. "
        "Not physical safety certification."
    )


class StructureInput(BaseModel):
    requirements: UserRequirements
    geometry: GeometryOutput
    payload_mass_kg: float
    payload_mass_provenance: MassProvenance
    previous_structure: Optional["StructureOutput"] = None
    review: Optional[DesignReviewOutput] = None


class StructureOutput(BaseModel):
    """Coarse structural architecture, not just a load vector."""

    concept: str
    nodes: List[Node] = Field(default_factory=list)
    members: List[Member] = Field(default_factory=list)
    attachment_regions: List[AttachmentRegion] = Field(default_factory=list)
    load_regions: List[LoadRegion] = Field(default_factory=list)
    boundary_conditions: List[BoundaryCondition] = Field(default_factory=list)
    load_paths: List[LoadPath] = Field(default_factory=list)
    load_cases: List[LoadCase] = Field(default_factory=list)
    parameters: StructureDesignParameters = Field(default_factory=StructureDesignParameters)
    iteration: int = 0

    @property
    def support_thickness_mm(self) -> float:
        return self.parameters.support_thickness_mm

    @property
    def brace_count(self) -> int:
        return self.parameters.brace_count

    @property
    def brace_thickness_mm(self) -> float:
        return self.parameters.brace_thickness_mm

    @property
    def ring_support_width_mm(self) -> float:
        return self.parameters.ring_support_width_mm


# ---------------------------------------------------------------------------
# Analysis / topology / CAD
# ---------------------------------------------------------------------------


class AnalysisInput(BaseModel):
    geometry: GeometryOutput
    structure: StructureOutput
    load_case_id: str
    load_force_N: Vec3
    material: str = "PLA"
    boundary_conditions: List[BoundaryCondition] = Field(default_factory=list)
    load_cases: List[LoadCase] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    """Mock or real analysis result. is_mock sits next to the numeric fields."""

    load_case_id: str
    load_force_N: Vec3
    is_mock: bool
    max_displacement_mm: float = 0.0
    max_stress_pa: float = 0.0
    factor_of_safety: Optional[float] = None
    reaction_forces_N: List[Vec3] = Field(default_factory=list)
    is_safety_validation: bool = False
    solver: str = "mock-fem-simulated"
    solver_status: str = "simulated_only"
    disclaimer: str = (
        "SIMULATED placeholder results. Not real FEM. "
        "Do not treat as engineering validation."
    )


class TopologySolverOptions(BaseModel):
    """Optional knobs for the live optimizer (to_agent). None = template defaults."""

    element_size_mm: Optional[float] = None
    max_iters: Optional[int] = None
    volume_fraction: Optional[float] = None
    safety_factor: Optional[float] = None
    time_budget_s: float = 150.0
    device: str = "auto"  # auto | cuda | cpu


class TopologyInput(BaseModel):
    design_domain: PartGeometry
    fixed_regions: List[AttachmentRegion] = Field(default_factory=list)
    load_regions: List[LoadRegion] = Field(default_factory=list)
    loads: List[LoadCase] = Field(default_factory=list)
    material: str = "PLA"
    target_volume_fraction: float = 0.4
    max_part_mass_kg: Optional[float] = None
    # Integration (Shohom): the warm-start candidate whose mesh frame is the problem frame.
    candidate: Optional["ImportedCandidateGeometry"] = None
    desk_thickness_mm: Optional[float] = None
    solver_options: Optional[TopologySolverOptions] = None
    # Enough context to design from scratch when there is no candidate mesh.
    envelope: Optional[DesignEnvelope] = None
    attachment_method: Optional[str] = None
    payload_kind: Optional[str] = None  # cylinder | strap | box
    payload_size_mm: Optional[float] = None
    # Coarse structural layout: its members are rasterized as the warm-start density field
    # when there is no candidate mesh, so SIMP starts from a real load path.
    structure: Optional["StructureOutput"] = None


class TopologyOutput(BaseModel):
    is_mock: bool
    compliance: float = 1.0
    volume_fraction: float = 0.4
    mass_reduction_pct: float = 20.0
    optimized_geometry_ref: str = "mock://optimized_mesh"
    solver_status: str = "mock_converged"
    model: str = "placeholder-surrogate"
    # Live-run details (empty for mocks)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    iterations: Optional[int] = None
    wall_time_s: Optional[float] = None
    converged: Optional[bool] = None
    post_check: Optional[AnalysisOutput] = None  # linear FE check of the optimized design
    problem_report: Dict[str, Any] = Field(default_factory=dict)


class CadInput(BaseModel):
    geometry: GeometryOutput
    topology: TopologyOutput


class CadOutput(BaseModel):
    is_mock: bool
    format: str = "stl"
    filename: str = "cup_holder.stl"
    notes: str = "Placeholder CAD export. Mesh vertices are not LLM-generated."


# ---------------------------------------------------------------------------
# Imported candidate geometry (experimental adapter, not Yujie GeometryOutput)
# ---------------------------------------------------------------------------


class ImportedCandidateGeometry(BaseModel):
    """External generated concept geometry. Not Yujie GeometryOutput."""

    mesh_path: str
    particle_path: Optional[str] = None
    dimensions_path: Optional[str] = None
    inner_diameter_mm: Optional[float] = None
    outer_diameter_mm: Optional[float] = None
    holder_height_mm: Optional[float] = None
    wall_thickness_mm: Optional[float] = None
    base_thickness_mm: Optional[float] = None
    arm_width_mm: Optional[float] = None
    top_plate_thickness_mm: Optional[float] = None
    desk_gap_mm: Optional[float] = None
    compatible_desk_min_mm: Optional[float] = None
    compatible_desk_max_mm: Optional[float] = None
    lower_hook_thickness_mm: Optional[float] = None
    clamp_reach_mm: Optional[float] = None
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    watertight: Optional[bool] = None
    connected_components: Optional[int] = None
    bbox_min_mm: Optional[Vec3] = None
    bbox_max_mm: Optional[Vec3] = None
    is_mock: bool = False
    provenance: str = "external generated concept geometry"
    # Integration fields (defaulted; older fixtures stay valid)
    candidate_name: str = "cupholder"
    task: str = "cupholder"  # builder key in to_agent.demo.registry, or "generated"
    dimensions: Dict[str, float] = Field(default_factory=dict)  # every numeric line of the dimensions file
    frame: str = "candidate_mesh_frame"  # "desk_edge_frame" for generated warm starts
    regions_path: Optional[str] = None  # <name>_regions.json (load / mounts / keep_out boxes)


class CandidateFitStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"


class CandidateFitCheck(BaseModel):
    name: str
    status: CandidateFitStatus
    required_mm: Optional[float] = None
    available_mm: Optional[float] = None
    desk_mm: Optional[float] = None
    supported_range_mm: Optional[Tuple[float, float]] = None
    message: str = ""


class CandidateFitResult(BaseModel):
    fits: bool
    checks: List[CandidateFitCheck] = Field(default_factory=list)
    message: str = ""


class FeasibilityViolation(BaseModel):
    code: str
    message: str = ""
    fields: List[str] = Field(default_factory=list)


class DesignFeasibilityResult(BaseModel):
    feasible: bool
    violations: List[FeasibilityViolation] = Field(default_factory=list)
    required_user_revision: bool = False
    message: str = ""


class DesignIteration(BaseModel):
    iteration: int
    structure: StructureOutput
    analysis: AnalysisOutput
    review: DesignReviewOutput


class VerificationResult(BaseModel):
    """Prototype artifact check. artifacts_complete is not physical validation."""

    artifacts_complete: bool
    analysis_is_mock: bool
    topology_is_mock: bool
    safety_validated: bool = False
    notes: str = ""


class ReasoningResult(BaseModel):
    role: ReasoningRole
    effort: ReasoningEffort
    notes: str
    is_mock: bool = True


class IntegrationFixtures(BaseModel):
    """Hypothetical teammate tool outputs injected at integration boundaries.

    Structure and verification remain our code. These fixtures stand in for
    Aman / Yujie / Shohom / analysis / CAD deliveries.
    """

    registration: Optional[RegistrationOutput] = None
    geometry: Optional[GeometryOutput] = None
    analysis: Optional[AnalysisOutput] = None
    topology: Optional[TopologyOutput] = None
    cad: Optional[CadOutput] = None


class TraceEvent(BaseModel):
    stage_executed: WorkflowStage
    next_stage: WorkflowStage
    component: str
    action: str
    decision: Optional[str] = None
    fields_changed: List[str] = Field(default_factory=list)
    notes: str = ""


TopologyInput.model_rebuild()
