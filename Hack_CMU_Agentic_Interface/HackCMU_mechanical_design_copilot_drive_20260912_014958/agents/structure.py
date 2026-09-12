"""Structure Agent: coarse structural architecture before topology optimization."""

from __future__ import annotations

from schemas import (
    BoundaryCondition,
    LoadCase,
    LoadPath,
    Member,
    Node,
    StructureDesignParameters,
    StructureInput,
    StructureOutput,
)

INITIAL_SUPPORT_THICKNESS_MM = 4.0
INITIAL_BRACE_COUNT = 1
INITIAL_BRACE_THICKNESS_MM = 3.0
INITIAL_RING_SUPPORT_WIDTH_MM = 8.0
THICKNESS_STEP_MM = 2.0
BRACE_THICKNESS_STEP_MM = 1.0
RING_WIDTH_STEP_MM = 2.0
MAX_BRACE_COUNT = 4


class StructureAgent:
    """Proposes nodes, members, regions, BCs, and qualitative load paths."""

    def run(self, inp: StructureInput) -> StructureOutput:
        geom = inp.geometry
        payload_mass = inp.payload_mass_kg
        desk_t = geom.environment.desk_thickness_mm
        protrusion = geom.design_envelope.max_protrusion_mm
        if desk_t is None or protrusion is None:
            raise ValueError(
                "After GEOMETRY, desk thickness and design envelope must come "
                "from GeometryOutput, not silent defaults"
            )
        force = -9.81 * payload_mass
        parameters, iteration = self._next_parameters(inp)

        nodes = [
            Node(id="mount_upper", position_mm=(0.0, 0.0, desk_t), role="anchor"),
            Node(id="mount_lower", position_mm=(0.0, 0.0, 0.0), role="anchor"),
            Node(id="cup_ring", position_mm=(protrusion, 0.0, desk_t), role="load"),
            Node(id="cup_base", position_mm=(protrusion, 0.0, desk_t * 0.3), role="load"),
        ]
        members = [
            Member(
                id="upper_arm",
                start_node_id="mount_upper",
                end_node_id="cup_ring",
                kind="beam",
                notes="mount_upper -> cup_ring",
            ),
            Member(
                id="lower_arm",
                start_node_id="mount_lower",
                end_node_id="cup_base",
                kind="beam",
                notes="mount_lower -> cup_base",
            ),
            Member(
                id="brace",
                start_node_id="mount_lower",
                end_node_id="cup_ring",
                kind="beam",
                notes="brace between mount_lower and cup_ring",
            ),
            Member(
                id="cup_wall",
                start_node_id="cup_ring",
                end_node_id="cup_base",
                kind="beam",
                notes="cup ring to cup base",
            ),
            Member(
                id="inner_brace",
                start_node_id="mount_upper",
                end_node_id="cup_base",
                kind="beam",
                notes="mount_upper -> cup_base",
            ),
        ]
        if parameters.brace_count >= 2:
            members.append(
                Member(
                    id="brace_2",
                    start_node_id="mount_upper",
                    end_node_id="cup_ring",
                    kind="beam",
                    notes="added brace for displacement control",
                )
            )
        if parameters.brace_count >= 3:
            members.append(
                Member(
                    id="brace_3",
                    start_node_id="mount_lower",
                    end_node_id="cup_base",
                    kind="beam",
                    notes="added brace for displacement control",
                )
            )
        if parameters.brace_count >= 4:
            members.append(
                Member(
                    id="brace_4",
                    start_node_id="mount_upper",
                    end_node_id="cup_ring",
                    kind="beam",
                    notes="added brace for displacement control",
                )
            )
        boundary_conditions = [
            BoundaryCondition(node_id="mount_upper", notes="fixed to desk top"),
            BoundaryCondition(node_id="mount_lower", notes="fixed to desk underside"),
        ]
        load_cases = [
            LoadCase(
                load_case_id="static_gravity",
                name="static_gravity",
                region_name="cup_cavity",
                force_N=(0.0, 0.0, force),
                notes=(
                    "quasi-static gravity from payload.filled_mass_kg="
                    f"{payload_mass} kg "
                    f"(provenance={inp.payload_mass_provenance.value}, not part mass)"
                ),
            )
        ]
        load_paths = [
            LoadPath(
                name="primary",
                node_ids=["cup_ring", "cup_base", "mount_upper", "mount_lower"],
                description=(
                    "bottle -> cup_ring/cup_base -> arms/brace -> "
                    "mount_upper/mount_lower -> desk"
                ),
            )
        ]

        return StructureOutput(
            concept="desk-edge clamp with upper arm, lower arm, and brace",
            nodes=nodes,
            members=members,
            attachment_regions=list(geom.attachment_regions),
            load_regions=list(geom.load_regions),
            boundary_conditions=boundary_conditions,
            load_paths=load_paths,
            load_cases=load_cases,
            parameters=parameters,
            iteration=iteration,
        )

    def _next_parameters(self, inp: StructureInput):
        previous = inp.previous_structure
        review = inp.review
        if previous is None or review is None:
            return (
                StructureDesignParameters(
                    support_thickness_mm=INITIAL_SUPPORT_THICKNESS_MM,
                    brace_count=INITIAL_BRACE_COUNT,
                    brace_thickness_mm=INITIAL_BRACE_THICKNESS_MM,
                    ring_support_width_mm=INITIAL_RING_SUPPORT_WIDTH_MM,
                ),
                0,
            )

        params = previous.parameters.model_copy()
        for change in review.requested_changes:
            if change.action != "increase":
                continue
            if change.parameter == "support_thickness_mm":
                params.support_thickness_mm += THICKNESS_STEP_MM
            elif change.parameter == "brace_count":
                params.brace_count = min(params.brace_count + 1, MAX_BRACE_COUNT)
            elif change.parameter == "brace_thickness_mm":
                params.brace_thickness_mm += BRACE_THICKNESS_STEP_MM
            elif change.parameter == "ring_support_width_mm":
                params.ring_support_width_mm += RING_WIDTH_STEP_MM

        if params == previous.parameters:
            params.support_thickness_mm += THICKNESS_STEP_MM
        return params, previous.iteration + 1
