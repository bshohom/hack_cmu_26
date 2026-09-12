"""CLI demo: a non-expert request driven only through the Orchestrator."""

from __future__ import annotations

from orchestrator import Orchestrator
from schemas import RequirementsUpdate, SafetyStatus, WorkflowStage


USER_MESSAGE = (
    "I want a cup holder attached to this desk that supports a full 1 L bottle."
)


def print_questions(state) -> None:
    print("  Clarification questions:")
    for q in state.clarifications:
        print(f"    - [{q.priority}] {q.question}")
    vr = state.vision_request
    print("  Vision interface (not implemented):")
    print(f"    photo={vr.want_environment_photo} reference_object={vr.want_reference_object}")
    print(f"    requested_measurements={vr.requested_measurements}")


def print_stage_result(completed: WorkflowStage, state) -> None:
    if completed == WorkflowStage.GEOMETRY and state.geometry:
        g = state.geometry
        print("  environment:", g.environment.kind, "desk_t=", g.environment.desk_thickness_mm, "mm")
        print(
            "  payload object:",
            g.payload_object.kind,
            "d=",
            g.payload_object.bottle_diameter_mm,
            "mm",
        )
        print("  envelope protrusion=", g.design_envelope.max_protrusion_mm, "mm")
        print(
            "  part:",
            g.part.shape,
            f"{g.part.length_mm:.0f}x{g.part.width_mm:.0f}x{g.part.height_mm:.0f} mm",
            f"est_mass={g.part.estimated_mass_kg:.4f} kg" if g.part.estimated_mass_kg else "",
        )
        print("  attachment regions:", [r.name for r in g.attachment_regions])
        print("  load regions:", [r.name for r in g.load_regions])
    elif completed == WorkflowStage.STRUCTURE and state.structure:
        s = state.structure
        print("  concept:", s.concept)
        print("  nodes:", [n.id for n in s.nodes])
        print("  members:", [f"{m.start_node_id}->{m.end_node_id}" for m in s.members])
        print("  BCs:", [b.node_id for b in s.boundary_conditions])
        print("  load paths:", [p.description for p in s.load_paths])
    elif completed == WorkflowStage.ANALYSIS and state.analysis:
        a = state.analysis
        print("  MOCK analysis only — not engineering validation")
        print(f"  is_mock={a.is_mock} is_safety_validation={a.is_safety_validation}")
        print(f"  solver={a.solver} status={a.solver_status}")
        print(f"  placeholder max_stress={a.max_stress_pa:.2f} Pa")
        print(f"  placeholder max_disp={a.max_displacement_mm:.4f} mm")
        print(f"  {a.disclaimer}")
    elif completed == WorkflowStage.TOPOLOGY_OPTIMIZATION and state.topology:
        t = state.topology
        print(f"  is_mock={t.is_mock} model={t.model} status={t.solver_status}")
        print(f"  volume_fraction={t.volume_fraction} mass_reduction={t.mass_reduction_pct}%")
        print(f"  geometry_ref={t.optimized_geometry_ref}")
    elif completed == WorkflowStage.VERIFICATION and state.verification:
        v = state.verification
        print(f"  complete={v.complete} analysis_is_mock={v.analysis_is_mock}")
        print(f"  safety_validated={v.safety_validated}")
        print(f"  {v.notes}")
        if state.cad:
            print(f"  CAD (mock): {state.cad.filename} [{state.cad.format}]")
            print(f"  {state.cad.notes}")


def main() -> None:
    print("=" * 60)
    print("Mechanical Design Workflow Demo")
    print("=" * 60)
    print(f"\n[User] {USER_MESSAGE}")

    orchestrator = Orchestrator()
    state = orchestrator.ingest_user_request(USER_MESSAGE)

    print(f"\n[System] decision={state.interaction_decision.value}")
    print_questions(state)

    print("\n[User] Supplying mock measurements...")
    state = orchestrator.apply_answers(
        RequirementsUpdate(
            filled_bottle_mass_kg=1.1,
            bottle_diameter_mm=70.0,
            bottle_height_mm=250.0,
            desk_thickness_mm=24.0,
            attachment_method="clamp",
            allowed_contact_region="desk_front_edge",
            max_protrusion_mm=120.0,
            manufacturing_method="3d_print",
            material="PLA",
            max_part_mass_kg=0.3,
        )
    )

    if state.stage == WorkflowStage.REQUEST_INFORMATION:
        print("\n[System] Still need more information:")
        print_questions(state)
        return
    if state.stage == WorkflowStage.REJECTED:
        print(f"\n[System] Rejected: {state.reject_reason}")
        return

    print("\n[System] Requirements complete. Orchestrator will drive remaining stages.")
    print("\n" + "=" * 60)
    print("Running workflow via Orchestrator")
    print("=" * 60)

    while state.stage not in (
        WorkflowStage.COMPLETE,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUEST_INFORMATION,
        WorkflowStage.DESIGN_REVIEW_FAILED,
    ):
        completed = state.stage
        print(f"\n[Stage: {completed.value}]")
        state = orchestrator.step()
        print_stage_result(completed, state)

    print(f"\n[Stage: {state.stage.value}]")
    print(f"  safety_status={state.safety_status.value}")
    if state.safety_status != SafetyStatus.UNVERIFIED:
        print("  warning: mock analysis must not mark the design safe")
    if state.requirements:
        req = state.requirements
        print("  payload mass (kg):", req.payload.filled_mass_kg)
        print("  part mass limit (kg):", req.part_mass.max_part_mass_kg)
        print("  bottle diameter (mm):", req.object_geometry.bottle_diameter_mm)
        print("  envelope protrusion (mm):", req.design_envelope.max_protrusion_mm)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
