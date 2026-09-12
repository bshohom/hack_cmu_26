"""Deterministic fixture-driven replay. No LLM, no external API."""

from __future__ import annotations

import json

from fixtures import (
    EXAMPLE_DIR,
    load_clarifications,
    load_integration_fixtures,
    load_user_request,
)
from orchestrator import Orchestrator
from schemas import InteractionDecision, SafetyStatus, WorkflowStage

TRACE_PATH = EXAMPLE_DIR / "execution_trace.json"
FINAL_STATE_PATH = EXAMPLE_DIR / "final_state.json"


def _print_block(title: str, lines: list) -> None:
    print(f"\n[{title}]")
    for line in lines:
        print(line)


def _latest(orch: Orchestrator):
    return orch.trace[-1] if orch.trace else None


def run_cup_holder_fixture_demo() -> Orchestrator:
    message = load_user_request()
    answers = load_clarifications()
    fixtures = load_integration_fixtures()
    orch = Orchestrator(fixtures=fixtures)

    print("=" * 64)
    print("Fixture-driven cup-holder workflow")
    print("No LLM. No external API. Orchestrator owns every transition.")
    print("=" * 64)

    print(f"\n[SYNTHETIC USER REQUEST]\n{message}")

    orch.ingest_user_request(message)
    event = _latest(orch)
    missing = [m.field for m in orch.state.missing_information]
    _print_block(
        "REQUIREMENTS",
        [
            f"component: {event.component if event else 'InteractionAgent'}",
            f"decision: {orch.state.interaction_decision.value if orch.state.interaction_decision else 'none'}",
            "missing:" if missing else "missing: (none)",
            *[f"- {field}" for field in missing],
            f"fields changed: {event.fields_changed if event else []}",
        ],
    )

    if orch.state.stage == WorkflowStage.REQUEST_INFORMATION:
        _print_block(
            "SYNTHETIC USER RESPONSE",
            [
                f"payload mass: {answers.filled_bottle_mass_kg} kg",
                f"bottle diameter: {answers.bottle_diameter_mm} mm",
                f"desk thickness: {answers.desk_thickness_mm} mm",
                f"attachment: {answers.attachment_method} ({answers.attachment_notes})",
                f"contact region: {answers.allowed_contact_region}",
                f"max protrusion: {answers.max_protrusion_mm} mm",
                f"manufacturing: {answers.manufacturing_method} / {answers.material}",
            ],
        )
        orch.apply_answers(answers)
        event = _latest(orch)
        _print_block(
            "REQUIREMENTS",
            [
                f"component: {event.component if event else 'InteractionAgent'}",
                f"decision: {orch.state.interaction_decision.value if orch.state.interaction_decision else 'none'}",
                f"fields changed: {event.fields_changed if event else []}",
            ],
        )

    while orch.state.stage not in (
        WorkflowStage.COMPLETE,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUEST_INFORMATION,
        WorkflowStage.DESIGN_REVIEW_FAILED,
    ):
        started = orch.state.stage
        orch.step()
        event = _latest(orch)
        state = orch.state
        trace_line = (
            f"stage_executed={event.stage_executed.value} next_stage={event.next_stage.value}"
            if event
            else ""
        )

        if started == WorkflowStage.GEOMETRY:
            reg = state.registration
            geom = state.geometry
            _print_block(
                "GEOMETRY",
                [
                    f"component: {event.component}",
                    trace_line,
                    "loaded Aman mock registration" if reg else "no registration",
                    f"frame: {reg.frame_id if reg else 'n/a'}  mock={reg.is_mock if reg else None}  confidence={reg.confidence if reg else None}",
                    f"geometry frame: {geom.coordinate_frame if geom else None}  is_mock={geom.is_mock if geom else None}",
                    "loaded Yujie mock fixture" if geom else "geometry missing",
                    f"environment: {geom.environment.kind if geom else None}  desk_t={geom.environment.desk_thickness_mm if geom else None} mm",
                    f"payload: d={geom.payload_object.bottle_diameter_mm if geom else None} mm  mass={geom.payload_object.filled_mass_kg if geom else None} kg",
                    f"envelope protrusion: {geom.design_envelope.max_protrusion_mm if geom else None} mm",
                    f"part estimated mass: {geom.part.estimated_mass_kg if geom else None} kg (not payload)",
                    f"attachment regions: {[r.name for r in geom.attachment_regions] if geom else []}",
                    f"load regions: {[r.name for r in geom.load_regions] if geom else []}",
                    f"fields changed: {event.fields_changed}",
                ],
            )
        elif started == WorkflowStage.STRUCTURE and state.structure:
            s = state.structure
            _print_block(
                "STRUCTURE",
                [
                    f"component: {event.component}",
                    trace_line,
                    event.notes,
                    f"concept: {s.concept}",
                    f"nodes: {[n.id for n in s.nodes]}",
                    f"members: {[f'{m.start_node_id}->{m.end_node_id}' for m in s.members]}",
                    f"constrained DOF: {[(b.node_id, b.constrained_dof) for b in s.boundary_conditions]}",
                    f"load path: {s.load_paths[0].description if s.load_paths else ''}",
                    f"gravity load notes: {s.load_cases[0].notes if s.load_cases else ''}",
                    (
                        f"payload mass provenance: {state.resolved_payload_mass.provenance.value} "
                        f"mass={state.resolved_payload_mass.mass_kg} kg"
                        if state.resolved_payload_mass
                        else "payload mass provenance: missing"
                    ),
                    f"fields changed: {event.fields_changed}",
                ],
            )
        elif started == WorkflowStage.ANALYSIS and state.analysis:
            a = state.analysis
            _print_block(
                "ANALYSIS",
                [
                    f"component: {event.component}",
                    trace_line,
                    "MOCK RESULT",
                    f"load_case_id={a.load_case_id} load_force_N={a.load_force_N}",
                    f"is_mock={a.is_mock}  is_safety_validation={a.is_safety_validation}",
                    f"solver={a.solver}  status={a.solver_status}",
                    f"placeholder max_stress={a.max_stress_pa:.1f} Pa",
                    f"placeholder max_disp={a.max_displacement_mm:.3f} mm",
                    f"simulated FoS={a.factor_of_safety} (not a safety certification)",
                    a.disclaimer,
                    f"fields changed: {event.fields_changed}",
                ],
            )
        elif started == WorkflowStage.TOPOLOGY_OPTIMIZATION and state.topology:
            t = state.topology
            _print_block(
                "TOPOLOGY OPTIMIZATION",
                [
                    f"component: {event.component}",
                    trace_line,
                    "loaded Shohom mock fixture",
                    f"is_mock={t.is_mock}  model={t.model}  status={t.solver_status}",
                    f"volume_fraction={t.volume_fraction}  mass_reduction={t.mass_reduction_pct}%",
                    f"geometry_ref={t.optimized_geometry_ref}",
                    f"fields changed: {event.fields_changed}",
                ],
            )
        elif started == WorkflowStage.VERIFICATION and state.verification:
            v = state.verification
            _print_block(
                "VERIFICATION",
                [
                    f"component: {event.component}",
                    trace_line,
                    f"artifacts_complete={v.artifacts_complete}",
                    f"engineering validation: {state.safety_status.value.upper()}",
                    f"reason: mock analysis data",
                    f"safety_validated={v.safety_validated}  analysis_is_mock={v.analysis_is_mock}",
                    v.notes,
                    f"CAD: {state.cad.filename if state.cad else 'none'}  mock={state.cad.is_mock if state.cad else None}",
                    f"fields changed: {event.fields_changed}",
                ],
            )

        if orch.state.contract_error:
            _print_block(
                "CONTRACT ERROR",
                [orch.state.contract_error, f"stage remains {orch.state.stage.value}"],
            )
            break
        if orch.state.stage == started:
            _print_block(
                orch.state.stage.value.upper(),
                ["workflow blocked; stage did not advance"],
            )
            break

    _print_block(
        "COMPLETE" if orch.state.stage == WorkflowStage.COMPLETE else orch.state.stage.value.upper(),
        [
            f"workflow stage: {orch.state.stage.value}",
            f"decision: {orch.state.interaction_decision.value if orch.state.interaction_decision else 'none'}",
            f"safety: {orch.state.safety_status.value}",
            f"vision outstanding: {orch.state.vision_request.want_environment_photo}",
            "workflow completed as a prototype" if orch.state.stage == WorkflowStage.COMPLETE else "workflow stopped",
            "NOT validated for physical use",
        ],
    )

    if orch.state.safety_status == SafetyStatus.UNVERIFIED:
        print("safety remains UNVERIFIED because engineering tools are mocked")
    if orch.state.interaction_decision == InteractionDecision.REJECT_OR_ESCALATE:
        print(f"rejected: {orch.state.reject_reason}")

    FINAL_STATE_PATH.write_text(orch.state.model_dump_json(indent=2) + "\n")
    TRACE_PATH.write_text(
        json.dumps([event.model_dump(mode="json") for event in orch.trace], indent=2) + "\n"
    )
    print("\nWrote examples/cup_holder/final_state.json")
    print("Wrote examples/cup_holder/execution_trace.json")
    return orch


def main() -> None:
    run_cup_holder_fixture_demo()


if __name__ == "__main__":
    main()
