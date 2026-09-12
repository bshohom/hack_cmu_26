"""Warm-start geometry generation: an LLM writes a constrained trimesh script, we run it.

The model never emits mesh vertices. It writes parametric Python (boxes, cylinders,
booleans) that is executed by tools/warmstart_runner.py in a subprocess, validated
(watertight, one body, inside the envelope) and written as the candidate triple
`<name>.stl` + `<name>_particles.obj` + `<name>_dimensions.txt` (+ `<name>_regions.json`).
Frame is task-specific. Desk-edge clamp language is used only for cupholder/desk_hook
tasks — never as a default for bed_handle.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import ImportedCandidateGeometry, UserRequirements

RUNNER = Path(__file__).resolve().parent / "warmstart_runner.py"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "generated"
MAX_ATTEMPTS = 3
RUN_TIMEOUT_S = 180

SYSTEM_CONTRACT = """You write Python that builds ONE 3D-printable part as a watertight triangle mesh.
Rules (violations fail validation and you will be asked to fix them):
- Allowed imports: numpy, trimesh, math only. No file I/O, no network, no plotting, no input().
- Units are millimetres. +Z is up (gravity is -Z).
- Installed trimesh is 5.x. FORBIDDEN (these methods do not exist): mesh.remove_duplicate_faces(),
  mesh.remove_degenerate_faces(), or any other removed Trimesh 3.x cleanup method.
  If you need cleanup, call cleanup_mesh(mesh) — it is provided in the runtime. The runner also
  calls cleanup_mesh after build(), so you may omit cleanup.
  Approved mesh APIs only: trimesh.creation.box / cylinder,
  union_all(parts) (REQUIRED for final assembly; provided in the runtime),
  trimesh.boolean.difference([a, b], engine="manifold"),
  mesh.merge_vertices(), mesh.update_faces(mesh.unique_faces()),
  mesh.remove_unreferenced_vertices(), mesh.process(validate=True), cleanup_mesh(mesh).
  FORBIDDEN assembly: trimesh.util.concatenate, Trimesh() vertex stacking, or returning
  separate un-unioned parts. concatenate is not a solid. Adjoining primitives must
  volumetrically overlap by >= 1 mm (coincident faces are not enough), then
  return union_all(parts).
- Frame: follow the USER prompt only. Do not invent a desk, bottle, cup, or clamp unless
  that prompt asks for one.
- Define at module level:
    PARAMS: dict            # named dimensions in mm used by build()
    def build(params: dict) -> trimesh.Trimesh   # returns ONE watertight connected body
    DIMENSIONS: dict        # key dimensions in mm for humans
    REGIONS: dict
    NOTES: list[str]
- REGIONS boxes are axis-aligned {"min":[x,y,z], "max":[x,y,z]} in the same frame as the mesh:
    "load":   ONE box on the PART SURFACE that carries the user load (solid material).
    "mounts": [{"name":..., box}] on the PART SURFACES that touch the support.
    "keep_out": optional empty volumes. Omit unless the user prompt needs a cavity.
- Build primitives, overlap every joint by >= 1 mm, then return union_all(parts).
  Keep sections <= 96 and total faces < 300000.
- Use generous wall thicknesses for FDM printing (>= 4 mm). Stay inside the envelope.
- Return ONLY one ```python code block, no prose."""


BED_HANDLE_CONTRACT = """Task kind: bed_handle
Build a bed-assist HANDLE. This is NOT a cup holder and NOT a desk clamp.

Required concept (choose dimensions yourself):
- one mounting block on the attachment surface (x=0, support occupies x<=0)
- one or two structural arms reaching +x
- one grip member the person pulls
- optional brace
- every joint overlaps by 1-2 mm
- final line of build(): return union_all(parts)

Do NOT create: bottle cavity, cup wall, cup radius, annular ring, desk clamp,
desk thickness, clamp gap, or any desk-edge geometry.

Frame: attachment plane x=0; part occupies x>0; +Z is up.
REGIONS['load'] = the grip. REGIONS['mounts'] = the mounting-block face on x=0.
Do not use annulus."""


CUPHOLDER_CONTRACT = """Task kind: cupholder
The payload is a bottle/cup standing upright. Build a ring wall plus a floor slab
cantilevered from a desk-edge clamp.
Frame (desk_edge_frame): desk underside z=0, desk top z=support thickness, desk
front edge x=0, desk occupies x<0. The desk is NOT part of the mesh.
REGIONS['load'] = the floor slab. REGIONS['keep_out'] = the cavity above that floor.
REGIONS['mounts'] = clamp faces that touch the desk.
Overlap joints by >= 1 mm and return union_all(parts)."""


DESK_HOOK_CONTRACT = """Task kind: desk_hook
Build a HOOK: a desk-edge clamp, an arm that descends, and a horizontal strap seat.
REGIONS['load'] = the top face of the horizontal arm. No bottle cavity.
Overlap joints by >= 1 mm and return union_all(parts)."""


WALL_SHELF_CONTRACT = """Task kind: wall_shelf
Build a WALL-MOUNTED shelf: a wall pad at x=0 and a platform reaching +x.
There is no desk and no bottle. REGIONS['load'] = platform top.
Overlap joints by >= 1 mm and return union_all(parts)."""


@dataclass
class WarmStartResult:
    ok: bool
    candidate: Optional[ImportedCandidateGeometry] = None
    attempts: int = 0
    problems: List[str] = field(default_factory=list)
    error: str = ""
    model: str = ""
    latency_s: float = 0.0
    out_dir: str = ""
    script_path: str = ""
    regions: Dict[str, Any] = field(default_factory=dict)


def _task_kind(req: UserRequirements) -> str:
    from agents.interaction import classify_design_task

    return req.task_kind or classify_design_task(req.user_message or req.description)


def task_contract(task: str) -> str:
    from agents.interaction import TASK_BED_HANDLE, TASK_CUPHOLDER, TASK_DESK_HOOK, TASK_WALL_SHELF

    if task == TASK_BED_HANDLE:
        return BED_HANDLE_CONTRACT
    if task == TASK_WALL_SHELF:
        return WALL_SHELF_CONTRACT
    if task == TASK_DESK_HOOK:
        return DESK_HOOK_CONTRACT
    if task == TASK_CUPHOLDER:
        return CUPHOLDER_CONTRACT
    return (
        f"Task kind: {task or 'generic'}\n"
        "Build the part described in the user request. Do not invent a bottle, cup, "
        "or desk clamp unless the request asks for one. Overlap joints by >= 1 mm "
        "and return union_all(parts)."
    )


def system_prompt_for(req: UserRequirements) -> str:
    return SYSTEM_CONTRACT + "\n\n" + task_contract(_task_kind(req))


def requirements_prompt(req: UserRequirements, name: str) -> str:
    from agents.interaction import TASK_BED_HANDLE, TASK_CUPHOLDER, TASK_DESK_HOOK

    extras = req.task_answers or {}
    task = _task_kind(req)
    obj = req.object_geometry
    env = req.environment
    attach_on = (
        req.attachment.allowed_contact_region
        or extras.get("attachment_structure")
        or extras.get("mounting_region")
        or "the stated support"
    )
    lines = [
        f"Design `{name}` for task={task}.",
        f"User request: {req.user_message or req.description}",
    ]
    if task == TASK_BED_HANDLE:
        lines += [
            f"Supported load: {req.payload.filled_mass_kg} kg",
            f"Attachment: {req.attachment.method or 'unspecified'} on {attach_on}",
        ]
        if extras.get("handle_location"):
            lines.append(f"Handle location: {extras['handle_location']}")
        if extras.get("drilling_allowed") is not None:
            lines.append(f"Drilling allowed: {extras['drilling_allowed']}")
        if extras.get("required_reach_mm") is not None:
            lines.append(f"Required functional reach: {extras['required_reach_mm']} mm")
        if req.design_envelope.max_protrusion_mm is not None:
            lines.append(
                f"Maximum protrusion (hard envelope, x <= this): "
                f"{req.design_envelope.max_protrusion_mm} mm"
            )
        lines.append(
            "Primitives: mounting block + arm(s) + grip + optional brace. "
            "Overlap 1-2 mm. return union_all(parts)."
        )
        lines.append("Do not invent cup, bottle, desk, clamp-gap, or annular-wall geometry.")
    else:
        lines.append(
            f"Payload / load: {req.payload.description or 'user load'}, "
            f"filled mass {req.payload.filled_mass_kg} kg"
            + (f", diameter {obj.bottle_diameter_mm} mm" if obj.bottle_diameter_mm else "")
        )
        if env.desk_thickness_mm and task in {TASK_CUPHOLDER, TASK_DESK_HOOK}:
            lines.append(f"Desk / support thickness: {env.desk_thickness_mm} mm")
        lines.append(f"Attachment: {req.attachment.method or 'unspecified'} on {attach_on}")
        if req.design_envelope.max_protrusion_mm is not None:
            lines.append(
                f"Maximum protrusion (hard envelope, x <= this): "
                f"{req.design_envelope.max_protrusion_mm} mm"
            )
        lines.append("Overlap every joint by >= 1 mm and return union_all(parts).")
    lines.append(
        f"Manufacturing: {req.manufacturing.method or '3d_print'} in {req.manufacturing.material or 'PLA'}"
    )
    lines.append("Return the script only.")
    return "\n".join(lines)


def prompt_summary(req: UserRequirements, name: str) -> str:
    task = _task_kind(req)
    extras = req.task_answers or {}
    return (
        f"task={task} name={name} load={req.payload.filled_mass_kg} "
        f"reach={extras.get('required_reach_mm')} "
        f"protrusion={req.design_envelope.max_protrusion_mm} "
        f"attach={req.attachment.method}/{req.attachment.allowed_contact_region or extras.get('attachment_structure')}"
    )


def combined_prompt(req: UserRequirements, name: str) -> str:
    return system_prompt_for(req) + "\n\n---\n\n" + requirements_prompt(req, name)


def script_uses_concatenate_as_final_assembly(source: str) -> bool:
    """True when concatenate is the last assembly, not a discarded intermediate."""
    if "concatenate" not in (source or ""):
        return False
    returns = re.findall(r"^\s*return\s+(.+?)(?:#.*)?$", source, re.M)
    if returns and "concatenate" in returns[-1]:
        return True
    last_concat = source.rfind("concatenate")
    last_union = max(source.rfind("union_all("), source.rfind("boolean.union"))
    return last_concat > last_union


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


_UNSUPPORTED_ATTR_RE = re.compile(r"has no attribute '([^']+)'")


def _trimesh_version() -> str:
    try:
        import trimesh

        return str(getattr(trimesh, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def script_failure_feedback(result: dict) -> str:
    """Retry text for Grok. Names unsupported APIs explicitly when that is the cause."""
    error = (result.get("error") or "").strip()
    problems = [str(item) for item in (result.get("problems") or []) if item]
    lines: List[str] = []
    match = _UNSUPPORTED_ATTR_RE.search(error)
    if match:
        method = match.group(1)
        lines.append(
            f"UNSUPPORTED API: trimesh {_trimesh_version()} has no attribute '{method}'. "
            f"Do not call mesh.{method}(). Use cleanup_mesh(mesh) for cleanup, or omit "
            "cleanup — the runner already calls cleanup_mesh after build(). "
            "Approved: trimesh.creation.box/cylinder, union_all(parts), "
            "trimesh.boolean.difference(..., engine='manifold'), "
            "mesh.merge_vertices(), mesh.update_faces(mesh.unique_faces()), "
            "mesh.remove_unreferenced_vertices(), mesh.process(validate=True), cleanup_mesh(mesh)."
        )
        lines.append("Traceback:")
        lines.append(error)
    elif error:
        lines.append(error)
    lines.extend(problems)
    return "\n".join(lines) or "unknown failure"


def run_script(script: Path, out_dir: Path, name: str, envelope: dict, python: Optional[str] = None) -> dict:
    out_dir = Path(out_dir).resolve()
    script = Path(script).resolve()
    env_path = out_dir / f"{name}_envelope.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps(envelope))
    cmd = [python or sys.executable, str(RUNNER), str(script), str(out_dir), name, str(env_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, cwd=str(out_dir))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"script timed out after {RUN_TIMEOUT_S}s (reduce mesh resolution)"}
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        result = {"ok": False, "error": (proc.stderr or proc.stdout)[-2000:]}
    result.setdefault("stderr", proc.stderr[-2000:])
    return result


def _envelope(req: UserRequirements) -> dict:
    e = req.design_envelope
    return {"max_protrusion_mm": e.max_protrusion_mm, "max_width_mm": e.max_width_mm, "max_height_mm": e.max_height_mm}


def _candidate_from_result(result: dict, name: str, model: str) -> ImportedCandidateGeometry:
    regions_path = result.get("regions_path")
    dims: Dict[str, float] = {}
    try:
        from imported_candidate import parse_candidate_dimensions

        dims = {k: v for k, v in parse_candidate_dimensions(Path(result["dimensions_path"]).read_text()).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    except Exception:  # noqa: BLE001 — dimensions are informational
        pass
    return ImportedCandidateGeometry(
        mesh_path=result["mesh_path"],
        particle_path=result.get("particle_path"),
        dimensions_path=result.get("dimensions_path"),
        vertex_count=result.get("vertex_count"),
        face_count=result.get("face_count"),
        watertight=result.get("watertight"),
        connected_components=result.get("connected_components"),
        bbox_min_mm=tuple(result["bbox_min_mm"]) if result.get("bbox_min_mm") else None,
        bbox_max_mm=tuple(result["bbox_max_mm"]) if result.get("bbox_max_mm") else None,
        is_mock=False,
        provenance=f"generated warm start: {model}; regions={regions_path}",
        candidate_name=name,
        task="generated",
        dimensions=dims,
        frame="desk_edge_frame",
        regions_path=regions_path,
    )


def generate_warm_start(
    req: UserRequirements,
    provider,
    name: str = "warm_start",
    out_dir: Optional[Path] = None,
    max_attempts: int = MAX_ATTEMPTS,
    python: Optional[str] = None,
    log=None,
) -> WarmStartResult:
    """Ask `provider` (needs `_chat(messages)` and `.model`) for a script; run, validate, retry."""
    out_dir = Path(out_dir or DEFAULT_OUT / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(provider, "model", provider.__class__.__name__)
    task = _task_kind(req)
    system = system_prompt_for(req)
    user = requirements_prompt(req, name)
    summary = prompt_summary(req, name)
    print(f"[GROK] TASK_PROMPT {summary}", flush=True)
    print(f"[GROK] TASK_CONTRACT\n{task_contract(task)}", flush=True)
    if log:
        log(f"TASK_PROMPT {summary}")
        log(f"TASK_CONTRACT for {task} attached; cup-holder defaults are off")
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    (out_dir / f"{name}_prompt.txt").write_text(system + "\n\n---\n\n" + user)
    started = time.perf_counter()
    result: dict = {}
    for attempt in range(1, max_attempts + 1):
        if log:
            log(f"warm start: attempt {attempt}/{max_attempts} with {model}")
        try:
            content = provider._chat(messages, timeout=180)
        except Exception as exc:  # noqa: BLE001
            return WarmStartResult(ok=False, attempts=attempt, error=f"{model} call failed: {exc}", model=model,
                                   latency_s=time.perf_counter() - started, out_dir=str(out_dir))
        (out_dir / f"{name}_response_{attempt}.txt").write_text(content)
        script = out_dir / f"{name}_script_{attempt}.py"
        code = extract_code(content)
        script.write_text(code)
        result = run_script(script, out_dir, name, _envelope(req), python=python)
        if script_uses_concatenate_as_final_assembly(code):
            problems = list(result.get("problems") or [])
            msg = (
                "ASSEMBLY REJECTED: final assembly uses concatenate. "
                "Overlap adjoining primitives by 1-2 mm and return union_all(parts)."
            )
            if not any("concatenate" in item.lower() for item in problems):
                problems.insert(0, msg)
            result["ok"] = False
            result["problems"] = problems
        if result.get("ok"):
            candidate = _candidate_from_result(result, name, model)
            regions = {}
            if result.get("regions_path"):
                regions = json.loads(Path(result["regions_path"]).read_text()).get("regions", {})
            return WarmStartResult(ok=True, candidate=candidate, attempts=attempt, model=model,
                                   latency_s=time.perf_counter() - started, out_dir=str(out_dir),
                                   script_path=str(script), regions=regions)
        feedback = script_failure_feedback(result)
        if log:
            log(f"attempt {attempt} failed: {feedback[:400]}")
            if attempt < max_attempts:
                log("retry feedback sent")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "The script failed validation:\n" + feedback +
                         "\nReturn the corrected COMPLETE script as one ```python block. "
                         "Do not call remove_duplicate_faces. Do not use concatenate. "
                         "Overlap every joint by >= 1 mm and return union_all(parts)."})
    return WarmStartResult(ok=False, attempts=max_attempts, problems=result.get("problems") or [],
                           error=result.get("error", ""), model=model, latency_s=time.perf_counter() - started,
                           out_dir=str(out_dir))


def warm_start_enabled() -> bool:
    return os.environ.get("WARM_START_MODE", "auto") != "off"
