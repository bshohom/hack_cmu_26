"""Warm-start geometry generation: an LLM writes a constrained trimesh script, we run it.

The model never emits mesh vertices. It writes parametric Python (boxes, cylinders,
booleans) that is executed by tools/warmstart_runner.py in a subprocess, validated
(watertight, one body, inside the envelope) and written as the candidate triple
`<name>.stl` + `<name>_particles.obj` + `<name>_dimensions.txt` (+ `<name>_regions.json`).
Frame: desk_edge_frame — desk underside z=0, desk top z=desk_thickness, desk front edge
x=0 with the desk occupying x<0; the part protrudes toward +x; +Z is up.
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
- Frame (desk_edge_frame): the desk underside is the plane z=0, the desk top is z=desk_thickness_mm,
  the desk front edge is the plane x=0 and the desk occupies x<0. The desk itself is NOT part of the mesh.
  The part protrudes toward +x. Clamp arms that grip the desk lie at x<0 above z=desk_thickness_mm and below z=0.
- Define at module level:
    PARAMS: dict            # named dimensions in mm used by build()
    def build(params: dict) -> trimesh.Trimesh   # returns ONE watertight connected body
    DIMENSIONS: dict        # key dimensions in mm for humans (e.g. {"Inner diameter": 90.0})
    REGIONS: dict           # {"load": {"min":[x,y,z],"max":[x,y,z]},                # where the payload weight acts
                            #  "mounts": [{"name": str, "min":[...], "max":[...]}],  # desk contact faces (fixed)
                            #  "keep_out": [{"name": str, "min":[...], "max":[...]}]}  # cavities that must stay empty
    NOTES: list[str]
- Build from trimesh.creation.box(extents, transform) / cylinder(radius, height, sections, transform) /
  annulus(r_min, r_max, height, transform) and combine with trimesh.boolean.union(meshes, engine="manifold")
  and trimesh.boolean.difference([a, b], engine="manifold"). Overlap bodies by >=1 mm before union so
  no coincident faces remain. Keep sections <= 96 and total faces < 300000.
- Use generous wall thicknesses for FDM printing (>= 4 mm). Make the part fit the design envelope exactly.
- Return ONLY one ```python code block, no prose."""


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


def requirements_prompt(req: UserRequirements, name: str) -> str:
    env = req.environment
    obj = req.object_geometry
    desk_t = env.desk_thickness_mm
    lines = [
        f"Design a {req.description or 'desk-mounted holder'} named `{name}`.",
        f"User request: {req.user_message or req.description}",
        f"Payload: {req.payload.description}, filled mass {req.payload.filled_mass_kg} kg"
        + (f", diameter {obj.bottle_diameter_mm} mm" if obj.bottle_diameter_mm else "")
        + (f", height {obj.bottle_height_mm} mm" if obj.bottle_height_mm else ""),
        f"Desk thickness: {desk_t} mm (clamp gap must be >= desk thickness + 1 mm)",
        f"Attachment: {req.attachment.method or 'clamp'} on {req.attachment.allowed_contact_region or 'desk front edge'}; "
        f"{req.attachment.notes or 'no screws or adhesive'}",
        f"Design envelope: max protrusion from desk edge {req.design_envelope.max_protrusion_mm} mm (x <= this)"
        + (f", max width {req.design_envelope.max_width_mm} mm" if req.design_envelope.max_width_mm else "")
        + (f", max height {req.design_envelope.max_height_mm} mm" if req.design_envelope.max_height_mm else ""),
        f"Manufacturing: {req.manufacturing.method or '3d_print'} in {req.manufacturing.material or 'PLA'}",
    ]
    if req.part_mass.max_part_mass_kg:
        lines.append(f"Printed part mass limit: {req.part_mass.max_part_mass_kg} kg")
    lines.append(
        "The payload must be held with clearance >= 2 mm; REGIONS['load'] is the box where its weight rests "
        "(e.g. the cup floor); REGIONS['mounts'] are the faces touching the desk top and underside; "
        "REGIONS['keep_out'] includes the payload cavity."
    )
    return "\n".join(lines)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


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
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_CONTRACT},
        {"role": "user", "content": requirements_prompt(req, name)},
    ]
    (out_dir / f"{name}_prompt.txt").write_text(messages[0]["content"] + "\n\n---\n\n" + messages[1]["content"])
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
        script.write_text(extract_code(content))
        result = run_script(script, out_dir, name, _envelope(req), python=python)
        if result.get("ok"):
            candidate = _candidate_from_result(result, name, model)
            regions = {}
            if result.get("regions_path"):
                regions = json.loads(Path(result["regions_path"]).read_text()).get("regions", {})
            return WarmStartResult(ok=True, candidate=candidate, attempts=attempt, model=model,
                                   latency_s=time.perf_counter() - started, out_dir=str(out_dir),
                                   script_path=str(script), regions=regions)
        feedback = "\n".join(result.get("problems") or []) or result.get("error", "unknown failure")
        if log:
            log(f"warm start: attempt {attempt} rejected: {feedback[:300]}")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "The script failed validation:\n" + feedback +
                         "\nReturn the corrected COMPLETE script as one ```python block."})
    return WarmStartResult(ok=False, attempts=max_attempts, problems=result.get("problems") or [],
                           error=result.get("error", ""), model=model, latency_s=time.perf_counter() - started,
                           out_dir=str(out_dir))


def warm_start_enabled() -> bool:
    return os.environ.get("WARM_START_MODE", "auto") != "off"
