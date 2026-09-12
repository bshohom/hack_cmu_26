"""Load external concept-geometry artifacts for an experimental fit check.

This adapter is not Yujie's GeometryOutput and not scene reconstruction.
It reads the three cup-holder files under examples/imported_candidate/.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from schemas import (
    CandidateFitCheck,
    CandidateFitResult,
    CandidateFitStatus,
    ClarificationQuestion,
    ImportedCandidateGeometry,
    UserRequirements,
)

CANDIDATE_DIR = Path(__file__).resolve().parent / "examples" / "imported_candidate"
DIMENSIONS_NAME = "cupholder_dimensions.txt"
MESH_NAME = "cupholder_single_piece_PLA.obj"
PARTICLES_NAME = "cupholder_surface_particles.obj"

PROVENANCE = "external generated concept geometry"

_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def default_candidate_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    directory = Path(root) if root is not None else CANDIDATE_DIR
    return {
        "dimensions": directory / DIMENSIONS_NAME,
        "mesh": directory / MESH_NAME,
        "particles": directory / PARTICLES_NAME,
    }


def load_imported_candidate(root: Optional[Path] = None) -> ImportedCandidateGeometry:
    paths = default_candidate_paths(root)
    mesh_path = paths["mesh"]
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Imported candidate mesh is missing: {mesh_path}")
    dimensions_path = paths["dimensions"] if paths["dimensions"].is_file() else None
    particle_path = paths["particles"] if paths["particles"].is_file() else None

    meta: Dict[str, object] = {}
    if dimensions_path is not None:
        meta = parse_candidate_dimensions(dimensions_path.read_text())

    vertices, faces = parse_obj_mesh(mesh_path)
    bbox_min, bbox_max = bounding_box(vertices)

    return ImportedCandidateGeometry(
        mesh_path=str(mesh_path),
        particle_path=str(particle_path) if particle_path is not None else None,
        dimensions_path=str(dimensions_path) if dimensions_path is not None else None,
        inner_diameter_mm=_as_float(meta.get("inner_diameter_mm")),
        outer_diameter_mm=_as_float(meta.get("outer_diameter_mm")),
        holder_height_mm=_as_float(meta.get("holder_height_mm")),
        wall_thickness_mm=_as_float(meta.get("wall_thickness_mm")),
        base_thickness_mm=_as_float(meta.get("base_thickness_mm")),
        arm_width_mm=_as_float(meta.get("arm_width_mm")),
        top_plate_thickness_mm=_as_float(meta.get("top_plate_thickness_mm")),
        desk_gap_mm=_as_float(meta.get("desk_gap_mm")),
        compatible_desk_min_mm=_as_float(meta.get("compatible_desk_min_mm")),
        compatible_desk_max_mm=_as_float(meta.get("compatible_desk_max_mm")),
        lower_hook_thickness_mm=_as_float(meta.get("lower_hook_thickness_mm")),
        clamp_reach_mm=_as_float(meta.get("clamp_reach_mm")),
        vertex_count=len(vertices),
        face_count=len(faces),
        watertight=_as_bool(meta.get("watertight")),
        connected_components=_as_int(meta.get("connected_components")),
        bbox_min_mm=bbox_min,
        bbox_max_mm=bbox_max,
        is_mock=False,
        provenance=PROVENANCE,
    )


def parse_candidate_dimensions(text: str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        label, rest = line.split(":", 1)
        key = _normalize_label(label)
        payload = rest.strip()
        if key == "compatible_desk_range":
            match = _RANGE_RE.search(payload)
            if match:
                values["compatible_desk_min_mm"] = float(match.group(1))
                values["compatible_desk_max_mm"] = float(match.group(2))
            continue
        if key == "watertight":
            values["watertight"] = payload.lower() in {"true", "yes", "1"}
            continue
        if key in {"vertices", "faces", "connected_components"}:
            number = _NUMBER_RE.search(payload)
            if number:
                mapped = {
                    "vertices": "vertex_count",
                    "faces": "face_count",
                    "connected_components": "connected_components",
                }[key]
                values[mapped] = int(float(number.group(1)))
            continue
        if key.endswith("_mm") or key in {
            "inner_diameter_mm",
            "outer_diameter_mm",
            "holder_height_mm",
            "wall_thickness_mm",
            "base_thickness_mm",
            "arm_width_mm",
            "top_plate_thickness_mm",
            "desk_gap_mm",
            "lower_hook_thickness_mm",
            "clamp_reach_mm",
        }:
            number = _NUMBER_RE.search(payload)
            if number:
                values[key] = float(number.group(1))
    return values


def parse_obj_mesh(path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idxs = []
                for token in line.split()[1:]:
                    idxs.append(int(token.split("/")[0]) - 1)
                if len(idxs) >= 3:
                    faces.append((idxs[0], idxs[1], idxs[2]))
    return vertices, faces


def parse_obj_points(path: Path, limit: Optional[int] = None) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            points.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if limit is not None and len(points) >= limit:
                break
    return points


def bounding_box(
    vertices: Sequence[Tuple[float, float, float]],
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    if not vertices:
        return None, None
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def check_candidate_fit(
    requirements: UserRequirements,
    candidate: ImportedCandidateGeometry,
) -> CandidateFitResult:
    checks: List[CandidateFitCheck] = []

    payload = requirements.object_geometry.bottle_diameter_mm
    inner = candidate.inner_diameter_mm
    if payload is None or inner is None:
        checks.append(
            CandidateFitCheck(
                name="payload_fit",
                status=CandidateFitStatus.NA,
                required_mm=payload,
                available_mm=inner,
                message="Payload or holder inner diameter is unavailable; value was not invented.",
            )
        )
    elif payload <= inner:
        checks.append(
            CandidateFitCheck(
                name="payload_fit",
                status=CandidateFitStatus.PASS,
                required_mm=payload,
                available_mm=inner,
                message="Payload fits holder opening.",
            )
        )
    else:
        checks.append(
            CandidateFitCheck(
                name="payload_fit",
                status=CandidateFitStatus.FAIL,
                required_mm=payload,
                available_mm=inner,
                message="Payload does not fit holder opening.",
            )
        )

    desk = requirements.environment.desk_thickness_mm
    desk_min = candidate.compatible_desk_min_mm
    desk_max = candidate.compatible_desk_max_mm
    if desk is None or desk_min is None or desk_max is None:
        checks.append(
            CandidateFitCheck(
                name="desk_fit",
                status=CandidateFitStatus.NA,
                desk_mm=desk,
                supported_range_mm=(
                    (desk_min, desk_max) if desk_min is not None and desk_max is not None else None
                ),
                message="Desk thickness or candidate desk range is unavailable; value was not invented.",
            )
        )
    elif desk_min <= desk <= desk_max:
        checks.append(
            CandidateFitCheck(
                name="desk_fit",
                status=CandidateFitStatus.PASS,
                desk_mm=desk,
                supported_range_mm=(desk_min, desk_max),
                message="Desk thickness is inside the candidate clamp range.",
            )
        )
    else:
        checks.append(
            CandidateFitCheck(
                name="desk_fit",
                status=CandidateFitStatus.FAIL,
                desk_mm=desk,
                supported_range_mm=(desk_min, desk_max),
                message="Desk thickness is outside the candidate clamp range.",
            )
        )

    protrusion = requirements.design_envelope.max_protrusion_mm
    reach = candidate.clamp_reach_mm
    if protrusion is None or reach is None:
        checks.append(
            CandidateFitCheck(
                name="envelope_fit",
                status=CandidateFitStatus.NA,
                required_mm=protrusion,
                available_mm=reach,
                message="Envelope or clamp reach is unavailable; value was not invented.",
            )
        )
    elif reach <= protrusion:
        checks.append(
            CandidateFitCheck(
                name="envelope_fit",
                status=CandidateFitStatus.PASS,
                required_mm=protrusion,
                available_mm=reach,
                message="Candidate clamp reach is within the design envelope.",
            )
        )
    else:
        checks.append(
            CandidateFitCheck(
                name="envelope_fit",
                status=CandidateFitStatus.FAIL,
                required_mm=protrusion,
                available_mm=reach,
                message="Candidate clamp reach exceeds the allowed design envelope.",
            )
        )

    failed = [check for check in checks if check.status == CandidateFitStatus.FAIL]
    fits = not failed
    if fits:
        message = "Imported candidate geometry fits the resolved user requirements."
    else:
        details = "; ".join(check.message for check in failed if check.message)
        message = "CANDIDATE GEOMETRY REJECTED. " + details
    return CandidateFitResult(fits=fits, checks=checks, message=message)


def candidate_fit_questions(result: CandidateFitResult) -> List[ClarificationQuestion]:
    questions: List[ClarificationQuestion] = []
    by_name = {check.name: check for check in result.checks}
    payload = by_name.get("payload_fit")
    if payload is not None and payload.status == CandidateFitStatus.FAIL:
        questions.append(
            ClarificationQuestion(
                field="bottle_diameter_mm",
                question="Payload does not fit holder opening. Revise payload diameter or use a different candidate.",
                priority="high",
            )
        )
    desk = by_name.get("desk_fit")
    if desk is not None and desk.status == CandidateFitStatus.FAIL:
        questions.append(
            ClarificationQuestion(
                field="desk_thickness_mm",
                question="Desk thickness is outside the imported candidate clamp range. Revise desk thickness or use a different candidate.",
                priority="high",
            )
        )
    envelope = by_name.get("envelope_fit")
    if envelope is not None and envelope.status == CandidateFitStatus.FAIL:
        questions.append(
            ClarificationQuestion(
                field="max_protrusion_mm",
                question="Candidate clamp reach exceeds the allowed protrusion. Revise the design envelope or use a different candidate.",
                priority="high",
            )
        )
    if not questions:
        questions.append(
            ClarificationQuestion(
                field="bottle_diameter_mm",
                question="Revise the requirements so they fit the imported candidate geometry.",
                priority="high",
            )
        )
    return questions


def _normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    mapping = {
        "inner_diameter": "inner_diameter_mm",
        "outer_diameter": "outer_diameter_mm",
        "holder_height": "holder_height_mm",
        "wall_thickness": "wall_thickness_mm",
        "base_thickness": "base_thickness_mm",
        "arm_width": "arm_width_mm",
        "top_plate_thickness": "top_plate_thickness_mm",
        "nominal_desk_gap": "desk_gap_mm",
        "compatible_desk_range_from_concept": "compatible_desk_range",
        "compatible_desk_range": "compatible_desk_range",
        "lower_hook_thickness": "lower_hook_thickness_mm",
        "clamp_reach": "clamp_reach_mm",
        "connected_components": "connected_components",
        "vertices": "vertices",
        "faces": "faces",
        "watertight": "watertight",
    }
    return mapping.get(cleaned, cleaned)


def _as_float(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _as_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None
