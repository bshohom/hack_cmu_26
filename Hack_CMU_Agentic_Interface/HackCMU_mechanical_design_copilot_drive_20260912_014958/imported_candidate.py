"""Load external concept-geometry artifacts for an experimental fit check.

This adapter is not Yujie's GeometryOutput and not scene reconstruction.
It reads the three cup-holder files under examples/imported_candidate/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
# Demo triples dropped at the hack_cmu_26 repo root (not copied: 7-13 MB STLs).
REPO_ROOT = Path(__file__).resolve().parents[2]

PROVENANCE = "external generated concept geometry"

_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class CandidateFiles:
    """One `<mesh> + <name>_dimensions.txt + <name>_particles.obj` triple and its TO task key."""

    name: str
    task: str  # builder key in to_agent.demo.registry
    directory: Path
    mesh: str
    dimensions: str
    particles: str
    label: str = ""

    def paths(self) -> Dict[str, Path]:
        return {
            "dimensions": self.directory / self.dimensions,
            "mesh": self.directory / self.mesh,
            "particles": self.directory / self.particles,
        }


CANDIDATES: Dict[str, CandidateFiles] = {
    "cupholder": CandidateFiles(
        "cupholder", "cupholder", CANDIDATE_DIR, MESH_NAME, DIMENSIONS_NAME, PARTICLES_NAME,
        "Desk-clamp cup holder (1 L bottle)",
    ),
    "desk_bag_hook": CandidateFiles(
        "desk_bag_hook", "desk_bag_hook", REPO_ROOT,
        "desk_bag_hook_5kg_100mm_final.stl",
        "desk_bag_hook_5kg_100mm_final_dimensions.txt",
        "desk_bag_hook_5kg_100mm_final_particles.obj",
        "Desk bag hook (5 kg, 100 mm reach)",
    ),
    "stapler_shelf": CandidateFiles(
        "stapler_shelf", "stapler_shelf", REPO_ROOT,
        "stapler_shelf_100mm_guaranteed_flat_top.stl",
        "stapler_shelf_100mm_guaranteed_flat_top_dimensions.txt",
        "stapler_shelf_100mm_guaranteed_flat_top_particles.obj",
        "Stapler shelf (100 mm lift, flat top)",
    ),
}


def default_candidate_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    directory = Path(root) if root is not None else CANDIDATE_DIR
    return {
        "dimensions": directory / DIMENSIONS_NAME,
        "mesh": directory / MESH_NAME,
        "particles": directory / PARTICLES_NAME,
    }


def load_imported_candidate(root: Optional[Path] = None) -> ImportedCandidateGeometry:
    return _load_candidate_paths(default_candidate_paths(root), name="cupholder", task="cupholder")


def load_candidate(name: str = "cupholder") -> ImportedCandidateGeometry:
    """Load any registered candidate triple by name."""
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}; known: {sorted(CANDIDATES)}")
    files = CANDIDATES[name]
    return _load_candidate_paths(files.paths(), name=files.name, task=files.task)


def _mesh_stats(mesh_path: Path) -> Dict[str, object]:
    """Vertex/face counts, watertightness, components and bbox via trimesh (STL/OBJ/PLY)."""
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh")
        lo, hi = mesh.bounds
        return {
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "connected_components": int(len(mesh.split(only_watertight=False))),
            "bbox_min": tuple(float(v) for v in lo),
            "bbox_max": tuple(float(v) for v in hi),
        }
    except Exception:  # noqa: BLE001 — fall back to the OBJ text parser
        vertices, faces = parse_obj_mesh(mesh_path)
        bbox_min, bbox_max = bounding_box(vertices)
        return {"vertex_count": len(vertices), "face_count": len(faces), "watertight": None,
                "connected_components": None, "bbox_min": bbox_min, "bbox_max": bbox_max}


def _load_candidate_paths(paths: Dict[str, Path], name: str, task: str) -> ImportedCandidateGeometry:
    mesh_path = paths["mesh"]
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Imported candidate mesh is missing: {mesh_path}")
    dimensions_path = paths["dimensions"] if paths["dimensions"].is_file() else None
    particle_path = paths["particles"] if paths["particles"].is_file() else None

    meta: Dict[str, object] = {}
    numeric: Dict[str, float] = {}
    if dimensions_path is not None:
        text = dimensions_path.read_text()
        meta = parse_candidate_dimensions(text)
        numeric = parse_all_numeric(text)

    stats = _mesh_stats(mesh_path)
    bbox_min, bbox_max = stats["bbox_min"], stats["bbox_max"]

    # Fit-check fields for clamp-style parts whose dimension files use other labels
    desk_gap = _as_float(meta.get("desk_gap_mm"))
    desk_min = _as_float(meta.get("compatible_desk_min_mm"))
    desk_max = _as_float(meta.get("compatible_desk_max_mm"))
    if desk_gap is None and "clamp_internal_gap" in numeric:
        desk_gap = numeric["clamp_internal_gap"]
    if desk_min is None and desk_max is None and desk_gap is not None and "desk_thickness_reference" in numeric:
        desk_min, desk_max = numeric["desk_thickness_reference"] - 2.0, desk_gap
    clamp_reach = _as_float(meta.get("clamp_reach_mm"))
    if clamp_reach is None and desk_gap is not None and bbox_max is not None:
        clamp_reach = float(bbox_max[0])  # protrusion of a clamp part = x extent in desk_edge_frame

    watertight = _as_bool(meta.get("watertight"))
    components = _as_int(meta.get("connected_components"))
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
        desk_gap_mm=desk_gap,
        compatible_desk_min_mm=desk_min,
        compatible_desk_max_mm=desk_max,
        lower_hook_thickness_mm=_as_float(meta.get("lower_hook_thickness_mm")),
        clamp_reach_mm=clamp_reach,
        vertex_count=stats["vertex_count"],
        face_count=stats["face_count"],
        watertight=watertight if watertight is not None else stats["watertight"],
        connected_components=components if components is not None else stats["connected_components"],
        bbox_min_mm=bbox_min,
        bbox_max_mm=bbox_max,
        is_mock=False,
        provenance=PROVENANCE,
        candidate_name=name,
        task=task,
        dimensions=numeric,
        frame="candidate_mesh_frame",
    )


def parse_all_numeric(text: str) -> Dict[str, float]:
    """Every `label: number` line (bullets and `Z=100.0` tolerated) keyed by normalized label."""
    values: Dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•* ").strip()
        if not line or ":" not in line:
            continue
        label, rest = line.split(":", 1)
        key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
        if not key:
            continue
        match = _RANGE_RE.search(rest)
        if match and "range" in key:
            values[key + "_min"] = float(match.group(1))
            values[key + "_max"] = float(match.group(2))
            continue
        number = _NUMBER_RE.search(rest)
        if number:
            values[key] = float(number.group(1))
    return values


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


def _dim(candidate: ImportedCandidateGeometry, *keys: str) -> Optional[float]:
    """A declared numeric line from the candidate dimensions file. Not invented."""
    dims = candidate.dimensions or {}
    for key in keys:
        value = dims.get(key)
        if value is not None:
            return float(value)
    return None


def _payload_size_mm(requirements: UserRequirements) -> Optional[float]:
    return requirements.object_geometry.bottle_diameter_mm


def _payload_mass_kg(requirements: UserRequirements) -> Optional[float]:
    return requirements.payload.filled_mass_kg or requirements.object_geometry.filled_mass_kg


def _platform_min_mm(candidate: ImportedCandidateGeometry) -> Optional[float]:
    length = _dim(candidate, "platform_length")
    width = _dim(candidate, "platform_width")
    if length is None or width is None:
        return None
    return min(length, width)


def _hook_opening_mm(candidate: ImportedCandidateGeometry) -> Optional[float]:
    """Measured strap opening on the hook arm. Never inner_diameter."""
    return _dim(candidate, "hook_opening")


def _hook_reach_mm(candidate: ImportedCandidateGeometry) -> Optional[float]:
    if candidate.clamp_reach_mm is not None:
        return float(candidate.clamp_reach_mm)
    return _dim(candidate, "desk_face_to_hook_centerline")


def _shelf_reach_mm(candidate: ImportedCandidateGeometry) -> Optional[float]:
    declared = _dim(candidate, "platform_length")
    if declared is not None:
        return declared
    if candidate.bbox_min_mm is not None and candidate.bbox_max_mm is not None:
        return float(candidate.bbox_max_mm[0] - candidate.bbox_min_mm[0])
    return None


def _leq_check(
    name: str,
    required: Optional[float],
    available: Optional[float],
    *,
    pass_msg: str,
    fail_msg: str,
    unknown_msg: str,
    invert: bool = False,
) -> CandidateFitCheck:
    """PASS when required <= available, unless invert (then available <= required)."""
    if required is None or available is None:
        return CandidateFitCheck(
            name=name,
            status=CandidateFitStatus.NA,
            required_mm=required,
            available_mm=available,
            message=unknown_msg,
        )
    ok = (available <= required) if invert else (required <= available)
    return CandidateFitCheck(
        name=name,
        status=CandidateFitStatus.PASS if ok else CandidateFitStatus.FAIL,
        required_mm=required,
        available_mm=available,
        message=pass_msg if ok else fail_msg,
    )


def _desk_check(requirements: UserRequirements, candidate: ImportedCandidateGeometry) -> CandidateFitCheck:
    desk = requirements.environment.desk_thickness_mm
    desk_min = candidate.compatible_desk_min_mm
    desk_max = candidate.compatible_desk_max_mm
    if desk is None or desk_min is None or desk_max is None:
        return CandidateFitCheck(
            name="desk_fit",
            status=CandidateFitStatus.NA,
            desk_mm=desk,
            supported_range_mm=(
                (desk_min, desk_max) if desk_min is not None and desk_max is not None else None
            ),
            message="Desk thickness or candidate desk range is unavailable; value was not invented.",
        )
    if desk_min <= desk <= desk_max:
        return CandidateFitCheck(
            name="desk_fit",
            status=CandidateFitStatus.PASS,
            desk_mm=desk,
            supported_range_mm=(desk_min, desk_max),
            message="Desk thickness is inside the candidate clamp range.",
        )
    return CandidateFitCheck(
        name="desk_fit",
        status=CandidateFitStatus.FAIL,
        desk_mm=desk,
        supported_range_mm=(desk_min, desk_max),
        message="Desk thickness is outside the candidate clamp range.",
    )


def _cupholder_checks(
    requirements: UserRequirements, candidate: ImportedCandidateGeometry
) -> List[CandidateFitCheck]:
    payload = _payload_size_mm(requirements)
    inner = candidate.inner_diameter_mm
    return [
        _leq_check(
            "payload_fit",
            payload,
            inner,
            pass_msg="Payload fits holder opening.",
            fail_msg="Payload does not fit holder opening.",
            unknown_msg="Payload or holder inner diameter is unavailable; value was not invented.",
        ),
        _desk_check(requirements, candidate),
        _leq_check(
            "envelope_fit",
            requirements.design_envelope.max_protrusion_mm,
            candidate.clamp_reach_mm,
            invert=True,
            pass_msg="Candidate clamp reach is within the design envelope.",
            fail_msg="Candidate clamp reach exceeds the allowed design envelope.",
            unknown_msg="Envelope or clamp reach is unavailable; value was not invented.",
        ),
    ]


def _hook_checks(
    requirements: UserRequirements, candidate: ImportedCandidateGeometry
) -> List[CandidateFitCheck]:
    """Strap hook: opening, clamp range, outward reach, declared load rating. No inner_diameter."""
    return [
        _leq_check(
            "payload_fit",
            _payload_size_mm(requirements),
            _hook_opening_mm(candidate),
            pass_msg="Strap / handle width fits the measured hook opening.",
            fail_msg="Strap / handle width does not fit the measured hook opening.",
            unknown_msg=(
                "Strap width or the candidate's measured hook opening is unavailable; "
                "value was not invented. Holder inner diameter is not a hook metric."
            ),
        ),
        _desk_check(requirements, candidate),
        _leq_check(
            "envelope_fit",
            requirements.design_envelope.max_protrusion_mm,
            _hook_reach_mm(candidate),
            invert=True,
            pass_msg="Hook outward reach is within the allowed protrusion.",
            fail_msg="Hook outward reach exceeds the allowed design envelope.",
            unknown_msg="Envelope or hook reach is unavailable; value was not invented.",
        ),
        _leq_check(
            "load_rating",
            _payload_mass_kg(requirements),
            _dim(candidate, "nominal_concept_target_load"),
            pass_msg="Payload mass is within the candidate's declared concept load rating.",
            fail_msg="Payload mass exceeds the candidate's declared concept load rating.",
            unknown_msg=(
                "Payload mass or the candidate's declared concept load rating is unavailable; "
                "value was not invented."
            ),
        ),
    ]


def _shelf_checks(
    requirements: UserRequirements, candidate: ImportedCandidateGeometry
) -> List[CandidateFitCheck]:
    """Shelf: payload footprint vs platform, envelope vs declared platform length. No cup diameter."""
    return [
        _leq_check(
            "payload_fit",
            _payload_size_mm(requirements),
            _platform_min_mm(candidate),
            pass_msg="Payload footprint fits the declared platform.",
            fail_msg="Payload footprint does not fit the declared platform.",
            unknown_msg=(
                "Payload footprint or the candidate's platform length/width is unavailable; "
                "value was not invented. Holder inner diameter is not a shelf metric."
            ),
        ),
        _leq_check(
            "envelope_fit",
            requirements.design_envelope.max_protrusion_mm,
            _shelf_reach_mm(candidate),
            invert=True,
            pass_msg="Platform length is within the allowed envelope.",
            fail_msg="Platform length exceeds the allowed design envelope.",
            unknown_msg="Envelope or platform length is unavailable; value was not invented.",
        ),
    ]


_FIT_FAMILIES = {
    "cupholder": _cupholder_checks,
    "desk_bag_hook": _hook_checks,
    "stapler_shelf": _shelf_checks,
}


def fit_family_for(candidate: ImportedCandidateGeometry) -> str:
    """Which declared check family this candidate uses. Task/name, not UI branching."""
    task = (candidate.task or candidate.candidate_name or "").strip().lower()
    if task in _FIT_FAMILIES:
        return task
    return "cupholder"


def check_candidate_fit(
    requirements: UserRequirements,
    candidate: ImportedCandidateGeometry,
) -> CandidateFitResult:
    family = fit_family_for(candidate)
    checks = _FIT_FAMILIES[family](requirements, candidate)

    failed = [check for check in checks if check.status == CandidateFitStatus.FAIL]
    unknown = [check for check in checks if check.status == CandidateFitStatus.NA]
    # "No failed checks" is not the same as "checks passed". A candidate whose checks
    # all returned n/a used to report that it fits the resolved requirements, having
    # verified nothing at all.
    fits = bool(checks) and not failed and not unknown
    if fits:
        message = (
            "Imported candidate geometry passes the checks that could be run "
            f"({', '.join(c.name for c in checks)}). These cover the {family} fit "
            "metrics only; usability and assembly clearance are not checked."
        )
    elif failed:
        details = "; ".join(check.message for check in failed if check.message)
        message = "CANDIDATE GEOMETRY REJECTED. " + details
    else:
        names = ", ".join(check.name for check in unknown)
        message = (
            f"CANDIDATE FIT UNKNOWN. Not verified: {names}. Missing dimensions were not "
            "invented, so the candidate is not accepted on the strength of unrun checks."
        )
    return CandidateFitResult(fits=fits, checks=checks, message=message)


# Per-check clarification copy. Hook/shelf must not ask for holder inner diameter.
_FIT_QUESTIONS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "cupholder": {
        "payload_fit": (
            "bottle_diameter_mm",
            "Payload does not fit holder opening. Revise payload diameter or use a different candidate.",
            "Payload fit could not be checked: the payload diameter or the candidate's "
            "inner diameter is unknown. Give the payload diameter in mm.",
        ),
        "desk_fit": (
            "desk_thickness_mm",
            "Desk thickness is outside the imported candidate clamp range. Revise desk thickness or use a different candidate.",
            "Desk fit could not be checked: the desk thickness or the candidate's supported "
            "clamp range is unknown. Give the desk thickness in mm.",
        ),
        "envelope_fit": (
            "max_protrusion_mm",
            "Candidate clamp reach exceeds the allowed protrusion. Revise the design envelope or use a different candidate.",
            "Envelope fit could not be checked: the allowed protrusion or the candidate's "
            "clamp reach is unknown. Give the maximum protrusion in mm.",
        ),
    },
    "desk_bag_hook": {
        "payload_fit": (
            "bottle_diameter_mm",
            "Strap / handle width does not fit the measured hook opening. Revise the strap width or use a different candidate.",
            "Strap fit could not be checked: the strap width or the candidate's measured "
            "hook opening is unknown. Give the strap / handle width in mm.",
        ),
        "desk_fit": (
            "desk_thickness_mm",
            "Desk thickness is outside the imported candidate clamp range. Revise desk thickness or use a different candidate.",
            "Desk fit could not be checked: the desk thickness or the candidate's supported "
            "clamp range is unknown. Give the desk thickness in mm.",
        ),
        "envelope_fit": (
            "max_protrusion_mm",
            "Hook reach exceeds the allowed protrusion. Revise the design envelope or use a different candidate.",
            "Envelope fit could not be checked: the allowed protrusion or the hook reach "
            "is unknown. Give the maximum protrusion in mm.",
        ),
        "load_rating": (
            "filled_bottle_mass_kg",
            "Payload mass exceeds the candidate's declared concept load rating. Revise the payload mass or use a different candidate.",
            "Load rating could not be checked: the payload mass or the candidate's declared "
            "concept load is unknown. Give the payload mass in kg.",
        ),
    },
    "stapler_shelf": {
        "payload_fit": (
            "bottle_diameter_mm",
            "Payload footprint does not fit the declared platform. Revise the footprint or use a different candidate.",
            "Platform fit could not be checked: the payload footprint or the candidate's "
            "platform size is unknown. Give the payload footprint width in mm.",
        ),
        "envelope_fit": (
            "max_protrusion_mm",
            "Platform length exceeds the allowed envelope. Revise the design envelope or use a different candidate.",
            "Envelope fit could not be checked: the allowed envelope or the platform length "
            "is unknown. Give the maximum protrusion in mm.",
        ),
    },
}


def candidate_fit_questions(result: CandidateFitResult, family: str = "cupholder") -> List[ClarificationQuestion]:
    questions: List[ClarificationQuestion] = []
    by_name = {check.name: check for check in result.checks}
    prompts = _FIT_QUESTIONS.get(family) or _FIT_QUESTIONS["cupholder"]
    for name, check in by_name.items():
        prompt = prompts.get(name)
        if prompt is None:
            continue
        field, fail_q, unknown_q = prompt
        if check.status == CandidateFitStatus.FAIL:
            questions.append(ClarificationQuestion(field=field, question=fail_q, priority="high"))
        elif check.status == CandidateFitStatus.NA:
            questions.append(ClarificationQuestion(field=field, question=unknown_q, priority="high"))

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
