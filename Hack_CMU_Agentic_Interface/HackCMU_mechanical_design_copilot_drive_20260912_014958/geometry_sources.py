"""UI-only fixture assembly for geometry source modes.

Does not change Orchestrator transition or consistency semantics.
Golden Fixture injects the existing Yujie fixture. Adaptive Synthetic Mock
leaves geometry unset so GeometryAgent builds GeometryOutput from requirements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fixtures import load_integration_fixtures
from imported_candidate import load_imported_candidate
from schemas import ImportedCandidateGeometry, IntegrationFixtures
from state import DesignState

GEOM_ADAPTIVE = "Adaptive Synthetic Mock"
GEOM_IMPORTED = "Imported Candidate Geometry"
GEOM_GENERATED = "Generated Warm Start (Grok)"
GEOM_GOLDEN = "Golden Fixture"
GEOM_LIVE = "Live Geometry"

GEOM_MODE_OPTIONS = [GEOM_ADAPTIVE, GEOM_IMPORTED, GEOM_GENERATED, GEOM_GOLDEN, GEOM_LIVE]

GOLDEN_PROVENANCE = "Fixed golden integration fixture. Regression / integration test."
ADAPTIVE_PROVENANCE = (
    "Synthetic geometry generated from user-entered requirements. "
    "The uploaded image was NOT geometrically reconstructed."
)
IMPORTED_PROVENANCE = (
    "external generated concept geometry. "
    "Not Yujie GeometryOutput and not reconstructed scene geometry."
)
LIVE_PROVENANCE = "Live Geometry is not connected yet."
GENERATED_PROVENANCE = (
    "Warm-start mesh generated from the structured requirements: the reasoning model writes a "
    "parametric trimesh script that is executed and validated (watertight, single body, inside "
    "the envelope). Not reconstructed scene geometry; mesh vertices are not LLM-emitted."
)

FIELD_LABELS = {
    "filled_bottle_mass_kg": "Filled bottle / payload mass (kg)",
    "bottle_diameter_mm": "Bottle / payload diameter (mm)",
    "bottle_height_mm": "Bottle / payload height (mm)",
    "desk_thickness_mm": "Desk / mounting-surface thickness (mm)",
    "attachment_method": "Attachment method (clamp / screws / adhesive)",
    "allowed_contact_region": "Allowed mount region on the desk",
    "max_protrusion_mm": "Maximum outward protrusion from desk edge (mm)",
    "manufacturing_method": "Manufacturing method",
    "material": "Material",
}

_FLOAT_ABS_TOL = 1e-3

_COMPARE_FIELDS = [
    (
        "Bottle / payload diameter (mm)",
        lambda req: req.object_geometry.bottle_diameter_mm,
        lambda geom: geom.payload_object.bottle_diameter_mm,
    ),
    (
        "Filled payload mass (kg)",
        lambda req: req.payload.filled_mass_kg,
        lambda geom: geom.payload_object.filled_mass_kg,
    ),
    (
        "Desk / mounting-surface thickness (mm)",
        lambda req: req.environment.desk_thickness_mm,
        lambda geom: geom.environment.desk_thickness_mm,
    ),
    (
        "Maximum outward protrusion from desk edge (mm)",
        lambda req: req.design_envelope.max_protrusion_mm,
        lambda geom: geom.design_envelope.max_protrusion_mm,
    ),
]


def effective_geometry_mode(mode: Optional[str]) -> str:
    if mode in GEOM_MODE_OPTIONS:
        return mode
    return GEOM_ADAPTIVE


def uses_synthetic_geometry(mode: Optional[str]) -> bool:
    return effective_geometry_mode(mode) in {GEOM_ADAPTIVE, GEOM_IMPORTED, GEOM_GENERATED}


def fixtures_for_geometry_mode(mode: Optional[str], topology_live: bool = False) -> IntegrationFixtures:
    """`topology_live` drops the topology/CAD fixtures so the real tools run (to_agent)."""
    base = load_integration_fixtures()
    if uses_synthetic_geometry(mode):
        # Keep registration / topology / CAD fixtures. Drop geometry so GeometryAgent
        # synthesizes from UserRequirements. Drop analysis so the golden FEM load
        # vector cannot fight a different payload mass.
        return IntegrationFixtures(
            registration=base.registration,
            geometry=None,
            analysis=None,
            topology=None if topology_live else base.topology,
            cad=None if topology_live else base.cad,
        )
    if topology_live:
        return IntegrationFixtures(
            registration=base.registration,
            geometry=base.geometry,
            analysis=base.analysis,
            topology=None,
            cad=None,
        )
    return base


def imported_candidate_for_mode(mode: Optional[str]) -> Optional[ImportedCandidateGeometry]:
    if effective_geometry_mode(mode) != GEOM_IMPORTED:
        return None
    return load_imported_candidate()


def geometry_provenance_text(mode: Optional[str]) -> str:
    effective = effective_geometry_mode(mode)
    if effective == GEOM_IMPORTED:
        return IMPORTED_PROVENANCE
    if effective == GEOM_ADAPTIVE:
        return ADAPTIVE_PROVENANCE
    if effective == GEOM_LIVE:
        return LIVE_PROVENANCE
    if effective == GEOM_GENERATED:
        return GENERATED_PROVENANCE
    return GOLDEN_PROVENANCE


def golden_fixture_assumptions() -> Dict[str, Any]:
    geom = load_integration_fixtures().geometry
    assert geom is not None
    return {
        "payload_mass_kg": geom.payload_object.filled_mass_kg,
        "bottle_diameter_mm": geom.payload_object.bottle_diameter_mm,
        "desk_thickness_mm": geom.environment.desk_thickness_mm,
        "max_protrusion_mm": geom.design_envelope.max_protrusion_mm,
        "attachment": "clamp / no drilling",
        "manufacturing": "FDM / 3d_print, PLA",
        "frame": geom.coordinate_frame,
    }


def requirement_geometry_rows(state: DesignState) -> List[Dict[str, Any]]:
    """Display-only comparison. Does not decide whether values agree."""
    req = state.requirements
    geom = state.geometry
    if req is None or geom is None:
        return []
    rows: List[Dict[str, Any]] = []
    for label, req_get, geom_get in _COMPARE_FIELDS:
        left = req_get(req)
        right = geom_get(geom)
        if left is None and right is None:
            continue
        disagree = (
            left is not None
            and right is not None
            and abs(float(left) - float(right)) > _FLOAT_ABS_TOL
        )
        rows.append(
            {
                "label": label,
                "requirement": left,
                "geometry": right,
                "disagree": disagree,
            }
        )
    return rows


def format_mismatch_message(state: DesignState) -> str:
    rows = requirement_geometry_rows(state)
    lines = ["Requirement vs Geometry"]
    any_disagree = False
    for row in rows:
        left = row["requirement"]
        right = row["geometry"]
        mark = "  !=  " if row["disagree"] else "  ==  "
        if row["disagree"]:
            any_disagree = True
        unit = " mm" if "(mm)" in row["label"] else (" kg" if "(kg)" in row["label"] else "")
        lines.append(f"{row['label']}:")
        lines.append(f"{_fmt(left)}{unit}{mark}{_fmt(right)}{unit}")
    if any_disagree or state.contract_error:
        lines.append("")
        lines.append(
            "The system refuses to silently choose between conflicting engineering inputs."
        )
    if state.contract_error:
        lines.append("")
        lines.append(state.contract_error)
    return "\n".join(lines)


def chat_error_key(state: DesignState) -> Tuple[str, str]:
    return (state.stage.value, state.contract_error or "")


def _fmt(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
