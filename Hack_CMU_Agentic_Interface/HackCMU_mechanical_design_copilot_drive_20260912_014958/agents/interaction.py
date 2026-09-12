"""Interaction Agent: requirements, clarifications, and scope gating.

Rule-based placeholder for later LLM reasoning. Does not chat with other agents.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional, Tuple

from schemas import (
    ClarificationQuestion,
    InteractionDecision,
    InteractionResult,
    RequirementsUpdate,
    UserRequirements,
    VisionRequest,
)

# --------------------------------------------------------------------------- hazard floor
#
# Two tiers, because a single substring list cannot separate "a step to reach the sink"
# (a person stands on it) from "a laptop stand on my desk" (an object rests on it).
#
#   HARD_HAZARDS      unambiguous: refuse outright, and reasoning may not overturn it.
#   AMBIGUOUS_HAZARDS could go either way: never decided by keyword. They escalate — live
#                     reasoning classifies them, and failing that the user is asked a direct
#                     question. Nothing that carries a person proceeds undetermined, but an
#                     ordinary object holder is no longer refused for containing "bench".
#
# Patterns are regexes matched with word boundaries, so "bench" does not fire on
# "workbench" and "gun" does not fire on "gun-metal".

HARD_HAZARDS: List[Tuple[str, str]] = [
    (r"\bstep[- ]?stools?\b", "Human-supporting furniture is out of scope."),
    (r"\bfoot[- ]?stools?\b", "Human-supporting furniture is out of scope."),
    (r"\bfoot[- ]?rests?\b", "Human-supporting furniture is out of scope."),
    (r"\bbar[- ]?stools?\b", "Human-supporting furniture is out of scope."),
    (r"\bstools?\b", "Human-supporting furniture is out of scope."),
    (r"\bchairs?\b", "Human-supporting furniture is out of scope."),
    (r"\b(child|car|booster)[- ]seats?\b", "Human-supporting furniture is out of scope."),
    (r"\bladders?\b", "Human-supporting structures are out of scope."),
    (r"\bgrab[- ]bars?\b", "Human-supporting / mobility aids are out of scope."),
    (r"\bhand[- ]?rails?\b", "Human-supporting / mobility aids are out of scope."),
    (r"\bkneelers?\b", "Human-supporting furniture is out of scope."),
    (r"\b(support|hold|bear|take)s?\s+(my|his|her|their|your|a person'?s|someone'?s)\s+(body\s+)?weight\b",
     "Human-supporting structures are out of scope."),
    # "stand" is only a hazard as a VERB. "to stand on", "stands on it", "standing on this"
    # are a person; "a laptop stand on my desk" is a noun and must not be caught here.
    # A determiner after "on" means the thing stands on a surface ("a laptop stand on my
    # desk", "a rack that sits on the shelf"). Anything else — "to stand on", "sit on it",
    # "a bracket I can sit on" — is a person putting their weight on it.
    (r"\b(sit|sits|sitting|stand|stands|standing|kneel|kneels|kneeling|perch|perches)\s+on\b"
     r"(?!\s+(my|the|a|an|your|his|her|its|their|each|either)\b)",
     "Anything a person puts their weight on is out of scope."),
    (r"\bweapons?\b", "Weapons are unsupported."),
    (r"\bfirearms?\b", "Weapons are unsupported."),
    (r"\bguns?\b", "Weapons are unsupported."),
    (r"\bexplosives?\b", "This request is unsupported."),
]

# (pattern, hazard class, the question that settles it)
AMBIGUOUS_HAZARDS: List[Tuple[str, str, str]] = [
    (r"\bstep\b|\bsteps\b|\bclimb|\bstep up\b",
     "human_support",
     "Will a person put any of their body weight on this part — standing, sitting, leaning "
     "or pulling themselves up on it?"),
    (r"\bseat\b|\bbench\b|\bperch\b",
     "human_support",
     "Will a person sit on or otherwise put their body weight on this part?"),
    # A person word on its own is not a signal — "a rack for my kid's books" holds books.
    # It only matters together with a weight-bearing action, which the hard rule above covers.
    (r"\bceiling\b|\boverhead\b|\babove head\b|\bjoist\b|\brafter\b",
     "overhead",
     "Will this be mounted above head height, where the part or its load could fall on someone?"),
    (r"\bimpact\b|\bcrash\b|\bshock load",
     "impact",
     "Will this part take sudden impact or shock loading, rather than a steady static load?"),
    (r"\bfatigue\b|\bcyclic\b|\brepeated load",
     "impact",
     "Will this part see repeated load cycles where fatigue matters?"),
    (r"\bmotor\b|\bhinge mechanism\b|\bactuator\b|\bspring loaded\b",
     "dynamic",
     "Does this part contain or drive a moving mechanism, rather than being a static fixture?"),
]

# Back-compat: the flat list other modules import. Hard hazards only — the ambiguous ones
# are deliberately not auto-refusals any more.
UNSUPPORTED_PATTERNS: List[Tuple[str, str]] = list(HARD_HAZARDS)


# A person word AND a weight-bearing action in the same request is a person being supported,
# even when no furniture noun appears: "a step to help my kid reach the sink". Neither half
# alone is evidence — "a laptop stand on my desk" has the action and no person; "a rack for
# my kid's books" has the person and no action.
PERSON_WORDS = r"\b(kid|kids|child|children|toddler|baby|person|people|human|adult|myself|himself|herself)\b"
WEIGHT_BEARING_ACTS = r"\b(stand|stands|standing|step|steps|stepping|sit|sits|sitting|climb|climbs|climbing|kneel|perch|reach|reaches)\b"


class HazardSignal(NamedTuple):
    """`verdict` is one of clear | refuse | ambiguous."""

    verdict: str
    hazard_class: str = "none"
    reasons: List[str] = []
    question: str = ""


def classify_hazard(text: str) -> HazardSignal:
    """Keyword floor only. It refuses the unmistakable and defers everything else."""
    lowered = (text or "").lower()
    for pattern, reason in HARD_HAZARDS:
        if re.search(pattern, lowered):
            return HazardSignal("refuse", "human_support", [reason], "")
    if re.search(PERSON_WORDS, lowered) and re.search(WEIGHT_BEARING_ACTS, lowered):
        return HazardSignal(
            "refuse",
            "human_support",
            ["the request describes a person standing, stepping, sitting or climbing on the "
             "part, which is out of scope however it is worded"],
            "",
        )
    for pattern, hazard_class, question in AMBIGUOUS_HAZARDS:
        if re.search(pattern, lowered):
            return HazardSignal(
                "ambiguous",
                hazard_class,
                [f"wording suggests a possible {hazard_class.replace('_', ' ')} hazard; not decided by keyword"],
                question,
            )
    return HazardSignal("clear")

# (keyword in the request, payload description, payload kind) — first match wins.
PAYLOAD_KEYWORDS: List[Tuple[str, str, str]] = [
    ("cup holder", "bottle", "cylinder"),
    ("bottle", "bottle", "cylinder"),
    ("mug", "mug", "cylinder"),
    ("bag", "bag", "strap"),
    ("hook", "bag", "strap"),
    ("headphone", "headphones", "strap"),
    ("stapler", "stapler", "box"),
    ("shelf", "stapler", "box"),
    ("phone", "phone", "box"),
    ("laptop", "laptop", "box"),
    ("monitor", "monitor", "box"),
]

# The size question depends on the payload kind; the field name stays `bottle_diameter_mm`
# for contract compatibility (it is the payload's characteristic size in mm).
SIZE_QUESTIONS = {
    "cylinder": "What is the {payload} diameter (mm)?",
    "strap": "What is the {payload} strap / handle width that rests on the hook (mm)?",
    "box": "What is the {payload} footprint width (mm)?",
}

REQUIRED_FIELDS = [
    (
        "filled_bottle_mass_kg",
        "What is the {payload} mass (kg), filled / fully loaded?",
        "high",
    ),
    (
        "bottle_diameter_mm",
        "What is the {payload} characteristic size (mm) — diameter, strap width or footprint?",
        "high",
    ),
    (
        "desk_thickness_mm",
        "What is the desk / mounting-surface thickness (mm)?",
        "high",
    ),
    (
        "attachment_method",
        "How should it attach (clamp / screws / adhesive)?",
        "high",
    ),
    (
        "allowed_contact_region",
        "Where on the desk may it mount (e.g. front edge)?",
        "high",
    ),
    (
        "max_protrusion_mm",
        "What is the maximum allowed protrusion from the desk (mm)?",
        "high",
    ),
    (
        "manufacturing_method",
        "How will it be made (e.g. 3d_print)?",
        "medium",
    ),
]

TASK_CUPHOLDER = "cupholder"
TASK_DESK_HOOK = "desk_hook"
TASK_DESK_SHELF = "desk_shelf"
TASK_BED_HANDLE = "bed_handle"
TASK_WALL_SHELF = "wall_shelf"
TASK_GENERIC = "generic"


def classify_design_task(text: str) -> str:
    """Choose a requirement pack from the request. Unknown tasks stay generic."""
    lowered = (text or "").lower()
    bedish = any(token in lowered for token in ("bed", "mattress", "headboard"))
    assist = any(token in lowered for token in ("handle", "assist", "get up", "wake", "stand up"))
    if bedish and assist:
        return TASK_BED_HANDLE
    if "shelf" in lowered and any(token in lowered for token in ("wall", "router")):
        return TASK_WALL_SHELF
    if any(token in lowered for token in ("cup holder", "cupholder", "bottle", "mug")):
        return TASK_CUPHOLDER
    if any(token in lowered for token in ("hook", "bag")):
        return TASK_DESK_HOOK
    if "shelf" in lowered or "stapler" in lowered:
        return TASK_DESK_SHELF
    return TASK_GENERIC


def _spec(
    field: str,
    question: str,
    *,
    kind: str = "text",
    unit: Optional[str] = None,
    options: Optional[List[str]] = None,
    reason: str = "",
    source: str = "task_reasoning",
    priority: str = "high",
) -> dict:
    return {
        "field": field,
        "question": question,
        "kind": kind,
        "unit": unit,
        "options": options,
        "reason": reason,
        "source": source,
        "priority": priority,
    }


def clarification_specs_for_task(task: str, payload: str = "payload") -> List[dict]:
    """Task-relevant questions only. Demo cup-holder fields are not a global default."""
    if task == TASK_CUPHOLDER:
        return [
            _spec("filled_bottle_mass_kg", f"What is the {payload} mass (kg), filled / fully loaded?", kind="numeric", unit="kg", reason="load_case"),
            _spec("bottle_diameter_mm", SIZE_QUESTIONS.get("cylinder", "What is the payload diameter (mm)?").format(payload=payload), kind="numeric", unit="mm", reason="functional_geometry"),
            _spec("desk_thickness_mm", "What is the desk / mounting-surface thickness (mm)?", kind="numeric", unit="mm", reason="attachment_interface"),
            _spec("attachment_method", "How should it attach (clamp / screws / adhesive)?", kind="categorical", options=["clamp", "screws", "adhesive"], reason="mounting_method"),
            _spec("allowed_contact_region", "Where on the desk may it mount (e.g. front edge)?", kind="categorical", options=["desk_front_edge", "desk_side_edge", "desk_underside", "desk_top"], reason="attachment_interface"),
            _spec("max_protrusion_mm", "What is the maximum allowed protrusion from the desk (mm)?", kind="numeric", unit="mm", reason="design_envelope"),
            _spec("manufacturing_method", "How will it be made (e.g. 3d_print)?", kind="categorical", options=["3d_print", "fdm", "sla"], reason="manufacturing_constraint", priority="medium"),
        ]
    if task == TASK_DESK_HOOK:
        return [
            _spec("filled_bottle_mass_kg", "What bag / load mass should it support (kg)?", kind="numeric", unit="kg", reason="load_case"),
            _spec("desk_thickness_mm", "What is the desk thickness (mm)?", kind="numeric", unit="mm", reason="attachment_interface"),
            _spec("attachment_method", "How should it mount?", kind="categorical", options=["clamp", "screws", "adhesive"], reason="mounting_method"),
            _spec("allowed_contact_region", "Where on the desk may it mount?", kind="categorical", options=["desk_front_edge", "desk_side_edge", "desk_underside"], reason="attachment_interface"),
            _spec("max_protrusion_mm", "How far may it stick out from the desk (mm)?", kind="numeric", unit="mm", reason="design_envelope"),
            _spec("manufacturing_method", "How will it be made?", kind="categorical", options=["3d_print", "fdm", "sla"], reason="manufacturing_constraint", priority="medium"),
        ]
    if task == TASK_DESK_SHELF:
        return [
            _spec("filled_bottle_mass_kg", "What load should the shelf support (kg)?", kind="numeric", unit="kg", reason="load_case"),
            _spec("payload_size_mm", "What is the object footprint width (mm)?", kind="numeric", unit="mm", reason="functional_geometry"),
            _spec("desk_thickness_mm", "What is the desk thickness (mm)?", kind="numeric", unit="mm", reason="attachment_interface"),
            _spec("attachment_method", "How should it attach?", kind="categorical", options=["clamp", "screws", "adhesive"], reason="mounting_method"),
            _spec("allowed_contact_region", "Where on the desk may it mount?", kind="categorical", options=["desk_front_edge", "desk_top", "desk_underside"], reason="attachment_interface"),
            _spec("max_protrusion_mm", "How far may it extend (mm)?", kind="numeric", unit="mm", reason="design_envelope"),
            _spec("manufacturing_method", "How will it be made?", kind="categorical", options=["3d_print", "fdm", "sla"], reason="manufacturing_constraint", priority="medium"),
        ]
    if task == TASK_BED_HANDLE:
        return [
            _spec("supported_load_kg", "What load should it support (kg)?", kind="numeric", unit="kg", reason="load_case"),
            _spec("attachment_structure", "What should it attach to?", kind="categorical", options=["bed_frame", "headboard", "mattress_edge", "wall", "other"], reason="attachment_interface"),
            _spec("handle_location", "Where should the handle be?", kind="categorical", options=["bedside", "headboard", "mattress_edge", "other"], reason="functional_geometry"),
            _spec("mounting_region", "Where can it mount?", kind="categorical", options=["bed_frame", "headboard", "wall", "other"], reason="attachment_interface"),
            _spec("drilling_allowed", "Is drilling allowed?", kind="boolean", reason="user_constraint"),
            _spec("required_reach_mm", "How much reach does the handle need (mm)?", kind="numeric", unit="mm", reason="clearance"),
            _spec("manufacturing_method", "How will it be made?", kind="categorical", options=["3d_print", "fdm", "sla"], reason="manufacturing_constraint", priority="medium"),
        ]
    if task == TASK_WALL_SHELF:
        return [
            _spec("supported_load_kg", "What load should it support (kg)?", kind="numeric", unit="kg", reason="load_case"),
            _spec("payload_size_mm", "What is the object size (mm)?", kind="numeric", unit="mm", reason="functional_geometry"),
            _spec("attachment_method", "How should it attach to the wall?", kind="categorical", options=["screws", "adhesive", "other"], reason="mounting_method"),
            _spec("mounting_region", "Where on the wall may it mount?", kind="text", reason="attachment_interface"),
            _spec("wall_clearance_mm", "How much clearance is needed (mm)?", kind="numeric", unit="mm", reason="clearance"),
            _spec("manufacturing_method", "How will it be made?", kind="categorical", options=["3d_print", "fdm", "sla"], reason="manufacturing_constraint", priority="medium"),
        ]
    return [
        _spec("supported_load_kg", "What load should it support (kg)?", kind="numeric", unit="kg", reason="load_case"),
        _spec("attachment_structure", "What should it attach to?", kind="text", reason="attachment_interface"),
        _spec("attachment_method", "How should it attach?", kind="categorical", options=["clamp", "screws", "adhesive", "other"], reason="mounting_method"),
        _spec("required_reach_mm", "What functional reach is required (mm)?", kind="numeric", unit="mm", reason="clearance"),
        _spec("max_protrusion_mm", "What is the maximum allowed overall size (mm)?", kind="numeric", unit="mm", reason="design_envelope", priority="medium"),
        _spec("manufacturing_method", "How will it be made?", kind="categorical", options=["3d_print", "fdm", "sla"], reason="manufacturing_constraint", priority="medium"),
    ]


class InteractionAgent:
    """Extracts known requirements and decides PROCEED / REQUEST / REJECT."""

    def assess(self, message: str, existing: Optional[UserRequirements] = None) -> InteractionResult:
        lowered = message.lower()
        hazard = classify_hazard(message)
        if hazard.verdict == "refuse":
            req = existing or UserRequirements(user_message=message, description=message)
            return InteractionResult(
                decision=InteractionDecision.REJECT_OR_ESCALATE,
                requirements=req,
                reject_reason=hazard.reasons[0] if hazard.reasons else "Out of scope.",
            )

        req = existing or UserRequirements()
        req.user_message = message
        if not req.description:
            req.description = message.strip()
        req.task_kind = classify_design_task(message)
        self._extract_keywords(req, lowered)
        return self._decide(req)

    def apply_update(
        self, requirements: UserRequirements, update: RequirementsUpdate
    ) -> InteractionResult:
        extras = dict(requirements.task_answers or {})
        if update.filled_bottle_mass_kg is not None:
            requirements.payload.filled_mass_kg = update.filled_bottle_mass_kg
        if update.bottle_diameter_mm is not None:
            requirements.object_geometry.bottle_diameter_mm = update.bottle_diameter_mm
        if update.bottle_height_mm is not None:
            requirements.object_geometry.bottle_height_mm = update.bottle_height_mm
        if update.desk_thickness_mm is not None:
            requirements.environment.desk_thickness_mm = update.desk_thickness_mm
        if update.attachment_method is not None:
            requirements.attachment.method = update.attachment_method
        if update.allowed_contact_region is not None:
            requirements.attachment.allowed_contact_region = update.allowed_contact_region
        if update.attachment_notes is not None:
            requirements.attachment.notes = update.attachment_notes
        if update.max_protrusion_mm is not None:
            requirements.design_envelope.max_protrusion_mm = update.max_protrusion_mm
        if update.manufacturing_method is not None:
            requirements.manufacturing.method = update.manufacturing_method
        if update.material is not None:
            requirements.manufacturing.material = update.material
        if update.max_part_mass_kg is not None:
            requirements.part_mass.max_part_mass_kg = update.max_part_mass_kg
        if update.supported_load_kg is not None:
            extras["supported_load_kg"] = update.supported_load_kg
            if requirements.payload.filled_mass_kg is None:
                requirements.payload.filled_mass_kg = update.supported_load_kg
        if update.payload_size_mm is not None:
            extras["payload_size_mm"] = update.payload_size_mm
            if requirements.object_geometry.bottle_diameter_mm is None:
                requirements.object_geometry.bottle_diameter_mm = update.payload_size_mm
        if update.required_reach_mm is not None:
            extras["required_reach_mm"] = update.required_reach_mm
        if update.wall_clearance_mm is not None:
            extras["wall_clearance_mm"] = update.wall_clearance_mm
        if update.attachment_structure is not None:
            extras["attachment_structure"] = update.attachment_structure
            if requirements.attachment.allowed_contact_region is None:
                requirements.attachment.allowed_contact_region = update.attachment_structure
        if update.handle_location is not None:
            extras["handle_location"] = update.handle_location
        if update.mounting_region is not None:
            extras["mounting_region"] = update.mounting_region
            if requirements.attachment.allowed_contact_region is None:
                requirements.attachment.allowed_contact_region = update.mounting_region
        if update.drilling_allowed is not None:
            extras["drilling_allowed"] = update.drilling_allowed
        requirements.task_answers = extras
        if not requirements.task_kind:
            requirements.task_kind = classify_design_task(
                requirements.user_message or requirements.description
            )
        return self._decide(requirements)

    def _extract_keywords(self, req: UserRequirements, lowered: str) -> None:
        task = req.task_kind or classify_design_task(lowered)
        req.task_kind = task
        if task == TASK_CUPHOLDER:
            for keyword, description, kind in PAYLOAD_KEYWORDS:
                if keyword in {"hook", "bag", "shelf", "stapler"}:
                    continue
                if keyword in lowered:
                    req.payload.description = description
                    req.object_geometry.kind = kind
                    break
            else:
                req.payload.description = req.payload.description or "bottle"
                req.object_geometry.kind = req.object_geometry.kind or "cylinder"
            req.environment.kind = "desk_plane"
        elif task == TASK_DESK_HOOK:
            req.payload.description = "bag"
            req.object_geometry.kind = "strap"
            req.environment.kind = "desk_plane"
        elif task == TASK_DESK_SHELF:
            req.payload.description = "object"
            req.object_geometry.kind = "box"
            req.environment.kind = "desk_plane"
        elif task == TASK_WALL_SHELF:
            req.payload.description = "router" if "router" in lowered else "object"
            req.object_geometry.kind = "box"
            req.environment.kind = "wall"
        elif task == TASK_BED_HANDLE:
            req.payload.description = "assist_load"
            req.object_geometry.kind = "handle"
            req.environment.kind = "bed"
        if "1 l" in lowered or "1l" in lowered or "one liter" in lowered:
            req.payload.volume_l = 1.0
        if "clamp" in lowered:
            req.attachment.method = "clamp"
        if "screw" in lowered:
            req.attachment.method = "screws"
        if "adhesive" in lowered or "tape" in lowered:
            req.attachment.method = "adhesive"
        if "3d print" in lowered or "3d-print" in lowered or "fdm" in lowered:
            req.manufacturing.method = "3d_print"

    def _decide(self, req: UserRequirements) -> InteractionResult:
        missing = self._missing(req)
        questions = [
            ClarificationQuestion(
                field=spec["field"],
                question=spec["question"],
                priority=spec.get("priority") or "high",
                kind=spec.get("kind"),
                unit=spec.get("unit"),
                options=spec.get("options"),
                reason=spec.get("reason"),
                source=spec.get("source"),
            )
            for spec in missing
        ]
        vision = VisionRequest(
            want_environment_photo=True,
            want_reference_object=True,
            requested_measurements=[spec["field"] for spec in missing],
            notes=(
                "Vision is not implemented. A later step may request a photo, "
                "a known-size reference object, and critical measurements."
            ),
        )
        req.vision = vision

        if questions:
            return InteractionResult(
                decision=InteractionDecision.REQUEST_INFORMATION,
                requirements=req,
                questions=questions,
                vision_request=vision,
            )
        resolved_vision = VisionRequest(
            want_environment_photo=False,
            want_reference_object=False,
            requested_measurements=[],
            notes="No outstanding vision request. Vision remains unimplemented.",
        )
        req.vision = resolved_vision
        return InteractionResult(
            decision=InteractionDecision.PROCEED,
            requirements=req,
            vision_request=resolved_vision,
        )

    def _field_value(self, req: UserRequirements, field: str) -> object:
        extras = req.task_answers or {}
        if field in extras and extras[field] not in (None, ""):
            return extras[field]
        mapped = {
            "filled_bottle_mass_kg": req.payload.filled_mass_kg,
            "bottle_diameter_mm": req.object_geometry.bottle_diameter_mm,
            "desk_thickness_mm": req.environment.desk_thickness_mm,
            "attachment_method": req.attachment.method,
            "allowed_contact_region": req.attachment.allowed_contact_region,
            "max_protrusion_mm": req.design_envelope.max_protrusion_mm,
            "manufacturing_method": req.manufacturing.method,
            "supported_load_kg": req.payload.filled_mass_kg,
            "payload_size_mm": req.object_geometry.bottle_diameter_mm,
            "required_reach_mm": extras.get("required_reach_mm"),
            "wall_clearance_mm": extras.get("wall_clearance_mm"),
            "attachment_structure": extras.get("attachment_structure"),
            "handle_location": extras.get("handle_location"),
            "mounting_region": extras.get("mounting_region") or req.attachment.allowed_contact_region,
            "drilling_allowed": extras.get("drilling_allowed"),
        }
        return mapped.get(field)

    def _field_answered(self, req: UserRequirements, field: str) -> bool:
        value = self._field_value(req, field)
        if field == "drilling_allowed":
            return value is not None
        return bool(value)

    def _missing(self, req: UserRequirements) -> List[dict]:
        task = req.task_kind or classify_design_task(req.user_message or req.description)
        payload = req.payload.description or "payload"
        return [
            spec
            for spec in clarification_specs_for_task(task, payload)
            if not self._field_answered(req, spec["field"])
        ]


def missing_requirement_fields(req: UserRequirements) -> List[str]:
    """Authoritative unreadiness list. Geometry and the UI must use this."""
    return [spec["field"] for spec in InteractionAgent()._missing(req)]
