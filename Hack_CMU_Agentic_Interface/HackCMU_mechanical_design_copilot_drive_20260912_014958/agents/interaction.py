"""Interaction Agent: requirements, clarifications, and scope gating.

Rule-based placeholder for later LLM reasoning. Does not chat with other agents.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from schemas import (
    ClarificationQuestion,
    InteractionDecision,
    InteractionResult,
    MissingInformation,
    RequirementsUpdate,
    UserRequirements,
    VisionRequest,
)

# Out of scope for the hackathon MVP.
UNSUPPORTED_PATTERNS: List[Tuple[str, str]] = [
    ("stool", "Human-supporting furniture is out of scope."),
    ("chair", "Human-supporting furniture is out of scope."),
    ("seat", "Human-supporting furniture is out of scope."),
    ("ladder", "Human-supporting structures are out of scope."),
    ("person", "Human-supporting structures are out of scope."),
    ("human", "Human-supporting structures are out of scope."),
    # Body weight is often described without any of the words above ("a step to reach the
    # sink", "something to stand on"). These are the floor; live reasoning may add more.
    ("step stool", "Human-supporting furniture is out of scope."),
    ("step to", "Anything a person stands on is out of scope."),
    ("stand on", "Anything a person stands on is out of scope."),
    ("standing on", "Anything a person stands on is out of scope."),
    ("step up", "Anything a person stands on is out of scope."),
    ("footstool", "Human-supporting furniture is out of scope."),
    ("footrest", "Human-supporting furniture is out of scope."),
    ("kneeler", "Human-supporting furniture is out of scope."),
    ("bench", "Human-supporting furniture is out of scope."),
    ("climb", "Human-supporting structures are out of scope."),
    ("hold my weight", "Human-supporting structures are out of scope."),
    ("support my weight", "Human-supporting structures are out of scope."),
    ("grab bar", "Human-supporting / mobility aids are out of scope."),
    ("handrail", "Human-supporting / mobility aids are out of scope."),
    ("child seat", "Human-supporting furniture is out of scope."),
    ("overhead", "Overhead mounts where a drop could injure are out of scope."),
    ("ceiling", "Overhead mounts where a drop could injure are out of scope."),
    ("impact", "Impact loading is out of scope."),
    ("crash", "Impact loading is out of scope."),
    ("fatigue", "Fatigue analysis is out of scope."),
    ("weapon", "Weapons are unsupported."),
    ("explosive", "This request is unsupported."),
    ("gun", "Weapons are unsupported."),
    ("motor", "Dynamic mechanisms are out of scope."),
    ("hinge mechanism", "Dynamic mechanisms are out of scope."),
]

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


class InteractionAgent:
    """Extracts known requirements and decides PROCEED / REQUEST / REJECT."""

    def assess(self, message: str, existing: Optional[UserRequirements] = None) -> InteractionResult:
        lowered = message.lower()
        for pattern, reason in UNSUPPORTED_PATTERNS:
            if pattern in lowered:
                req = existing or UserRequirements(user_message=message, description=message)
                return InteractionResult(
                    decision=InteractionDecision.REJECT_OR_ESCALATE,
                    requirements=req,
                    reject_reason=reason,
                )

        req = existing or UserRequirements()
        req.user_message = message
        if not req.description:
            req.description = message.strip()
        self._extract_keywords(req, lowered)
        return self._decide(req)

    def apply_update(
        self, requirements: UserRequirements, update: RequirementsUpdate
    ) -> InteractionResult:
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
        return self._decide(requirements)

    def _extract_keywords(self, req: UserRequirements, lowered: str) -> None:
        for keyword, description, kind in PAYLOAD_KEYWORDS:
            if keyword in lowered:
                req.payload.description = description
                req.object_geometry.kind = kind
                break
        if "desk" in lowered:
            req.environment.kind = "desk_plane"
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
            ClarificationQuestion(field=m.field, question=m.reason, priority=m.priority)
            for m in missing
        ]
        vision = VisionRequest(
            want_environment_photo=True,
            want_reference_object=True,
            requested_measurements=[m.field for m in missing],
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

    def _missing(self, req: UserRequirements) -> List[MissingInformation]:
        values = {
            "filled_bottle_mass_kg": req.payload.filled_mass_kg,
            "bottle_diameter_mm": req.object_geometry.bottle_diameter_mm,
            "desk_thickness_mm": req.environment.desk_thickness_mm,
            "attachment_method": req.attachment.method,
            "allowed_contact_region": req.attachment.allowed_contact_region,
            "max_protrusion_mm": req.design_envelope.max_protrusion_mm,
            "manufacturing_method": req.manufacturing.method,
        }
        payload = req.payload.description or "payload"
        missing: List[MissingInformation] = []
        for field, question, priority in REQUIRED_FIELDS:
            if not values.get(field):
                if field == "bottle_diameter_mm":
                    question = SIZE_QUESTIONS.get(req.object_geometry.kind, question)
                missing.append(
                    MissingInformation(
                        field=field, reason=question.format(payload=payload), priority=priority
                    )
                )
        return missing
