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
    MissingInformation,
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
