"""Reasoning harness: validation, retry-with-feedback, and the product/developer split.

No network: a scripted provider returns canned responses so the contract is testable
offline and provider-agnostically.
"""

from __future__ import annotations

import json
import unittest

import reasoning_deterministic as det
from agents.interaction import classify_hazard
from providers import MockReasoningProvider, ReasoningProvider
from reasoning_contracts import (
    MeasurementPlan,
    ReasoningFailureKind,
    ReasoningMode,
    ReasoningOutcome,
    ReasoningTask,
    ScopeAssessment,
)
from reasoning_harness import call_reasoning, validate_measurement_plan, validate_scope


class ScriptedProvider(ReasoningProvider):
    """Returns queued responses in order; records the conversation it was given."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, responses, configured=True, structured=True):
        self.responses = list(responses)
        self.configured = configured
        self._structured = structured
        self.conversations = []

    def supports_structured(self) -> bool:
        return self._structured

    def structured_json(self, messages, timeout: int = 90) -> str:
        self.conversations.append(list(messages))
        if not self.responses:
            raise RuntimeError("no scripted response left")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def complete(self, role, design_state, reasoning_effort=None):
        raise NotImplementedError


GOOD_PLAN = json.dumps(
    {
        "payload_description": "bicycle",
        "payload_kind": "bicycle",
        "mount_description": "joist",
        "requests": [
            {"field": "bicycle_mass_kg", "question": "Mass?", "unit": "kg",
             "why_it_matters": "sets the load", "affects": "capacity", "priority": "high"},
            {"field": "joist_width_mm", "question": "Width?", "unit": "mm",
             "why_it_matters": "bearing area", "affects": "fit", "priority": "high"},
        ],
        "assumptions": [],
        "notes": "",
    }
)


def _plan_call(provider, mode):
    return call_reasoning(
        task=ReasoningTask.PLAN_MEASUREMENTS, provider=provider, prompt="p",
        schema=MeasurementPlan, mode=mode,
        fallback=lambda: det.plan_measurements("a cup holder"),
        validate=validate_measurement_plan,
    )


class HarnessTests(unittest.TestCase):
    def test_live_success(self) -> None:
        res = _plan_call(ScriptedProvider([GOOD_PLAN]), ReasoningMode.PRODUCT)
        self.assertFalse(res.blocked)
        self.assertEqual(res.trace.outcome, ReasoningOutcome.LIVE)
        self.assertEqual(len(res.trace.attempts), 1)
        self.assertEqual(res.data["payload_kind"], "bicycle")

    def test_retry_feeds_the_error_back(self) -> None:
        bad = json.dumps({"requests": [{"field": "Bad Name", "question": "q", "unit": "mm",
                                        "why_it_matters": "w", "affects": "fit", "priority": "high"}]})
        provider = ScriptedProvider([bad, GOOD_PLAN])
        res = _plan_call(provider, ReasoningMode.PRODUCT)
        self.assertFalse(res.blocked)
        self.assertEqual(res.trace.outcome, ReasoningOutcome.LIVE)
        self.assertEqual(res.trace.attempts[0].failure_kind, ReasoningFailureKind.SEMANTIC_INVALID)
        # the correction turn must carry the actual reason
        follow_up = provider.conversations[1][-1]["content"]
        self.assertIn("snake_case", follow_up)

    def test_product_mode_falls_back_visibly(self) -> None:
        res = _plan_call(ScriptedProvider(["not json", "still not json", "nope"]), ReasoningMode.PRODUCT)
        self.assertFalse(res.blocked)
        self.assertEqual(res.trace.outcome, ReasoningOutcome.FALLBACK)
        self.assertTrue(res.data["requests"])  # deterministic questionnaire
        self.assertIn("reasoning failed", res.trace.reason)

    def test_developer_mode_blocks_with_diagnostics(self) -> None:
        res = _plan_call(ScriptedProvider(["not json", "nope", "still nope"]), ReasoningMode.DEVELOPER)
        self.assertTrue(res.blocked)
        self.assertIsNone(res.data)
        self.assertEqual(res.trace.outcome, ReasoningOutcome.BLOCKED)
        self.assertEqual(len(res.trace.attempts), 3)
        self.assertTrue(all(a.failure_kind for a in res.trace.attempts))
        self.assertTrue(res.trace.attempts[0].raw_excerpt)

    def test_unconfigured_provider(self) -> None:
        res = _plan_call(ScriptedProvider([], configured=False), ReasoningMode.DEVELOPER)
        self.assertTrue(res.blocked)
        self.assertEqual(res.trace.attempts, [])

    def test_provider_without_structured_support(self) -> None:
        res = _plan_call(MockReasoningProvider(), ReasoningMode.PRODUCT)
        self.assertEqual(res.trace.outcome, ReasoningOutcome.FALLBACK)
        self.assertIn("structured", res.trace.reason)

    def test_transport_failure_does_not_retry(self) -> None:
        provider = ScriptedProvider([RuntimeError("connection reset"), GOOD_PLAN])
        res = _plan_call(provider, ReasoningMode.PRODUCT)
        self.assertEqual(len(res.trace.attempts), 1)
        self.assertEqual(res.trace.attempts[0].failure_kind, ReasoningFailureKind.TRANSPORT)


class SemanticValidatorTests(unittest.TestCase):
    def test_capacity_measurement_is_required(self) -> None:
        plan = MeasurementPlan.model_validate(json.loads(GOOD_PLAN))
        plan.requests = [r for r in plan.requests if r.affects != "capacity"]
        with self.assertRaises(Exception) as ctx:
            validate_measurement_plan(plan)
        self.assertIn("capacity", str(ctx.exception))

    def test_unit_suffix_must_match_unit(self) -> None:
        plan = MeasurementPlan.model_validate(json.loads(GOOD_PLAN))
        plan.requests[0].field = "bicycle_mass"  # kg without the _kg suffix
        with self.assertRaises(Exception):
            validate_measurement_plan(plan)

    def test_unknown_affects_is_normalised_not_rejected(self) -> None:
        plan = MeasurementPlan.model_validate(json.loads(GOOD_PLAN))
        plan.requests[1].affects = "tipping"
        validate_measurement_plan(plan)
        self.assertEqual(plan.requests[1].affects, "stability")

    def test_reasoning_cannot_widen_the_hazard_floor(self) -> None:
        floor = ScopeAssessment(in_scope=False, reasons=["human support"])
        with self.assertRaises(Exception) as ctx:
            validate_scope(ScopeAssessment(in_scope=True), floor)
        self.assertIn("cannot approve", str(ctx.exception))

    def test_reasoning_cannot_lower_the_safety_factor(self) -> None:
        floor = ScopeAssessment(in_scope=True)
        with self.assertRaises(Exception):
            validate_scope(ScopeAssessment(in_scope=True, recommended_safety_factor=1.1), floor)
        validate_scope(ScopeAssessment(in_scope=True, recommended_safety_factor=4.0), floor)


class HazardFloorTests(unittest.TestCase):
    """The floor refuses only the unmistakable; ambiguous wording is deferred, never guessed.

    A keyword must not be what refuses "a laptop stand on my desk", and a request that
    carries a person must not pass merely because it avoided the word "stool".
    """

    def test_body_weight_however_it_is_worded(self) -> None:
        for request in (
            "a step stool for the kitchen",
            "a footrest for under my desk",
            "a grab bar for the shower",
            "something that will hold my weight",
            "a bracket I can sit on",
            "something to stand on to reach a high shelf",
            "a platform to stand on while painting",
            "a step to help my kid reach the sink",          # no furniture noun at all
            "a little platform so my toddler can reach the sink",
            "Design a stool that supports a 100 kg person",
        ):
            self.assertEqual(classify_hazard(request).verdict, "refuse", request)
            self.assertFalse(det.assess_scope(request).in_scope, request)

    def test_ordinary_object_holders_are_not_refused(self) -> None:
        """Paraphrases that previously tripped the substring list."""
        for request in (
            "a laptop stand on my desk",        # "stand on" as a noun
            "a monitor stand on my desk",
            "a rack that sits on the shelf",
            "a workbench organizer",            # contains "bench"
            "a rack for my kid's books",        # person word, no weight-bearing action
            "a desk hook for a 5 kg bag",
            "a wall rack for my keys",
            "a cup holder that clamps to my desk",
            "a stand for my phone",
        ):
            self.assertEqual(classify_hazard(request).verdict, "clear", request)
            self.assertTrue(det.assess_scope(request).in_scope, request)

    def test_ambiguous_wording_is_deferred_with_a_question(self) -> None:
        """Flagged for reasoning or the user — in scope at the floor, but not silently."""
        for request, hazard in (
            ("a hook in the garage ceiling for a bike", "overhead"),
            ("a ceiling fan remote holder", "overhead"),
            ("a holder for my impact driver", "impact"),
        ):
            signal = classify_hazard(request)
            self.assertEqual(signal.verdict, "ambiguous", request)
            self.assertEqual(signal.hazard_class, hazard, request)
            self.assertTrue(signal.question, request)
            assessment = det.assess_scope(request)
            self.assertTrue(assessment.in_scope, request)
            self.assertTrue(any("unresolved" in r for r in assessment.reasons), request)

    def test_reasoning_may_still_refuse_a_deferred_request(self) -> None:
        """Deferring is safe precisely because reasoning may restrict what the floor allowed."""
        floor = det.assess_scope("a hook in the garage ceiling for a bike")
        self.assertTrue(floor.in_scope)
        validate_scope(ScopeAssessment(in_scope=False, hazard_class="overhead",
                                       reasons=["overhead mount"]), floor)

    def test_interaction_agent_rejects_hard_hazards_only(self) -> None:
        from agents.interaction import InteractionAgent
        from schemas import InteractionDecision

        agent = InteractionAgent()
        self.assertEqual(
            agent.assess("a step stool for the kitchen").decision,
            InteractionDecision.REJECT_OR_ESCALATE,
        )
        self.assertNotEqual(
            agent.assess("a laptop stand on my desk").decision,
            InteractionDecision.REJECT_OR_ESCALATE,
        )


if __name__ == "__main__":
    unittest.main()
