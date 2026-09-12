"""Reasoning harness: validation, retry-with-feedback, and the product/developer split.

No network: a scripted provider returns canned responses so the contract is testable
offline and provider-agnostically.
"""

from __future__ import annotations

import json
import unittest

import reasoning_deterministic as det
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
    def test_body_weight_without_the_obvious_words(self) -> None:
        for request in ("a step to help my kid reach the sink",
                        "something to stand on to reach a high shelf",
                        "a footrest for under my desk"):
            self.assertFalse(det.assess_scope(request).in_scope, request)

    def test_overhead_mounts_are_refused(self) -> None:
        self.assertFalse(det.assess_scope("a hook in the garage ceiling for a bike").in_scope)

    def test_ordinary_requests_pass(self) -> None:
        for request in ("a desk hook for a 5 kg bag", "a wall rack for my keys"):
            self.assertTrue(det.assess_scope(request).in_scope, request)


if __name__ == "__main__":
    unittest.main()
