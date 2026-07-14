"""Unit tests for the Judge N-sample majority-vote decision (P4-6)."""

from __future__ import annotations

import pytest

from my_curator.domain.judge.decision import (
    JudgeDecision,
    decide,
    safety_event_inconsistency,
)
from my_curator.domain.judge.verdict import Verdict

pytestmark = pytest.mark.unit


def _v(risk, rationale=None, scene="KEEP", confidence="high"):
    return Verdict(risk=risk, rationale=rationale, scene=scene, confidence=confidence, raw="")


def test_unanimous_keep():
    d = decide("nominal", [_v("KEEP"), _v("KEEP"), _v("KEEP")])
    assert isinstance(d, JudgeDecision)
    assert d.flipped is False
    assert d.final_risk == "nominal"
    assert d.agreement == 3 and d.n == 3


def test_majority_flip_up():
    d = decide(
        "nominal",
        [_v("critical", "R1"), _v("critical", "R1"), _v("KEEP")],
    )
    assert d.flipped is True
    assert d.final_risk == "critical"
    assert d.rationale == "R1"
    assert d.votes == {"critical": 2, "nominal": 1}


def test_majority_flip_down():
    d = decide("critical", [_v("elevated", "R3"), _v("elevated", "R3"), _v("elevated", "R3")])
    assert d.flipped is True
    assert d.final_risk == "elevated"


def test_split_vote_no_majority_keeps():
    # 1 critical / 1 elevated / 1 keep(->nominal): modal count 1 < threshold 2 -> KEEP.
    d = decide("nominal", [_v("critical"), _v("elevated"), _v("KEEP")])
    assert d.flipped is False
    assert d.final_risk == "nominal"


def test_tie_below_threshold_keeps():
    # N=2, threshold=2; one flip vote is not a majority -> KEEP (conservative).
    d = decide("elevated", [_v("critical"), _v("KEEP")])
    assert d.flipped is False
    assert d.final_risk == "elevated"


def test_keep_when_majority_equals_scout():
    d = decide("elevated", [_v("elevated"), _v("KEEP"), _v("critical")])
    # effective: elevated, elevated, critical -> modal elevated == scout -> KEEP
    assert d.flipped is False
    assert d.final_risk == "elevated"


def test_empty_verdicts_keep():
    d = decide("critical", [])
    assert d.flipped is False and d.final_risk == "critical" and d.n == 0


def test_scene_override_reported_when_majority():
    d = decide(
        "elevated",
        [_v("KEEP", scene="Rainy night."), _v("KEEP", scene="Rainy night."), _v("KEEP")],
    )
    assert d.flipped is False
    assert d.scene_override == "Rainy night."


def test_scene_override_none_when_minority():
    d = decide("elevated", [_v("KEEP", scene="Rainy."), _v("KEEP"), _v("KEEP")])
    assert d.scene_override is None


def test_confidence_is_modal_and_logged_only():
    d = decide(
        "nominal",
        [
            _v("KEEP", confidence="high"),
            _v("KEEP", confidence="medium"),
            _v("KEEP", confidence="medium"),
        ],
    )
    assert d.confidence == "medium"  # logged only; does not affect the KEEP decision
    assert d.flipped is False


def test_custom_majority_threshold():
    # Require unanimity (3/3) — a 2/3 flip is downgraded to KEEP.
    d = decide("nominal", [_v("critical"), _v("critical"), _v("KEEP")], majority=3)
    assert d.flipped is False


def test_safety_event_inconsistency_flags_indeterminate_collision():
    dna = {"planner_logic": {"safety_event": {"event_type": "collision", "collision_type": None}}}
    assert safety_event_inconsistency(dna) is not None


def test_safety_event_inconsistency_none_for_typed_collision():
    dna = {
        "planner_logic": {"safety_event": {"event_type": "collision", "collision_type": "rear_end"}}
    }
    assert safety_event_inconsistency(dna) is None


def test_safety_event_inconsistency_none_for_no_event():
    dna = {"planner_logic": {"safety_event": {"event_type": "none", "collision_type": None}}}
    assert safety_event_inconsistency(dna) is None


@pytest.mark.parametrize("bad", [None, {}, {"planner_logic": None}, {"planner_logic": {}}])
def test_safety_event_inconsistency_robust_to_missing(bad):
    assert safety_event_inconsistency(bad) is None
