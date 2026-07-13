"""Prompt-regression + builder tests for the Judge prompt (P4-6)."""

from __future__ import annotations

import pytest

from my_curator.domain.judge.prompt import (
    JUDGE_PROMPT_HASHES,
    assert_judge_prompt_registered,
    build_judge_user_prompt,
    judge_prompt_hash,
    load_system_prompt,
)

pytestmark = pytest.mark.prompt_regression


def test_shipped_prompt_hash_is_registered():
    h = judge_prompt_hash()
    assert h in JUDGE_PROMPT_HASHES, (
        f"judge_qwen3.v1.md hash {h!r} not registered — bump JUDGE_PROMPT_HASHES."
    )
    assert_judge_prompt_registered(h)  # does not raise


def test_prompt_hash_stable_value():
    # Pin the exact hash so any accidental prompt edit fails the regression gate.
    assert judge_prompt_hash() == "d06aef8a3365f0b2"


def test_assert_registered_rejects_unknown_hash():
    with pytest.raises(ValueError):
        assert_judge_prompt_registered("deadbeefdeadbeef")


def test_system_prompt_contains_rubric_and_output_contract():
    text = load_system_prompt()
    assert "R1." in text and "Any real collision is ALWAYS critical" in text
    assert "VERDICT_RISK:" in text and "CONFIDENCE:" in text
    assert "SOTIF" in text


def test_build_user_prompt_includes_four_context_fields():
    dna = {
        "scene_description": "A rainy night scene.",
        "planner_logic": {
            "risk_level": "nominal",
            "risk_level_rationale": "routine",
            "safety_event": {
                "has_event": False,
                "event_type": "none",
                "collision_type": None,
                "severity_estimate": None,
            },
        },
    }
    p = build_judge_user_prompt(dna)
    assert "A rainy night scene." in p
    assert "risk_level: nominal" in p
    assert 'risk_level_rationale: "routine"' in p
    assert "event_type=none" in p
    assert p.rstrip().endswith("per your instructions.")


def test_build_user_prompt_robust_to_missing_fields():
    # Must not raise on a sparse dict.
    p = build_judge_user_prompt({})
    assert "Scout Scenario DNA:" in p
