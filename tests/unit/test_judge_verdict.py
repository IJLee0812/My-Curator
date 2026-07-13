"""Unit tests for the Judge verdict parser (P4-6, domain/judge/verdict.py)."""

from __future__ import annotations

import pytest

from my_curator.domain.judge.verdict import (
    Verdict,
    effective_risk,
    parse_verdict,
    strip_thinking,
)

pytestmark = pytest.mark.unit


def test_strip_thinking_removes_block():
    assert (
        strip_thinking("<think>reasoning here</think>\nVERDICT_RISK: KEEP") == "VERDICT_RISK: KEEP"
    )


def test_strip_thinking_multiline_and_case_insensitive():
    txt = "<THINK>\nline1\nline2\n</THINK>\nafter"
    assert strip_thinking(txt) == "after"


def test_parse_keep_all_fields():
    txt = (
        "<think>looks fine</think>\n"
        "VERDICT_RISK: KEEP\n"
        "RATIONALE: -\n"
        "VERDICT_SCENE: KEEP\n"
        "CONFIDENCE: high"
    )
    v = parse_verdict(txt)
    assert v.risk == "KEEP"
    assert v.rationale is None
    assert v.scene == "KEEP"
    assert v.confidence == "high"


def test_parse_flip_with_rationale():
    txt = (
        "VERDICT_RISK: critical\n"
        "RATIONALE: R1: any collision = critical\n"
        "VERDICT_SCENE: KEEP\n"
        "CONFIDENCE: high"
    )
    v = parse_verdict(txt)
    assert v.risk == "critical"
    assert v.rationale == "R1: any collision = critical"
    assert v.scene == "KEEP"


def test_empty_rationale_does_not_bleed_into_next_line():
    # Line-anchored parsing: an omitted RATIONALE value must not capture VERDICT_SCENE.
    txt = "VERDICT_RISK: KEEP\nRATIONALE:\nVERDICT_SCENE: KEEP\nCONFIDENCE: medium"
    v = parse_verdict(txt)
    assert v.rationale is None
    assert v.scene == "KEEP"
    assert v.confidence == "medium"


def test_parse_scene_correction_strips_quotes():
    txt = 'VERDICT_RISK: KEEP\nVERDICT_SCENE: "Heavy rain, wipers active."\nCONFIDENCE: low'
    v = parse_verdict(txt)
    assert v.scene == "Heavy rain, wipers active."


def test_unparseable_risk_is_none():
    v = parse_verdict("VERDICT_RISK: maybe-critical\nCONFIDENCE: high")
    assert v.risk is None


def test_missing_block_yields_all_none():
    v = parse_verdict("<think>no verdict emitted</think>")
    assert v == Verdict(risk=None, rationale=None, scene=None, confidence=None, raw="")


def test_invalid_confidence_dropped():
    assert parse_verdict("VERDICT_RISK: KEEP\nCONFIDENCE: certain").confidence is None


@pytest.mark.parametrize(
    "risk,expected",
    [("KEEP", "elevated"), (None, "elevated"), ("critical", "critical"), ("nominal", "nominal")],
)
def test_effective_risk(risk, expected):
    v = Verdict(risk=risk, rationale=None, scene=None, confidence=None, raw="")
    assert effective_risk(v, "elevated") == expected


def test_bold_markdown_prefix_tolerated():
    v = parse_verdict("**VERDICT_RISK:** critical\nCONFIDENCE: high")
    assert v.risk == "critical"
