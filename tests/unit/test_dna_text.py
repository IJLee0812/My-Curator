"""Regression tests for the P4-7 DNA→text builder (my_curator.domain.scout.dna_text).

Locks the v0.2 field set fed to the narrative-text embedding tower: keep the
structured ODD / topology / maneuver tokens, ADD scene_description +
risk_level_rationale + a meaningful safety_event.event_type, and DROP the stale
v0.1 scene_summary.  Pure logic — no torch / GPU / server required.
"""

from __future__ import annotations

import pytest

from my_curator.domain.scout.dna_text import dna_to_text

pytestmark = pytest.mark.unit

# A realistic v0.2 DNA (nominal drive-thru scene, no safety event).
_V02_NOMINAL = {
    "odd": {"weather": "clear", "lighting": "day"},
    "topology": {"road_type": "primary", "lane_event": "normal"},
    "planner_logic": {
        "risk_level": "nominal",
        "ego_maneuver": "stop",
        "safety_event": {"has_event": False, "event_type": "none"},
        "risk_level_rationale": "The ego vehicle is stationary with no immediate threats.",
    },
    "scene_description": "A drive-thru restaurant under a canopy; the ego remains stationary.",
    # A stale v0.1 field that must be ignored:
    "scene_summary": "STALE V0.1 SUMMARY SHOULD NOT APPEAR",
}


def test_includes_v02_narrative_fields():
    text = dna_to_text(_V02_NOMINAL)
    assert "clear" in text and "day" in text
    assert "primary" in text and "normal" in text
    assert "stop" in text
    assert "The ego vehicle is stationary" in text  # risk_level_rationale
    assert "drive-thru restaurant under a canopy" in text  # scene_description


def test_drops_stale_scene_summary():
    assert "STALE V0.1 SUMMARY" not in dna_to_text(_V02_NOMINAL)


def test_skips_none_event_type():
    # 'none' safety events (the vast majority of the corpus) add no signal.
    assert "none" not in dna_to_text(_V02_NOMINAL).split()


def test_includes_meaningful_event_type():
    dna = {
        "odd": {"weather": "rain"},
        "planner_logic": {
            "ego_maneuver": "brake_hard",
            "safety_event": {"has_event": True, "event_type": "near_miss"},
        },
    }
    text = dna_to_text(dna)
    assert "near_miss" in text
    assert "brake_hard" in text


def test_empty_dna_returns_placeholder():
    assert dna_to_text({}) == "driving scene"


def test_robust_to_null_nested_blocks():
    # Null (not just missing) nested blocks must not raise.
    dna = {"odd": None, "topology": None, "planner_logic": None, "scene_description": None}
    assert dna_to_text(dna) == "driving scene"
