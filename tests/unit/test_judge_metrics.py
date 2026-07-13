"""Unit tests for Judge metrics (P4-6, domain/judge/metrics.py)."""

from __future__ import annotations

import pytest

from my_curator.domain.judge.metrics import JudgeRecord, compute_metrics

pytestmark = pytest.mark.unit


def test_car_counts_only_overrides_with_gold():
    recs = [
        JudgeRecord(scout="nominal", final="critical", gt="critical"),  # correct override
        JudgeRecord(scout="elevated", final="critical", gt="elevated"),  # wrong override
        JudgeRecord(scout="nominal", final="critical", gt=None),  # override, no gold -> excluded
        JudgeRecord(scout="nominal", final="nominal", gt="nominal"),  # pass-through -> excluded
    ]
    m = compute_metrics(recs)
    assert m["overrides"] == 2
    assert m["car_matches"] == 1
    assert m["car"] == pytest.approx(0.5)


def test_for_counts_needless_flips_on_correct_scout():
    recs = [
        JudgeRecord(scout="nominal", final="nominal", gt="nominal"),  # correct, kept -> ok
        JudgeRecord(
            scout="critical", final="elevated", gt="critical"
        ),  # correct, flipped -> FOR hit
        JudgeRecord(scout="elevated", final="elevated", gt="elevated"),  # correct, kept -> ok
    ]
    m = compute_metrics(recs)
    assert m["scout_correct"] == 3
    assert m["false_overrides"] == 1
    assert m["for"] == pytest.approx(1 / 3)


def test_nominal_passthrough_rate():
    recs = [
        JudgeRecord(scout="nominal", final="nominal", gt="nominal"),
        JudgeRecord(scout="nominal", final="critical", gt="critical"),  # nominal flipped
        JudgeRecord(scout="elevated", final="elevated", gt="elevated"),  # not nominal
    ]
    m = compute_metrics(recs)
    assert m["scout_nominal"] == 2
    assert m["nominal_passthrough"] == 1
    assert m["nominal_passthrough_rate"] == pytest.approx(0.5)


def test_rates_none_when_no_denominator():
    m = compute_metrics([JudgeRecord(scout="elevated", final="elevated", gt=None)])
    assert m["car"] is None  # no overrides-with-gold
    assert m["for"] is None  # no scout-correct-with-gold
    assert m["nominal_passthrough_rate"] is None  # no nominal scout labels


def test_accepts_dict_records():
    m = compute_metrics([{"scout": "nominal", "final": "critical", "gt": "critical"}])
    assert m["n"] == 1 and m["car"] == pytest.approx(1.0)


def test_gold_15_case_shape_matches_spike():
    # Mirrors the §13 v1 15-case result: 7 should-flip (all correct), 8 should-keep (0 flipped).
    recs = (
        [JudgeRecord("nominal", "critical", "critical")] * 2
        + [JudgeRecord("elevated", "critical", "critical")] * 2
        + [JudgeRecord("critical", "elevated", "elevated")]
        + [JudgeRecord("elevated", "nominal", "nominal")] * 2
        + [JudgeRecord("nominal", "nominal", "nominal")] * 3
        + [JudgeRecord("elevated", "elevated", "elevated")] * 4
        + [JudgeRecord("critical", "critical", "critical")]
    )
    m = compute_metrics(recs)
    assert m["car"] == pytest.approx(1.0)  # 7/7
    assert m["for"] == pytest.approx(0.0)  # 0/8
