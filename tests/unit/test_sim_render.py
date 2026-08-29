"""Choosing what to render, ordering the batch, and accounting for what happened."""

from __future__ import annotations

import pytest

from my_curator.domain.sim.render import (
    FAILED,
    RENDERED,
    RenderOutcome,
    SegmentRef,
    build_render_report,
    group_by_town,
    select_segment,
)

pytestmark = pytest.mark.unit


def segment(index: int, risk: str = "nominal", confidence: float = 0.5, **over) -> SegmentRef:
    base = dict(
        clip_id=f"clip-{index}",
        source_clip_id="source-a",
        segment_index=index,
        risk_level=risk,
        confidence=confidence,
        blob_uri="file://a/b.mp4",
        start_s=float(index) * 4.0,
        end_s=float(index) * 4.0 + 5.0,
    )
    base.update(over)
    return SegmentRef(**base)


class TestSelection:
    def test_the_highest_risk_segment_wins(self):
        chosen = select_segment([segment(0), segment(1, "critical"), segment(2, "elevated")])
        assert chosen.segment_index == 1

    def test_confidence_breaks_a_risk_tie(self):
        chosen = select_segment(
            [segment(0, "elevated", 0.4), segment(1, "elevated", 0.9), segment(2, "nominal", 1.0)]
        )
        assert chosen.segment_index == 1

    def test_position_breaks_a_full_tie(self):
        chosen = select_segment([segment(2, "elevated", 0.5), segment(0, "elevated", 0.5)])
        assert chosen.segment_index == 0

    def test_an_unknown_risk_never_beats_a_known_one(self):
        chosen = select_segment([segment(0, "unknown", 1.0), segment(1, "nominal", 0.1)])
        assert chosen.segment_index == 1

    def test_an_explicit_index_overrides_the_rule(self):
        chosen = select_segment([segment(0), segment(1, "critical")], index=0)
        assert chosen.segment_index == 0

    def test_an_index_that_does_not_exist_selects_nothing(self):
        assert select_segment([segment(0), segment(1)], index=7) is None

    def test_no_segments_selects_nothing(self):
        assert select_segment([]) is None


class TestBatchOrder:
    def test_segments_are_grouped_so_each_town_boots_once(self):
        plans = [
            (segment(0), "Town03"),
            (segment(1), "Town05"),
            (segment(2), "Town03"),
        ]
        grouped = group_by_town(plans)
        assert [town for town, _ in grouped] == ["Town03", "Town05"]
        assert len(grouped[0][1]) == 2

    def test_the_largest_town_goes_first(self):
        plans = [(segment(i), "Town01") for i in range(3)] + [(segment(9), "Town02")]
        assert [town for town, _ in group_by_town(plans)][0] == "Town01"

    def test_towns_of_equal_size_are_ordered_by_name(self):
        plans = [(segment(0), "Town05"), (segment(1), "Town01")]
        assert [town for town, _ in group_by_town(plans)] == ["Town01", "Town05"]


def outcome(status=RENDERED, risk="nominal", town="Town03", reason=None) -> RenderOutcome:
    return RenderOutcome(
        clip_id="c",
        source_clip_id="s",
        segment_index=0,
        risk_level=risk,
        status=status,
        town=town,
        failure_reason=reason,
    )


class TestReport:
    def test_a_clean_batch_reports_full_coverage(self):
        report = build_render_report([outcome(), outcome()])
        assert report.attempted == 2
        assert report.rendered == 2
        assert report.rendered_pct == 100.0

    def test_failures_are_counted_by_reason(self):
        report = build_render_report(
            [
                outcome(),
                outcome(FAILED, reason="spawn_rejected"),
                outcome(FAILED, reason="spawn_rejected"),
                outcome(FAILED, reason="encoding_failed"),
            ]
        )
        assert report.rendered == 1
        assert report.failures[0] == {"reason": "spawn_rejected", "count": 2}

    def test_risk_levels_are_ordered_most_severe_first(self):
        report = build_render_report(
            [outcome(risk="nominal"), outcome(risk="critical"), outcome(risk="elevated")]
        )
        assert [row["risk_level"] for row in report.by_risk] == [
            "critical",
            "elevated",
            "nominal",
        ]

    def test_an_empty_batch_does_not_divide_by_zero(self):
        report = build_render_report([])
        assert report.rendered_pct == 0.0
        assert report.to_dict()["attempted"] == 0

    def test_the_text_report_states_both_counts(self):
        text = build_render_report(
            [outcome(), outcome(FAILED, reason="spawn_rejected")]
        ).render_text()
        assert "attempted : 2" in text
        assert "spawn_rejected" in text
