"""Unit tests for BestOfNAggregator.

All tests use mock ScoutReports — no GPU or vllm required.
"""

import pytest

from src.scouts.aggregator import BestOfNAggregator
from src.scouts.base import ScoutReport


def _report(text: str, temperature: float, seed: int = 42, partial: bool = False) -> ScoutReport:
    r = ScoutReport(text=text, temperature=temperature, seed=seed, latency_ms=0.0)
    r.partial_sampling = partial
    return r


# ---------------------------------------------------------------------------
# Empty and single-report input
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyAndSingleInput:
    def test_empty_reports_returns_none(self):
        agg = BestOfNAggregator()
        assert agg.select([], {"car": 1}) is None

    def test_single_report_returned_directly(self):
        agg = BestOfNAggregator()
        r = _report("a car on the road", 0.3)
        assert agg.select([r], {"car": 1}) is r


# ---------------------------------------------------------------------------
# Normal selection (N > 1, clear winner)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalSelection:
    def test_highest_overlap_wins(self):
        agg = BestOfNAggregator()
        inventory = {"car": 2, "pedestrian": 1, "cyclist": 1}
        r03 = _report("a car is visible", 0.3)  # score 1
        r05 = _report("car and pedestrian interact", 0.5)  # score 2
        r07 = _report("car, pedestrian, and cyclist ahead", 0.7)  # score 3
        assert agg.select([r03, r05, r07], inventory) is r07

    def test_scoring_is_binary_not_count_weighted(self):
        agg = BestOfNAggregator()
        r03 = _report("car ahead", 0.3)  # score 1
        r05 = _report("no objects", 0.5)  # score 0
        assert agg.select([r03, r05], {"car": 5}) is r03
        assert agg.select([r03, r05], {"car": 1}) is r03

    def test_partial_overlap_vs_full_overlap(self):
        agg = BestOfNAggregator()
        inventory = {"car": 1, "truck": 1, "bus": 1}
        r03 = _report("car is present", 0.3)  # score 1
        r05 = _report("car and truck visible", 0.5)  # score 2
        assert agg.select([r03, r05], inventory) is r05


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTieBreaking:
    def test_tie_broken_by_lowest_temperature(self):
        agg = BestOfNAggregator()
        inventory = {"car": 1}
        r03 = _report("car detected", 0.3)  # score 1
        r05 = _report("a car visible", 0.5)  # score 1
        r07 = _report("car spotted", 0.7)  # score 1
        assert agg.select([r03, r05, r07], inventory) is r03

    def test_all_zero_scores_returns_lowest_temperature(self):
        agg = BestOfNAggregator()
        inventory = {"airplane": 1}  # never mentioned in any report
        r03 = _report("car and pedestrian", 0.3)
        r05 = _report("truck on road", 0.5)
        r07 = _report("cyclist visible", 0.7)
        assert agg.select([r03, r05, r07], inventory) is r03


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_inventory_returns_lowest_temperature(self):
        agg = BestOfNAggregator()
        r03 = _report("car visible", 0.3)
        r05 = _report("pedestrian crossing", 0.5)
        assert agg.select([r03, r05], {}) is r03

    def test_case_insensitive_matching(self):
        agg = BestOfNAggregator()
        r03 = _report("A Car is present", 0.3)  # score 1 (case-insensitive)
        r05 = _report("no objects here", 0.5)  # score 0
        assert agg.select([r03, r05], {"car": 1}) is r03

    def test_multiword_class_name_matched(self):
        agg = BestOfNAggregator()
        inventory = {"traffic light": 1}
        r03 = _report("traffic light is red", 0.3)  # score 1
        r05 = _report("intersection ahead", 0.5)  # score 0
        assert agg.select([r03, r05], inventory) is r03

    def test_partial_sampling_reports_participate_normally(self):
        agg = BestOfNAggregator()
        inventory = {"car": 1, "pedestrian": 1}
        r03 = _report("car and pedestrian", 0.3, partial=True)  # score 2
        r05 = _report("car only", 0.5, partial=True)  # score 1
        assert agg.select([r03, r05], inventory) is r03
