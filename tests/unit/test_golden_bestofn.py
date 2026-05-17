"""GT-3: golden Best-of-N aggregator selection.

Locks the *selected* ScoutReport text for 5 fixture ScoutReport sets across the
refactoring stages.  Selection logic moves from
``src/scouts/aggregator.py`` to ``my_curator/domain/scout/aggregator.py``;
behavior must be byte-identical.

References:
  docs/refactoring_plan.md  §3.1 GT-3.
"""

from __future__ import annotations

import pytest

from src.scouts.aggregator import BestOfNAggregator
from src.scouts.base import ScoutReport


def _r(text: str, t: float, partial: bool = False) -> ScoutReport:
    return ScoutReport(text=text, temperature=t, seed=42, latency_ms=0.0, partial_sampling=partial)


_GOLDEN_FIXTURES = [
    # (case_name, [reports], inventory, expected_winner_text)
    (
        "clear_winner_highest_overlap",
        [
            _r("a car is visible", 0.3),
            _r("car and pedestrian interact", 0.5),
            _r("car, pedestrian, and cyclist ahead", 0.7),
        ],
        {"car": 2, "pedestrian": 1, "cyclist": 1},
        "car, pedestrian, and cyclist ahead",
    ),
    (
        "tie_broken_by_lowest_temperature",
        [
            _r("car detected", 0.3),
            _r("a car visible", 0.5),
            _r("car spotted", 0.7),
        ],
        {"car": 1},
        "car detected",
    ),
    (
        "all_zero_scores_lowest_temp_wins",
        [
            _r("car and pedestrian", 0.3),
            _r("truck on road", 0.5),
            _r("cyclist visible", 0.7),
        ],
        {"airplane": 1},
        "car and pedestrian",
    ),
    (
        "empty_inventory_lowest_temp_wins",
        [
            _r("car visible", 0.3),
            _r("pedestrian crossing", 0.5),
        ],
        {},
        "car visible",
    ),
    (
        "partial_reports_participate",
        [
            _r("car and pedestrian", 0.3, partial=True),
            _r("car only", 0.5, partial=True),
        ],
        {"car": 1, "pedestrian": 1},
        "car and pedestrian",
    ),
]


@pytest.fixture(scope="module")
def aggregator() -> BestOfNAggregator:
    return BestOfNAggregator()


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,reports,inventory,expected_text",
    _GOLDEN_FIXTURES,
    ids=[c[0] for c in _GOLDEN_FIXTURES],
)
def test_golden_bestofn_select(aggregator, name, reports, inventory, expected_text):
    chosen = aggregator.select(reports, inventory)
    assert chosen is not None, f"[{name}] expected a winner, got None"
    assert chosen.text == expected_text, (
        f"[{name}] selection drift: got {chosen.text!r}, expected {expected_text!r}"
    )


@pytest.mark.unit
def test_golden_fixture_count():
    assert len(_GOLDEN_FIXTURES) == 5
