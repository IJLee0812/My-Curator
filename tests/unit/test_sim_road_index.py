"""Road selection: hard constraints first, relaxation recorded, choice reproducible."""

from __future__ import annotations

import pytest

from my_curator.adapters.storage.pg import SIM_ROAD_INDEX_DDL
from my_curator.domain.sim.reasons import DegradationCode
from my_curator.domain.sim.road_index import (
    ENTRY_MARGIN_M,
    RoadCandidate,
    select_road,
)
from my_curator.domain.sim.spec import RoadQuery

pytestmark = pytest.mark.unit


def candidate(**over) -> RoadCandidate:
    base = dict(
        town="Town03",
        road_id=10,
        lane_id=-1,
        lane_section_s=0.0,
        lane_section_end_s=200.0,
        driving_lanes=2,
        speed_kph=40.0,
        lane_types=frozenset({"driving", "sidewalk"}),
        junction_forms=frozenset({"signalized"}),
        in_junction=False,
    )
    base.update(over)
    return RoadCandidate(**base)


def query(**over) -> RoadQuery:
    base = dict(
        road_type="secondary",
        intersection_type="signalized",
        min_driving_lanes=1,
        speed_kph_range=(40, 60),
        required_lane_types=("driving",),
        candidate_towns=("Town03",),
    )
    base.update(over)
    return RoadQuery(**base)


class TestSelection:
    def test_returns_none_only_when_there_are_no_candidates(self):
        assert select_road(query(), [], seed="x") is None

    def test_an_exact_match_records_no_compromise(self):
        selection = select_road(query(), [candidate()], seed="x")
        assert selection.candidate.road_id == 10
        assert selection.degradations == ()

    def test_every_hard_constraint_is_honoured_when_it_can_be(self):
        wanted = candidate(road_id=99, driving_lanes=3)
        pool = [
            candidate(road_id=1, driving_lanes=1),
            candidate(road_id=2, speed_kph=100.0),
            candidate(road_id=3, junction_forms=frozenset({"unsignalized"})),
            wanted,
        ]
        selection = select_road(query(min_driving_lanes=3), pool, seed="x")
        assert selection.candidate.road_id == 99
        assert selection.degradations == ()

    def test_intersection_none_requires_a_road_outside_a_junction(self):
        pool = [candidate(road_id=1, in_junction=True), candidate(road_id=2, in_junction=False)]
        selection = select_road(query(intersection_type="none"), pool, seed="x")
        assert selection.candidate.road_id == 2
        assert selection.degradations == ()


class TestRelaxation:
    def test_speed_is_given_up_before_anything_else(self):
        only = candidate(speed_kph=100.0)
        selection = select_road(query(), [only], seed="x")
        codes = [d.code for d in selection.degradations]
        assert codes == [DegradationCode.ROAD_TYPE_SUBSTITUTED]
        assert selection.degradations[0].field_path == "topology.road.speed"

    def test_the_town_is_given_up_before_physical_properties(self):
        """A road elsewhere that fits beats a local road that does not."""
        elsewhere = candidate(town="Town05", road_id=7)
        local_but_wrong = candidate(town="Town03", road_id=8, driving_lanes=1)
        selection = select_road(query(min_driving_lanes=2), [elsewhere, local_but_wrong], seed="x")
        assert selection.candidate.town == "Town05"
        assert [d.field_path for d in selection.degradations] == ["topology.road.town"]

    def test_a_missing_junction_form_is_recorded_as_a_substitution(self):
        only = candidate(junction_forms=frozenset({"unsignalized"}))
        selection = select_road(query(intersection_type="roundabout"), [only], seed="x")
        codes = [d.code for d in selection.degradations]
        assert DegradationCode.INTERSECTION_SUBSTITUTED in codes

    def test_relaxation_stops_as_soon_as_something_matches(self):
        """Giving up speed must not also give up the junction form."""
        selection = select_road(query(), [candidate(speed_kph=100.0)], seed="x")
        assert len(selection.degradations) == 1

    def test_a_query_never_fails_while_any_candidate_exists(self):
        hopeless = candidate(
            town="Town01",
            speed_kph=100.0,
            driving_lanes=1,
            lane_types=frozenset({"driving"}),
            junction_forms=frozenset(),
            in_junction=True,
        )
        selection = select_road(
            query(
                intersection_type="roundabout",
                min_driving_lanes=4,
                required_lane_types=("driving", "parking"),
                candidate_towns=("Town05",),
            ),
            [hopeless],
            seed="x",
        )
        assert selection is not None
        assert len(selection.degradations) == 5


class TestDeterminism:
    def test_the_same_seed_always_picks_the_same_road(self):
        pool = [candidate(road_id=i) for i in range(20)]
        picks = {select_road(query(), pool, seed="seg-42").candidate.road_id for _ in range(10)}
        assert len(picks) == 1

    def test_input_order_does_not_change_the_choice(self):
        pool = [candidate(road_id=i) for i in range(20)]
        first = select_road(query(), pool, seed="seg-42").candidate
        second = select_road(query(), list(reversed(pool)), seed="seg-42").candidate
        assert first == second

    def test_different_segments_spread_across_the_candidates(self):
        """A fixed pick would stage every segment on one road."""
        pool = [candidate(road_id=i) for i in range(50)]
        picks = {select_road(query(), pool, seed=f"seg-{i}").candidate.road_id for i in range(50)}
        assert len(picks) > 1

    def test_a_long_enough_section_is_preferred(self):
        short = candidate(road_id=1, lane_section_end_s=20.0)
        long = candidate(road_id=2, lane_section_end_s=400.0)
        selection = select_road(query(), [short, long], seed="x", min_length_m=100.0)
        assert selection.candidate.road_id == 2

    def test_length_is_a_preference_not_a_requirement(self):
        short = candidate(lane_section_end_s=20.0)
        selection = select_road(query(), [short], seed="x", min_length_m=1000.0)
        assert selection is not None
        assert selection.degradations == ()


class TestEntryPoint:
    def test_entry_sits_inside_a_long_section(self):
        assert candidate(lane_section_s=100.0).entry_s == 100.0 + ENTRY_MARGIN_M

    def test_entry_stays_inside_a_short_section(self):
        c = candidate(lane_section_s=0.0, lane_section_end_s=4.0)
        assert c.entry_s == 2.0
        assert c.entry_s < c.lane_section_end_s


class TestDdlParity:
    def test_the_builder_and_init_sql_declare_the_same_table(self, tmp_path):
        """Two copies of a DDL drift; this is what stops them."""
        from pathlib import Path

        init_sql = (
            Path(__file__).resolve().parents[2] / "infra" / "init-sql" / "001_schema.sql"
        ).read_text(encoding="utf-8")

        def normalize(text: str) -> str:
            body = text[text.index("CREATE TABLE IF NOT EXISTS sim_road_index") :]
            body = body[: body.index("idx_sim_road_town")]
            return " ".join(body.split())

        assert normalize(init_sql) == normalize(SIM_ROAD_INDEX_DDL)


class TestTravelDirection:
    """Lane sign decides which way "ahead" is; getting it wrong hides every actor."""

    def test_a_right_hand_lane_runs_with_increasing_s(self):
        assert candidate(lane_id=-1).travel_direction == 1

    def test_a_left_hand_lane_runs_against_increasing_s(self):
        assert candidate(lane_id=2).travel_direction == -1

    def test_entry_leaves_room_behind_on_a_right_hand_lane(self):
        """A follower has to fit behind ego without overlapping it."""
        c = candidate(lane_id=-1, lane_section_s=0.0, lane_section_end_s=200.0)
        assert c.entry_s == pytest.approx(ENTRY_MARGIN_M)

    def test_entry_is_measured_from_the_far_end_on_a_left_hand_lane(self):
        c = candidate(lane_id=1, lane_section_s=0.0, lane_section_end_s=200.0)
        assert c.entry_s == pytest.approx(200.0 - ENTRY_MARGIN_M)

    def test_ahead_is_ahead_on_both_sides_of_the_road(self):
        right = candidate(lane_id=-1, lane_section_s=0.0, lane_section_end_s=200.0)
        left = candidate(lane_id=1, lane_section_s=0.0, lane_section_end_s=200.0)
        assert right.s_ahead(8.0) == pytest.approx(ENTRY_MARGIN_M + 8.0)
        assert left.s_ahead(8.0) == pytest.approx(200.0 - ENTRY_MARGIN_M - 8.0)
        for c in (right, left):
            assert (c.s_ahead(8.0) - c.entry_s) * c.travel_direction > 0

    def test_a_point_beyond_the_section_is_clamped_into_it(self):
        c = candidate(lane_id=-1, lane_section_s=10.0, lane_section_end_s=40.0)
        assert c.s_ahead(500.0) == pytest.approx(40.0)
        assert c.s_ahead(-500.0) == pytest.approx(10.0)

    def test_a_short_section_still_places_entities_inside_it(self):
        c = candidate(lane_id=1, lane_section_s=0.0, lane_section_end_s=4.0)
        assert c.lane_section_s <= c.entry_s <= c.lane_section_end_s
