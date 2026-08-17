"""OpenDRIVE parsing, against a synthetic network rather than CARLA's shipped files.

The fixture is hand-written so the expected output is knowable, and so the test runs
without the CARLA image present.
"""

from __future__ import annotations

import pytest

from my_curator.adapters.sim.xodr_parser import DEFAULT_SPEED_KPH, parse_town, parse_towns

pytestmark = pytest.mark.unit


XODR = """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="synthetic"/>
  <road name="declared" length="200.0" id="1" junction="-1">
    <link><successor elementType="junction" elementId="100"/></link>
    <type s="0.0" type="town"><speed max="25" unit="mph"/></type>
    <objects><object type="crosswalk" name="SimpleCrosswalk" id="1"/></objects>
    <signals><signal id="9" name="Signal_3Light_Post01" dynamic="yes"/></signals>
    <lanes>
      <laneSection s="0.0">
        <left>
          <lane id="2" type="driving" level="false"/>
          <lane id="1" type="driving" level="false"/>
        </left>
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false"/>
          <lane id="-2" type="sidewalk" level="false"/>
        </right>
      </laneSection>
      <laneSection s="120.0">
        <left><lane id="1" type="driving" level="false"/></left>
        <center><lane id="0" type="none" level="false"/></center>
        <right><lane id="-1" type="driving" level="false"/></right>
      </laneSection>
    </lanes>
  </road>
  <road name="inherits" length="50.0" id="2" junction="100">
    <lanes>
      <laneSection s="0.0">
        <right><lane id="-1" type="driving" level="false"/></right>
      </laneSection>
    </lanes>
  </road>
  <road name="orphan" length="80.0" id="3" junction="-1">
    <lanes>
      <laneSection s="0.0">
        <right><lane id="-1" type="driving" level="false"/></right>
      </laneSection>
    </lanes>
  </road>
  <junction id="100" name="junction100">
    <connection id="0" incomingRoad="1" connectingRoad="2" contactPoint="start"/>
    <connection id="1" incomingRoad="3" connectingRoad="2" contactPoint="end"/>
  </junction>
</OpenDRIVE>
"""


@pytest.fixture
def town(tmp_path):
    path = tmp_path / "Town01.xodr"
    path.write_text(XODR, encoding="utf-8")
    return path


@pytest.fixture
def rows(town):
    return parse_town(town, "Town01")


def by_road(rows, road_id, section_s=None):
    out = [r for r in rows if r.road_id == road_id]
    if section_s is not None:
        out = [r for r in out if r.lane_section_s == section_s]
    return out


class TestLaneFlattening:
    def test_one_row_per_driving_lane_per_section(self, rows):
        # road 1: section 0 has 2 left + 1 right driving, section 120 has 1 + 1;
        # road 2 and road 3 have 1 each.
        assert len(rows) == 3 + 2 + 1 + 1

    def test_non_driving_lanes_are_not_candidates(self, rows):
        assert all(r.lane_id != -2 for r in by_road(rows, 1, 0.0))

    def test_lane_types_describe_the_whole_section(self, rows):
        section = by_road(rows, 1, 0.0)[0]
        assert {"driving", "sidewalk", "none"} <= section.lane_types

    def test_driving_lanes_counts_one_direction_only(self, rows):
        """A lane change needs a neighbour going the same way, not an oncoming one."""
        left = next(r for r in by_road(rows, 1, 0.0) if r.lane_id > 0)
        right = next(r for r in by_road(rows, 1, 0.0) if r.lane_id < 0)
        assert left.driving_lanes == 2
        assert right.driving_lanes == 1

    def test_section_bounds_come_from_the_next_section_then_the_road_length(self, rows):
        first = by_road(rows, 1, 0.0)[0]
        second = by_road(rows, 1, 120.0)[0]
        assert (first.lane_section_s, first.lane_section_end_s) == (0.0, 120.0)
        assert (second.lane_section_s, second.lane_section_end_s) == (120.0, 200.0)


class TestSpeed:
    def test_mph_is_converted_and_quantized(self, rows):
        """25 mph is 40.23 km/h; the catalog was measured in steps of ten."""
        assert by_road(rows, 1)[0].speed_kph == 40.0

    def test_a_connecting_road_inherits_from_the_junction_it_serves(self, rows):
        assert by_road(rows, 2)[0].speed_kph == 40.0

    def test_an_unlinked_road_falls_back_rather_than_dropping_out(self, rows):
        assert by_road(rows, 3)[0].speed_kph == 40.0

    def test_a_network_with_no_declared_speed_uses_the_engine_default(self, tmp_path):
        path = tmp_path / "Town02.xodr"
        path.write_text(XODR.replace('<speed max="25" unit="mph"/>', ""), encoding="utf-8")
        assert all(r.speed_kph == DEFAULT_SPEED_KPH for r in parse_town(path, "Town02"))


class TestJunctionForms:
    def test_a_signal_on_an_arm_makes_the_junction_signalized(self, rows):
        assert "signalized" in by_road(rows, 1)[0].junction_forms

    def test_a_two_arm_junction_is_a_merge_not_a_crossing(self, rows):
        assert "direct_connection" in by_road(rows, 1)[0].junction_forms
        assert "t_junction" not in by_road(rows, 1)[0].junction_forms

    def test_a_crosswalk_object_is_carried_on_its_own_road(self, rows):
        assert "crosswalk" in by_road(rows, 1)[0].junction_forms
        assert "crosswalk" not in by_road(rows, 3)[0].junction_forms

    def test_roads_inside_a_junction_are_marked(self, rows):
        assert by_road(rows, 2)[0].in_junction
        assert not by_road(rows, 1)[0].in_junction

    def test_forms_reach_roads_that_only_link_to_the_junction(self, rows):
        """Road 3 is an arm; it is outside the junction but leads into it."""
        assert "signalized" in by_road(rows, 3)[0].junction_forms


class TestTownSet:
    def test_a_missing_town_is_skipped_rather_than_fatal(self, tmp_path, caplog):
        (tmp_path / "Town01.xodr").write_text(XODR, encoding="utf-8")
        rows = parse_towns(tmp_path, ("Town01", "Town99"))
        assert {r.town for r in rows} == {"Town01"}

    def test_parsing_is_reproducible(self, town):
        assert parse_town(town, "Town01") == parse_town(town, "Town01")
