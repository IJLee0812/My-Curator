"""Reading a compiled scenario back: what survives the round trip, and what must not pass.

The reader is the executor's only view of a scenario, so the two properties that matter are
that it recovers everything the compiler put in, and that it refuses anything the compiler
never emits — a silent skip there would render a scenario missing the maneuver it exists
for.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest

from my_curator.adapters.sim.xosc_reader import parse_program, read_program
from my_curator.domain.sim.program import (
    LaneChangeAction,
    LaneOffsetAction,
    SpeedAction,
    UnsupportedScenarioError,
)
from my_curator.domain.sim.spec import ControlMode, EgoSpec
from my_curator.domain.sim.xosc_compiler import EGO_NAME, compile_scenario
from tests.unit.test_xosc_compiler import actor, road_selection, sim_spec

pytestmark = pytest.mark.unit


def program_for(spec=None, road=None):
    return parse_program(compile_scenario(spec or sim_spec(), road or road_selection()))


class TestRoundTrip:
    def test_the_clip_and_town_survive(self):
        program = program_for(sim_spec(clip_id="abc-123"), road_selection(town="Town05"))
        assert program.clip_id == "abc-123"
        assert program.town == "Town05"

    def test_the_ego_is_identified_by_its_role(self):
        program = program_for()
        assert program.ego.name == EGO_NAME
        assert program.ego.is_ego
        assert not program.ego.is_walker

    def test_a_walker_is_recognized_as_one(self):
        spec = sim_spec(
            actors=[
                actor(
                    0,
                    actor_class="pedestrian",
                    blueprint_filter="walker.pedestrian.0001",
                    state="crossing",
                    maneuver_template="walker_cross_at_crosswalk",
                    control_mode=ControlMode.SCRIPTED,
                )
            ]
        )
        walkers = [e for e in program_for(spec).entities if e.is_walker]
        assert [w.blueprint_filter for w in walkers] == ["walker.pedestrian.0001"]

    def test_every_entity_is_placed_on_the_road_it_was_compiled_onto(self):
        road = road_selection(road_id=77, lane_id=-2)
        program = program_for(sim_spec(actors=[actor(0)]), road)
        assert program.init
        for state in program.init:
            assert state.placement.road_id == 77

    def test_the_stop_time_is_warmup_plus_segment(self):
        program = program_for(sim_spec(duration_s=1.733, warmup_s=3.0))
        assert program.stop_time_s == pytest.approx(4.733)

    def test_a_scripted_ego_maneuver_reads_back_as_its_action(self):
        spec = sim_spec(
            ego=EgoSpec("lane_change_left", "ego_lane_change_left", target_speed_kph=45.0)
        )
        program = program_for(spec)
        actions = [a for group in program.groups for e in group.events for a in e.actions]
        assert any(isinstance(a, LaneChangeAction) and a.relative_lane == 1 for a in actions)

    def test_an_ego_offset_reads_back_with_its_sign(self):
        spec = sim_spec(
            ego=EgoSpec("lateral_offset_right", "ego_lateral_offset_right", target_speed_kph=45.0)
        )
        offsets = [
            a
            for group in program_for(spec).groups
            for e in group.events
            for a in e.actions
            if isinstance(a, LaneOffsetAction)
        ]
        assert offsets and offsets[0].offset_m < 0

    def test_an_event_actor_triggers_off_distance_not_the_clock(self):
        spec = sim_spec(actors=[actor(0, distance_m=8.0, control_mode=ControlMode.EVENT)])
        program = program_for(spec)
        triggers = [e.trigger for g in program.groups for e in g.events if g.entity != EGO_NAME]
        assert any(t.kind == "distance" and t.value == pytest.approx(8.0) for t in triggers)

    def test_a_scripted_actor_triggers_off_the_clock(self):
        spec = sim_spec(actors=[actor(0, control_mode=ControlMode.SCRIPTED)])
        triggers = [
            e.trigger for g in program_for(spec).groups for e in g.events if g.entity != EGO_NAME
        ]
        assert triggers and all(t.is_timed for t in triggers)

    def test_ambient_actors_are_entities_without_a_maneuver_group(self):
        spec = sim_spec(actors=[actor(0, control_mode=ControlMode.AMBIENT)])
        program = program_for(spec)
        assert any(not e.is_ego for e in program.entities)
        assert all(g.entity == EGO_NAME for g in program.groups)

    def test_initial_speeds_come_through_in_metres_per_second(self):
        program = program_for(sim_spec())
        ego_state = program.init_for(EGO_NAME)
        assert ego_state.speed_mps == pytest.approx(45.0 / 3.6, abs=1e-3)


class TestEnvironment:
    def test_weather_inverts_back_to_carla_fields(self):
        program = program_for()
        weather = program.environment.to_weather()
        assert weather["sun_altitude_angle"] == pytest.approx(70.0)
        assert weather["sun_azimuth_angle"] == pytest.approx(250.0)
        assert weather["precipitation"] == pytest.approx(30.0, abs=0.5)

    def test_clear_weather_yields_no_fog(self):
        spec = sim_spec()
        clear = replace(spec.world, weather={**spec.world.weather, "fog_density": 0.0})
        program = program_for(replace(spec, world=clear))
        assert program.environment.to_weather()["fog_density"] == pytest.approx(0.0, abs=0.01)


class TestRefusal:
    def test_an_unknown_private_action_is_rejected(self):
        root = compile_scenario(sim_spec(), road_selection())
        action = root.find("Storyboard/Init/Actions/Private/PrivateAction")
        for child in list(action):
            action.remove(child)
        ET.SubElement(ET.SubElement(action, "RoutingAction"), "AssignRouteAction")
        with pytest.raises(UnsupportedScenarioError, match="unsupported PrivateAction"):
            parse_program(root)

    def test_an_unknown_trigger_is_rejected(self):
        root = compile_scenario(sim_spec(), road_selection())
        condition = root.find("Storyboard/StopTrigger/ConditionGroup/Condition")
        for child in list(condition):
            condition.remove(child)
        ET.SubElement(
            ET.SubElement(condition, "ByValueCondition"), "StoryboardElementStateCondition"
        )
        with pytest.raises(UnsupportedScenarioError, match="unsupported stop trigger"):
            parse_program(root)

    def test_a_document_that_is_not_a_scenario_is_rejected(self):
        with pytest.raises(UnsupportedScenarioError, match="not <OpenSCENARIO>"):
            parse_program(ET.Element("OpenDRIVE"))

    def test_an_entity_without_a_role_is_rejected(self):
        root = compile_scenario(sim_spec(), road_selection())
        for prop in root.find("Entities/ScenarioObject").iter("Property"):
            if prop.get("name") == "type":
                prop.set("name", "something_else")
        with pytest.raises(UnsupportedScenarioError, match="carries no role"):
            parse_program(root)

    def test_an_unplaced_entity_is_rejected(self):
        root = compile_scenario(sim_spec(), road_selection())
        private = root.find("Storyboard/Init/Actions/Private")
        for action in private.findall("PrivateAction"):
            if action.find("TeleportAction") is not None:
                private.remove(action)
        with pytest.raises(UnsupportedScenarioError, match="is never placed"):
            parse_program(root)


class TestFile:
    def test_a_written_scenario_reads_back(self, tmp_path):
        from my_curator.adapters.sim.xosc_writer import write

        path = write(compile_scenario(sim_spec(), road_selection()), tmp_path / "s.xosc")
        assert read_program(path).town == "Town03"


class TestEntityState:
    """The executor needs the DNA state: a pedestrian crossing the road and one walking
    along it are the same blueprint moving in different directions."""

    def _entity(self, state: str):
        spec = sim_spec(
            actors=[
                actor(
                    0,
                    actor_class="pedestrian",
                    blueprint_filter="walker.pedestrian.0001",
                    state=state,
                    maneuver_template="walker_cross_at_crosswalk",
                    control_mode=ControlMode.SCRIPTED,
                )
            ]
        )
        return next(e for e in program_for(spec).entities if not e.is_ego)

    def test_the_state_survives_the_round_trip(self):
        assert self._entity("jaywalking").state == "jaywalking"

    @pytest.mark.parametrize("state", ["crossing", "jaywalking", "hesitating", "emerging"])
    def test_a_crossing_state_moves_across_the_lane(self, state):
        assert self._entity(state).crosses_the_road

    @pytest.mark.parametrize("state", ["cutin", "tailing", "parked", "stopped"])
    def test_every_other_state_moves_along_it(self, state):
        assert not self._entity(state).crosses_the_road

    def test_the_ego_never_crosses(self):
        assert not program_for().ego.crosses_the_road
