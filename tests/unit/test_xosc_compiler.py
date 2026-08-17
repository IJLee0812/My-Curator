"""Compilation: every template expressible, every document schema-valid and reproducible.

The XSD check runs here, in the host venv, so the acceptance criterion is enforced on
every push rather than only when a simulator is available.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from my_curator.adapters.sim.xosc_writer import serialize, validate
from my_curator.domain.sim import catalog as cat
from my_curator.domain.sim.road_index import RoadCandidate, RoadSelection
from my_curator.domain.sim.spec import (
    ActorSpec,
    CameraSpec,
    ControlMode,
    EgoSpec,
    RoadQuery,
    SafetyEventSpec,
    SimSpec,
    WorldSpec,
)
from my_curator.domain.sim.templates import TEMPLATES, TemplateContext, build_events
from my_curator.domain.sim.xosc_compiler import EGO_NAME, compile_scenario

pytestmark = pytest.mark.unit


def road_selection(**over) -> RoadSelection:
    base = dict(
        town="Town03",
        road_id=42,
        lane_id=-1,
        lane_section_s=0.0,
        lane_section_end_s=300.0,
        driving_lanes=2,
        speed_kph=40.0,
        lane_types=frozenset({"driving", "sidewalk"}),
        junction_forms=frozenset({"signalized"}),
        in_junction=False,
    )
    base.update(over)
    return RoadSelection(candidate=RoadCandidate(**base))


def actor(index=0, **over) -> ActorSpec:
    base = dict(
        index=index,
        actor_class="vehicle_car",
        blueprint_filter="vehicle.*|base_type=car",
        state="cutin",
        maneuver_template="vehicle_lane_change_into_ego_lane",
        distance_bucket="near",
        distance_m=8.0,
        control_mode=ControlMode.EVENT,
    )
    base.update(over)
    return ActorSpec(**base)


def sim_spec(actors=(), **over) -> SimSpec:
    base = dict(
        clip_id="00000000-0000-0000-0000-000000000001",
        dna_version="0.2.0",
        duration_s=5.0,
        warmup_s=3.0,
        risk_level="critical",
        world=WorldSpec(
            weather={
                "cloudiness": 65.0,
                "precipitation": 30.0,
                "wetness": 40.0,
                "fog_density": 5.0,
                "sun_altitude_angle": 70.0,
                "sun_azimuth_angle": 250.0,
            },
            road=RoadQuery(
                road_type="secondary",
                intersection_type="signalized",
                min_driving_lanes=1,
                speed_kph_range=(40, 60),
                required_lane_types=("driving",),
                candidate_towns=("Town03",),
            ),
        ),
        ego=EgoSpec(
            maneuver="brake_hard", control_template="ego_brake_hard", target_speed_kph=45.0
        ),
        actors=tuple(actors),
        cameras=(CameraSpec(view="ego", transform=(1.6, 0.0, 1.5, 0.0, 0.0, 0.0)),),
        safety_event=SafetyEventSpec(
            has_event=True,
            event_type="near_miss",
            collision_type="none",
            severity_estimate="no_harm",
        ),
    )
    base.update(over)
    return SimSpec(**base)


class TestSchemaValidity:
    def test_a_minimal_scenario_validates(self):
        result = validate(compile_scenario(sim_spec(), road_selection()))
        assert result.is_valid, result.errors[:2]

    def test_a_scenario_with_vehicle_and_walker_validates(self):
        actors = [
            actor(0),
            actor(
                1,
                actor_class="pedestrian",
                blueprint_filter="walker.pedestrian.*",
                state="crossing",
                maneuver_template="walker_cross_at_crosswalk",
                control_mode=ControlMode.SCRIPTED,
            ),
        ]
        result = validate(compile_scenario(sim_spec(actors), road_selection()))
        assert result.is_valid, result.errors[:2]

    @pytest.mark.parametrize("template", sorted(t for t in TEMPLATES if t.startswith("ego_")))
    def test_every_ego_maneuver_compiles_to_a_valid_document(self, template):
        spec = sim_spec(ego=EgoSpec(maneuver="x", control_template=template, target_speed_kph=45.0))
        result = validate(compile_scenario(spec, road_selection()))
        assert result.is_valid, result.errors[:2]

    @pytest.mark.parametrize(
        "state,template",
        sorted((state, mapping.template) for state, mapping in cat.ACTOR_STATE.items()),
    )
    def test_every_actor_state_compiles_to_a_valid_document(self, state, template):
        spec = sim_spec([actor(0, state=state, maneuver_template=template)])
        result = validate(compile_scenario(spec, road_selection()))
        assert result.is_valid, result.errors[:2]

    def test_an_invalid_document_is_reported_rather_than_raised(self):
        broken = ET.Element("OpenSCENARIO")
        result = validate(broken, clip_id="x")
        assert not result.is_valid
        assert result.errors


class TestTemplateCoverage:
    def test_every_catalog_template_name_has_a_builder(self):
        """A template the catalog can emit but the compiler cannot express is a silent drop."""
        declared = {m.template for m in cat.ACTOR_STATE.values()}
        declared |= {m.template for m in cat.EGO_MANEUVER.values()}
        assert declared <= set(TEMPLATES), declared - set(TEMPLATES)

    def test_no_builder_is_unreachable_from_the_catalog(self):
        declared = {m.template for m in cat.ACTOR_STATE.values()}
        declared |= {m.template for m in cat.EGO_MANEUVER.values()}
        assert set(TEMPLATES) <= declared, set(TEMPLATES) - declared

    def test_every_template_produces_at_least_one_event(self):
        ctx = TemplateContext(
            entity="adversary_0",
            ego=EGO_NAME,
            target_speed_mps=12.5,
            trigger_distance_m=8.0,
            event_timed=True,
        )
        for name in TEMPLATES:
            assert build_events(name, ctx), name

    def test_an_unknown_template_yields_nothing_rather_than_raising(self):
        ctx = TemplateContext("a", EGO_NAME, 10.0, 8.0, False)
        assert build_events("no_such_template", ctx) == []


class TestEventTiming:
    def _events(self, control_mode):
        spec = sim_spec([actor(0, control_mode=control_mode)])
        return serialize(compile_scenario(spec, road_selection()))

    def test_the_event_actor_is_triggered_relative_to_ego(self):
        assert "RelativeDistanceCondition" in self._events(ControlMode.EVENT)

    def test_a_scripted_actor_is_triggered_off_the_clock(self):
        doc = self._events(ControlMode.SCRIPTED)
        assert "RelativeDistanceCondition" not in doc
        assert "SimulationTimeCondition" in doc

    def test_ambient_traffic_gets_no_maneuver(self):
        doc = self._events(ControlMode.AMBIENT)
        assert "adversary_0_group" not in doc

    def test_ambient_traffic_is_still_an_entity(self):
        doc = self._events(ControlMode.AMBIENT)
        assert 'ScenarioObject name="adversary_0"' in doc


class TestDeterminism:
    def test_recompiling_the_same_input_is_byte_identical(self):
        spec, road = sim_spec([actor(0)]), road_selection()
        assert serialize(compile_scenario(spec, road)) == serialize(compile_scenario(spec, road))

    def test_no_wall_clock_timestamp_leaks_into_the_document(self):
        from datetime import datetime

        doc = serialize(compile_scenario(sim_spec(), road_selection()))
        assert str(datetime.now().year) not in doc or "2026-01-01" in doc


class TestPlacement:
    def test_the_scenario_names_the_resolved_town(self):
        doc = serialize(compile_scenario(sim_spec(), road_selection(town="Town05")))
        assert 'LogicFile filepath="Town05"' in doc

    def test_entities_are_placed_on_the_resolved_road(self):
        doc = serialize(compile_scenario(sim_spec([actor(0)]), road_selection(road_id=77)))
        assert 'roadId="77"' in doc

    def test_a_follower_starts_behind_ego(self):
        spec = sim_spec(
            [
                actor(
                    0,
                    state="tailing",
                    maneuver_template="vehicle_follow_ego",
                    distance_m=20.0,
                    control_mode=ControlMode.AMBIENT,
                )
            ]
        )
        root = compile_scenario(spec, road_selection())
        positions = [el.get("s") for el in root.iter("LanePosition")]
        assert float(positions[1]) < float(positions[0])

    def test_a_cut_in_starts_in_a_neighbouring_lane(self):
        root = compile_scenario(sim_spec([actor(0)]), road_selection(lane_id=-1))
        lanes = [el.get("laneId") for el in root.iter("LanePosition")]
        assert lanes[0] == "-1"
        assert lanes[1] != "-1"

    def test_static_props_are_dressing_not_entities(self):
        prop = actor(
            0,
            actor_class="construction_object",
            blueprint_filter="static.prop.constructioncone",
            state="static",
            maneuver_template="actor_static",
            control_mode=ControlMode.SCRIPTED,
        )
        doc = serialize(compile_scenario(sim_spec([prop]), road_selection()))
        assert "adversary_0" not in doc
