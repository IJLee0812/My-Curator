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
from my_curator.domain.sim.xosc_compiler import (
    EGO_NAME,
    KPH_TO_MPS,
    _ego_at_record_s,
    compile_scenario,
)

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

    def test_scripted_cues_fire_inside_the_recorded_window(self):
        """Simulation time 0 is unrecorded warm-up; a cue before warmup_s is never on film.

        The first renders wrote cues at a bare 2.0 s against a 3 s warm-up, so every
        scripted maneuver had already happened when the cameras started.
        """
        spec = sim_spec([actor(0, control_mode=ControlMode.SCRIPTED)], duration_s=1.733)
        document = compile_scenario(spec, road_selection())
        cues = [
            float(node.get("value"))
            for node in document.iter("SimulationTimeCondition")
            if node.get("rule") == "greaterThan"
        ]
        stop_s = spec.warmup_s + spec.duration_s
        starts = [value for value in cues if value not in (0.0, stop_s)]
        assert starts, "the scripted actor emitted no timed cue"
        for value in starts:
            assert spec.warmup_s < value < stop_s, value

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

    def test_a_follower_ends_the_warmup_behind_ego(self):
        """Behind ego *when the cameras start* — which is ahead of ego's staged spot
        whenever the warm-up is longer than the gap, because ego drives past it."""
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
        selection = road_selection()
        root = compile_scenario(spec, selection)
        positions = [float(el.get("s")) for el in root.iter("LanePosition")]
        ego_at_record = _ego_at_record_s(positions[0], selection.candidate, spec)
        forward = selection.candidate.travel_direction
        assert (positions[1] - ego_at_record) * forward == pytest.approx(-20.0)

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


class TestActorPlacement:
    """Where actors land relative to ego — the defect that rendered empty roads.

    The first version added the actor's distance to ego's ``s`` unconditionally, which is
    only forward on a right-hand lane. On a left-hand lane every actor meant to be ahead of
    ego was placed behind it, out of both cameras, and the render looked like an empty road.
    """

    @staticmethod
    def _placements(root):
        """(lane, s) per entity, from the scenario's Init."""
        out = {}
        for private in root.findall("Storyboard/Init/Actions/Private"):
            lane = private.find("PrivateAction/TeleportAction/Position/LanePosition")
            if lane is not None:
                out[private.get("entityRef")] = (int(lane.get("laneId")), float(lane.get("s")))
        return out

    def _relative(self, ego_lane, state="cutout", distance_m=8.0, **road):
        """(lane, gap at the first recorded frame) — positive means ahead of ego.

        Measured from where ego will be once the warm-up has run, which is the frame the
        DNA's distance describes.
        """
        road_arguments = {"lane_id": ego_lane, "lane_section_s": 0.0, "lane_section_end_s": 900.0}
        road_arguments.update(road)
        selection = road_selection(**road_arguments)
        spec = sim_spec(actors=[actor(0, state=state, distance_m=distance_m)])
        places = self._placements(compile_scenario(spec, selection))
        _ego_lane_id, ego_s = places[EGO_NAME]
        adversary_lane, adversary_s = places["adversary_0"]
        candidate = selection.candidate
        ego_at_record = _ego_at_record_s(ego_s, candidate, spec)
        return adversary_lane, (adversary_s - ego_at_record) * candidate.travel_direction

    @pytest.mark.parametrize("ego_lane", [-1, -2, -3, 1, 2, 3])
    def test_an_actor_meant_to_be_ahead_is_ahead_on_every_lane(self, ego_lane):
        _, ahead_m = self._relative(ego_lane)
        assert ahead_m == pytest.approx(8.0)

    @pytest.mark.parametrize("state", ["stopped", "static", "parked", "cutout", "cutin"])
    def test_the_dna_distance_holds_at_the_first_recorded_frame(self, state):
        """Whatever the state, the gap the DNA recorded is the gap when recording starts —
        staged at the bare distance instead, ego's warm-up travel overran the actor."""
        _, ahead_m = self._relative(-1, state=state)
        assert ahead_m == pytest.approx(8.0)

    @pytest.mark.parametrize("ego_lane", [-1, -2, 1, 2])
    def test_a_follower_is_behind_on_every_lane(self, ego_lane):
        _, ahead_m = self._relative(ego_lane, state="tailing")
        assert ahead_m == pytest.approx(-8.0)

    @pytest.mark.parametrize("ego_lane", [-1, -2, -3, 1, 2, 3])
    def test_no_actor_is_ever_placed_on_the_reference_line(self, ego_lane):
        """Lane 0 is the centre line, never a driving lane; it cannot be spawned on."""
        for state in ("cutin", "cutout", "oncoming", "crossing", "tailing"):
            lane, _ = self._relative(ego_lane, state=state, driving_lanes=3)
            assert lane != 0, state

    def test_oncoming_traffic_crosses_the_centre_line(self):
        lane, ahead_m = self._relative(-1, state="oncoming")
        assert lane > 0
        assert ahead_m > 0

    @pytest.mark.parametrize("state", ["stopped", "static", "parked"])
    def test_a_stopped_actor_is_staged_beyond_the_warmup_travel(self, state):
        """Ego drives the whole warm-up at its Init speed while the actor is held still,
        so the staged ``s`` sits that much further out than the DNA gap."""
        selection = road_selection(lane_id=-1, lane_section_end_s=900.0)
        spec = sim_spec(actors=[actor(0, state=state)])
        places = self._placements(compile_scenario(spec, selection))
        staged_gap = places["adversary_0"][1] - places[EGO_NAME][1]
        assert staged_gap == pytest.approx(8.0 + 3.0 * 45.0 * KPH_TO_MPS)

    def test_head_on_traffic_is_staged_to_meet_ego_mid_segment(self):
        """Staged at the bare DNA gap the two close at their combined speed and the pass
        is over within a few frames, leaving an empty road for the rest of the clip."""
        _, ahead_m = self._relative(-1, state="oncoming")
        assert ahead_m > 8.0 + 30.0

    @pytest.mark.parametrize("ego_lane", [-1, -2, 1, 2])
    def test_a_cut_out_starts_ahead_in_ego_s_own_lane(self, ego_lane):
        """Leaving ego's lane only reads on camera if the actor was in it to begin with."""
        lane, ahead_m = self._relative(ego_lane, state="cutout")
        assert lane == ego_lane
        assert ahead_m == pytest.approx(8.0)

    def test_a_cut_in_comes_from_an_adjacent_lane(self):
        lane, _ = self._relative(-2, state="cutin", driving_lanes=2)
        assert lane == -1

    def test_a_cut_in_on_a_single_lane_carriageway_stays_in_ego_s_lane(self):
        """There is no same-direction neighbour to come from. Falling back to the opposing
        lane — the first version's choice — turned the cut-in into a head-on pass that
        swept by in under a second; slower traffic ahead is the honest degradation."""
        lane, _ = self._relative(-1, state="cutin", driving_lanes=1)
        assert lane == -1

    def test_a_cut_in_takes_the_outer_lane_when_the_inner_one_is_the_centre(self):
        lane, _ = self._relative(-1, state="cutin", driving_lanes=2)
        assert lane == -2

    def test_a_crossing_vehicle_waits_in_the_neighbouring_lane(self):
        """A vehicle cannot cross a road sideways the way a walker does: it pulls into
        ego's lane from next door instead — driven at ego's speed in ego's own lane (the
        old behavior) it left the scene during the warm-up."""
        lane, _ = self._relative(-2, state="crossing", driving_lanes=2)
        assert lane == -1

    def test_a_crossing_walker_stays_in_ego_s_lane(self):
        selection = road_selection(lane_id=-2, driving_lanes=2)
        walker = actor(
            0,
            state="crossing",
            actor_class="pedestrian",
            blueprint_filter="walker.pedestrian.*",
            maneuver_template="walker_cross_midblock",
        )
        spec = sim_spec(actors=[walker])
        lane, _ = self._placements(compile_scenario(spec, selection))["adversary_0"]
        assert lane == -2

    def test_a_crossing_actor_waits_at_speed_zero(self):
        spec = sim_spec(actors=[actor(0, state="crossing")])
        document = compile_scenario(spec, road_selection())
        init = document.find(".//Init")
        for private in init.iter("Private"):
            if private.get("entityRef") != "adversary_0":
                continue
            speed = private.find(".//AbsoluteTargetSpeed")
            assert float(speed.get("value")) == 0.0

    def test_a_crossing_vehicle_pulls_into_ego_s_lane(self):
        spec = sim_spec(actors=[actor(0, state="crossing")])
        assert "_pulls_into_lane" in serialize(compile_scenario(spec, road_selection()))

    def test_two_identical_actors_never_share_a_spawn(self):
        """The spawn lift stacks a second actor placed on the same waypoint onto the roof
        of the first, and the physics scatter that follows empties the scene."""
        spec = sim_spec(actors=[actor(0, state="cutin"), actor(1, state="cutin")])
        places = self._placements(compile_scenario(spec, road_selection(lane_id=-2)))
        (lane_a, s_a) = places["adversary_0"]
        (lane_b, s_b) = places["adversary_1"]
        assert lane_a != lane_b or abs(s_a - s_b) >= 7.0

    def test_placement_stays_inside_the_lane_section(self):
        selection = road_selection(lane_id=1, lane_section_s=100.0, lane_section_end_s=140.0)
        spec = sim_spec(actors=[actor(0, state="crossing", distance_m=500.0)])
        for lane_s in self._placements(compile_scenario(spec, selection)).values():
            assert 100.0 <= lane_s[1] <= 140.0


class TestEntityState:
    def test_an_actor_carries_its_dna_state_into_the_document(self):
        spec = sim_spec(actors=[actor(0, state="jaywalking")])
        root = compile_scenario(spec, road_selection())
        states = {
            prop.get("value")
            for prop in root.iter("Property")
            if prop.get("name") == "state" and prop.get("value")
        }
        assert states == {"jaywalking"}
