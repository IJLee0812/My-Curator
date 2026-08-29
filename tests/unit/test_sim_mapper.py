"""Mapper behavior: staging choices, recorded compromises, and exclusions."""

from __future__ import annotations

import copy
import itertools

import pytest

from my_curator.domain.sim import build_coverage_report, map_dna
from my_curator.domain.sim import catalog as cat
from my_curator.domain.sim.coverage import classify_scene_content
from my_curator.domain.sim.reasons import DegradationCode, ExclusionReason
from my_curator.domain.sim.spec import (
    DEFAULT_SEGMENT_S,
    MIN_SEGMENT_S,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    WARMUP_S,
    ControlMode,
)


def make_dna(**overrides):
    """A minimal, valid v0.2 document; ``overrides`` deep-merge into it."""
    dna = {
        "dna_version": "0.2.0",
        "clip_id": "11111111-1111-1111-1111-111111111111",
        "odd": {"weather": "clear", "lighting": "day", "sensor_fidelity": ["clean"]},
        "topology": {
            "road_type": "primary",
            "lane_event": "normal",
            "intersection_type": "none",
        },
        "actor_dynamics": [],
        "planner_logic": {
            "ego_maneuver": "cruise",
            "risk_level": "nominal",
            "risk_level_rationale": "clear road",
            "safety_event": {
                "has_event": False,
                "event_type": "none",
                "collision_type": None,
                "severity_estimate": None,
            },
        },
    }
    for block, value in overrides.items():
        if isinstance(value, dict) and isinstance(dna.get(block), dict):
            dna[block] = {**dna[block], **value}
        else:
            dna[block] = value
    return dna


def actor(actor_class="vehicle_car", state="cutin", distance_bucket="near"):
    return {
        "actor_class": actor_class,
        "state": state,
        "distance_bucket": distance_bucket,
        "confidence": 0.9,
        "grounded_by_yolo26": True,
    }


def degradation_codes(result):
    return {d.code for d in result.degradations}


@pytest.mark.unit
class TestNominalMapping:
    def test_clean_segment_maps_without_compromise(self):
        result = map_dna(make_dna())
        assert result.mapped
        assert result.exclusions == ()
        assert result.spec.degradations == ()
        assert not result.spec.is_degraded

    def test_render_contract_is_two_fixed_views(self):
        spec = map_dna(make_dna()).spec
        assert [c.view for c in spec.cameras] == ["ego", "chase"]
        for camera in spec.cameras:
            assert (camera.image_size_x, camera.image_size_y) == (RENDER_WIDTH, RENDER_HEIGHT)
            assert camera.fps == RENDER_FPS
            assert camera.fov == 90.0

    def test_segment_length_is_recorded_separately_from_warmup(self):
        spec = map_dna(make_dna()).spec
        assert spec.duration_s == DEFAULT_SEGMENT_S
        assert spec.warmup_s == WARMUP_S
        assert spec.warmup_s > 0

    def test_ego_target_speed_derives_from_road_and_maneuver(self):
        cruising = map_dna(make_dna()).spec
        stopping = map_dna(make_dna(planner_logic={"ego_maneuver": "stop"})).spec
        assert cruising.ego.target_speed_kph > stopping.ego.target_speed_kph
        assert stopping.ego.control_template == "ego_stop"

    def test_mapping_is_deterministic(self):
        dna = make_dna(actor_dynamics=[actor(), actor("pedestrian", "crossing", "mid")])
        first = map_dna(copy.deepcopy(dna))
        second = map_dna(copy.deepcopy(dna))
        assert first.spec.to_dict() == second.spec.to_dict()


@pytest.mark.unit
class TestExclusions:
    def test_blank_required_enum_is_dna_incomplete(self):
        """The VLM-degeneration quarantine leaves required enums present but empty."""
        result = map_dna(make_dna(odd={"weather": "", "lighting": ""}))
        assert not result.mapped
        reasons = {r for r, _ in result.exclusions}
        assert reasons == {ExclusionReason.DNA_INCOMPLETE}
        assert "odd.weather" in result.exclusions[0][1]

    def test_missing_block_is_dna_incomplete(self):
        dna = make_dna()
        del dna["topology"]
        assert not map_dna(dna).mapped

    def test_value_outside_the_schema_enum_is_a_distinct_reason(self):
        result = map_dna(make_dna(odd={"weather": "hurricane"}))
        assert {r for r, _ in result.exclusions} == {ExclusionReason.UNKNOWN_ENUM_VALUE}

    @pytest.mark.parametrize("road_type", ["walkway", "cycling"])
    def test_non_drivable_road_classes_are_excluded(self, road_type):
        result = map_dna(make_dna(topology={"road_type": road_type}))
        assert not result.mapped
        assert {r for r, _ in result.exclusions} == {ExclusionReason.UNSUPPORTED_ROAD_TYPE}

    def test_exclusion_detail_names_the_offending_field(self):
        result = map_dna(make_dna(topology={"road_type": "walkway"}))
        assert result.exclusions[0][1] == "topology.road_type=walkway"


@pytest.mark.unit
class TestDegradations:
    def test_snow_is_staged_but_flagged_because_carla_has_no_snow(self):
        result = map_dna(make_dna(odd={"weather": "heavy_snow"}))
        assert result.mapped
        assert DegradationCode.WEATHER_NO_SNOW in degradation_codes(result)

    def test_droplets_on_lens_records_the_missing_camera_channel(self):
        result = map_dna(make_dna(odd={"sensor_fidelity": ["droplets_on_lens"]}))
        assert DegradationCode.SENSOR_EFFECT_UNAVAILABLE in degradation_codes(result)
        # the stand-in attributes are still applied to both views
        for camera in result.spec.cameras:
            assert "chromatic_aberration_intensity" in camera.attributes

    def test_rural_substitution_names_the_unavailable_town(self):
        result = map_dna(make_dna(topology={"road_type": "rural"}))
        assert DegradationCode.ROAD_TYPE_SUBSTITUTED in degradation_codes(result)
        applied = next(
            d.applied for d in result.degradations if d.field_path == "topology.road_type"
        )
        assert "Town07" in applied

    def test_construction_divert_is_dressed_with_props(self):
        result = map_dna(make_dna(topology={"lane_event": "construction_divert"}))
        assert DegradationCode.LANE_EVENT_PROP_STAGED in degradation_codes(result)
        assert result.spec.world.props
        assert any("cone" in p.blueprint for p in result.spec.world.props)

    def test_every_degradation_states_what_replaced_what(self):
        result = map_dna(
            make_dna(
                odd={"weather": "sleet", "sensor_fidelity": ["droplets_on_lens"]},
                topology={"road_type": "service", "lane_event": "lane_closed"},
                actor_dynamics=[actor("standup_scooter_rider", "cutin")],
            )
        )
        assert result.spec.degradations
        for entry in result.spec.degradations:
            assert entry.requested and entry.applied and entry.field_path
            assert entry.note


@pytest.mark.unit
class TestActors:
    def test_actor_with_no_blueprint_is_dropped_not_excluded(self):
        result = map_dna(make_dna(actor_dynamics=[actor("animal", "crossing")]))
        assert result.mapped, "an unrepresentable actor must not sink the whole segment"
        assert result.spec.actors == ()
        assert DegradationCode.ACTOR_DROPPED in degradation_codes(result)

    def test_substituted_actor_is_still_spawned(self):
        result = map_dna(make_dna(actor_dynamics=[actor("standup_scooter_rider")]))
        assert len(result.spec.actors) == 1
        assert result.spec.actors[0].blueprint_filter == "vehicle.vespa.zx125"
        assert DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED in degradation_codes(result)

    def test_distance_bucket_becomes_a_spawn_distance(self):
        result = map_dna(make_dna(actor_dynamics=[actor(distance_bucket="far")]))
        assert result.spec.actors[0].distance_m == cat.DISTANCE_M["far"]

    def test_no_event_means_no_event_actor(self):
        result = map_dna(make_dna(actor_dynamics=[actor(), actor("pedestrian", "crossing")]))
        assert all(not a.is_event_actor for a in result.spec.actors)
        assert result.spec.event_actor is None

    def test_stated_behavior_survives_without_a_safety_event(self):
        """Most corpus segments carry actors but no ``safety_event``.

        Demoting those to ambient control would discard the ``state`` the DNA asserts —
        a cut-in would never happen and a parked car would drive away.
        """
        result = map_dna(
            make_dna(
                actor_dynamics=[actor("vehicle_car", "cutin"), actor("vehicle_car", "parked")],
                planner_logic={"risk_level": "elevated"},
            )
        )
        assert all(a.control_mode is ControlMode.SCRIPTED for a in result.spec.actors)

    @pytest.mark.parametrize("state", ["oncoming", "tailing"])
    def test_only_free_flowing_states_are_ambient(self, state):
        result = map_dna(make_dna(actor_dynamics=[actor("vehicle_car", state)]))
        assert result.spec.actors[0].control_mode is ControlMode.AMBIENT

    def test_event_promotion_does_not_demote_the_others(self):
        result = map_dna(
            make_dna(
                actor_dynamics=[actor("pedestrian", "jaywalking"), actor("vehicle_car", "parked")],
                planner_logic={
                    "safety_event": {
                        "has_event": True,
                        "event_type": "near_miss",
                        "collision_type": None,
                        "severity_estimate": "minor",
                    }
                },
            )
        )
        modes = [a.control_mode for a in result.spec.actors]
        assert modes == [ControlMode.EVENT, ControlMode.SCRIPTED]

    def test_exactly_one_event_actor_is_inferred(self):
        result = map_dna(
            make_dna(
                actor_dynamics=[
                    actor("vehicle_car", "parked", "far"),
                    actor("pedestrian", "jaywalking", "near"),
                    actor("vehicle_car", "oncoming", "mid"),
                ],
                planner_logic={
                    "safety_event": {
                        "has_event": True,
                        "event_type": "near_miss",
                        "collision_type": None,
                        "severity_estimate": "minor",
                    }
                },
            )
        )
        flagged = [a for a in result.spec.actors if a.is_event_actor]
        assert len(flagged) == 1
        assert flagged[0].state == "jaywalking"
        assert DegradationCode.EVENT_ACTOR_INFERRED in degradation_codes(result)

    def test_declared_causal_index_wins_over_inference(self):
        result = map_dna(
            make_dna(
                actor_dynamics=[
                    actor("pedestrian", "jaywalking", "near"),
                    actor("vehicle_car", "parked", "far"),
                ],
                planner_logic={
                    "causal_trigger_actor_index": 1,
                    "safety_event": {
                        "has_event": True,
                        "event_type": "near_miss",
                        "collision_type": None,
                        "severity_estimate": "minor",
                    },
                },
            )
        )
        assert result.spec.event_actor.index == 1
        assert DegradationCode.EVENT_ACTOR_INFERRED not in degradation_codes(result)

    def test_collision_is_staged_only_up_to_impact(self):
        result = map_dna(
            make_dna(
                actor_dynamics=[actor()],
                planner_logic={
                    "safety_event": {
                        "has_event": True,
                        "event_type": "collision",
                        "collision_type": "rear_end",
                        "severity_estimate": "major",
                    }
                },
            )
        )
        assert DegradationCode.COLLISION_NOT_STAGED in degradation_codes(result)


@pytest.mark.unit
class TestSchemaAnomalies:
    def test_out_of_schema_collision_type_is_reported_not_fatal(self):
        """The deployed corpus really does contain ``frontal``, which v0.2 never allowed."""
        result = map_dna(
            make_dna(
                planner_logic={
                    "safety_event": {
                        "has_event": True,
                        "event_type": "collision",
                        "collision_type": "frontal",
                        "severity_estimate": "major",
                    }
                }
            )
        )
        assert result.mapped
        assert any("frontal" in a for a in result.anomalies)

    def test_out_of_schema_actor_class_drops_only_that_actor(self):
        result = map_dna(make_dna(actor_dynamics=[actor("unicorn"), actor("vehicle_car")]))
        assert result.mapped
        assert len(result.spec.actors) == 1
        assert any("unicorn" in a for a in result.anomalies)


@pytest.mark.unit
class TestTownResolution:
    @pytest.mark.parametrize(
        ("road_type", "intersection_type"),
        list(itertools.product(cat.ROAD_TYPE, cat.INTERSECTION_TYPE)),
    )
    def test_every_combination_resolves_or_excludes_explicitly(self, road_type, intersection_type):
        """No combination may silently produce an empty town set."""
        result = map_dna(
            make_dna(topology={"road_type": road_type, "intersection_type": intersection_type})
        )
        if not result.mapped:
            assert result.exclusions
            return
        towns = result.spec.world.road.candidate_towns
        assert towns, f"{road_type}+{intersection_type} resolved to no town"
        assert set(towns) <= set(cat.LOADABLE_TOWNS)

    def test_residential_unsignalized_needs_a_town_with_unsignalized_junctions(self):
        """Town01/Town02 signalize every junction they own — the regression this guards."""
        result = map_dna(
            make_dna(topology={"road_type": "residential", "intersection_type": "unsignalized"})
        )
        assert result.spec.world.road.candidate_towns == ("Town05",)
        assert DegradationCode.INTERSECTION_SUBSTITUTED not in degradation_codes(result)

    def test_relaxing_the_junction_is_recorded(self):
        """A combination with no shared town keeps the road and flags the junction."""
        result = map_dna(
            make_dna(topology={"road_type": "motorway", "intersection_type": "roundabout"})
        )
        assert result.mapped
        assert result.spec.world.road.candidate_towns == cat.ROAD_TYPE["motorway"].towns
        assert DegradationCode.INTERSECTION_SUBSTITUTED in degradation_codes(result)


@pytest.mark.unit
class TestCoverageReport:
    def test_counts_and_percentages_add_up(self):
        results = [
            map_dna(make_dna()),
            map_dna(make_dna(planner_logic={"risk_level": "critical"})),
            map_dna(make_dna(odd={"weather": ""})),
            map_dna(make_dna(topology={"road_type": "walkway"})),
        ]
        report = build_coverage_report(results)
        assert report.total == 4
        assert report.mapped == 2
        assert report.excluded == 2
        assert report.coverage_pct == 50.0
        assert sum(b.total for b in report.by_risk) == 4

    def test_risk_breakdown_leads_with_critical(self):
        results = [
            map_dna(make_dna(planner_logic={"risk_level": "nominal"})),
            map_dna(make_dna(planner_logic={"risk_level": "critical"})),
        ]
        report = build_coverage_report(results)
        assert report.by_risk[0].risk_level == "critical"

    def test_degradation_counted_once_per_segment(self):
        """Two compromises of the same kind in one segment must not double-count."""
        result = map_dna(
            make_dna(actor_dynamics=[actor("standup_scooter_rider"), actor("e_bike_rider")])
        )
        report = build_coverage_report([result])
        assert report.degradation_counts[DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED.value] == 1
        assert report.degraded == 1

    def test_excluded_segments_group_under_unknown_risk(self):
        report = build_coverage_report([map_dna(make_dna(odd={"weather": ""}))])
        assert report.by_risk[0].risk_level == "unknown"
        assert report.coverage_pct == 0.0

    def test_report_renders_without_a_lookup_table(self):
        report = build_coverage_report(
            [map_dna(make_dna(odd={"weather": "snow"})), map_dna(make_dna(odd={"weather": ""}))]
        )
        text = report.render_text()
        assert "mappable" in text
        assert DegradationCode.WEATHER_NO_SNOW.value in text
        assert ExclusionReason.DNA_INCOMPLETE.value in text

    def test_scene_content_separates_coverage_from_demo_value(self):
        """A segment can map perfectly and still render as an empty road."""
        empty = map_dna(make_dna())
        ambient = map_dna(make_dna(actor_dynamics=[actor("vehicle_car", "oncoming")]))
        scripted = map_dna(make_dna(actor_dynamics=[actor("vehicle_car", "cutin")]))
        interaction = map_dna(
            make_dna(
                actor_dynamics=[actor("pedestrian", "jaywalking")],
                planner_logic={
                    "safety_event": {
                        "has_event": True,
                        "event_type": "near_miss",
                        "collision_type": None,
                        "severity_estimate": "minor",
                    }
                },
            )
        )
        assert classify_scene_content(empty.spec) == "ego_only"
        assert classify_scene_content(ambient.spec) == "ambient_only"
        assert classify_scene_content(scripted.spec) == "scripted_actors"
        assert classify_scene_content(interaction.spec) == "ego_interaction"

        report = build_coverage_report([empty, ambient, scripted, interaction])
        assert sum(report.scene_content_counts.values()) == report.mapped
        assert "ego_interaction" in report.render_text()

    def test_json_shape_is_serializable(self):
        import json

        report = build_coverage_report([map_dna(make_dna())])
        assert json.loads(json.dumps(report.to_dict()))["total_segments"] == 1


class TestSegmentDuration:
    """The reconstruction has to be as long as the segment it reconstructs, not a constant."""

    def test_the_source_range_sets_the_duration(self):
        dna = make_dna(timestamp_range={"start_s": 8.13, "end_s": 9.87})
        assert map_dna(dna).spec.duration_s == pytest.approx(1.74)

    def test_a_full_length_segment_keeps_the_nominal_duration(self):
        dna = make_dna(timestamp_range={"start_s": 4.13, "end_s": 9.13})
        assert map_dna(dna).spec.duration_s == pytest.approx(DEFAULT_SEGMENT_S)

    def test_dna_without_a_range_falls_back_to_the_nominal_duration(self):
        assert map_dna(make_dna()).spec.duration_s == DEFAULT_SEGMENT_S

    def test_an_unusable_range_falls_back_rather_than_producing_no_video(self):
        for broken in ({"start_s": 5.0, "end_s": 5.0}, {"start_s": "x", "end_s": 1.0}, {}, None):
            dna = make_dna(timestamp_range=broken)
            assert map_dna(dna).spec.duration_s == DEFAULT_SEGMENT_S

    def test_a_degenerate_range_is_floored_so_the_video_is_watchable(self):
        dna = make_dna(timestamp_range={"start_s": 0.0, "end_s": 0.2})
        assert map_dna(dna).spec.duration_s == MIN_SEGMENT_S

    def test_the_warmup_is_unaffected_by_the_segment_length(self):
        dna = make_dna(timestamp_range={"start_s": 8.13, "end_s": 9.87})
        assert map_dna(dna).spec.warmup_s == WARMUP_S
