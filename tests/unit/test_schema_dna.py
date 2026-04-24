"""Schema validation tests for Scenario DNA v0.1 (JSON Schema draft/2020-12).

Markers: schema + unit — runs on every PR push (no Docker, no GPU required).
"""

from __future__ import annotations

import json
import pathlib
import uuid

import pytest

SCHEMA_PATH = pathlib.Path(__file__).parents[2] / "schemas" / "scenario_dna_v0_1.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def validator(schema):
    import jsonschema

    return jsonschema.Draft202012Validator(schema)


def _valid_doc() -> dict:
    """Minimal fully-valid Scenario DNA v0.1 document."""
    return {
        "dna_version": "0.1.0",
        "clip_id": str(uuid.uuid4()),
        "timestamp_range": {"start_s": 0.0, "end_s": 10.0},
        "odd": {
            "weather": "clear",
            "lighting": "day",
            "sensor_fidelity": ["clean"],
        },
        "topology": {
            "road_type": "primary",
            "lane_event": "normal",
            "intersection_type": "none",
        },
        "actor_dynamics": [],
        "planner_logic": {
            "ego_maneuver": "cruise",
            "risk_level": "nominal",
            "causal_trigger_actor_index": None,
        },
        "confidence": {
            "overall": 0.9,
            "scout_agreement": 1.0,
            "hallucination_flags": [],
        },
        "provenance": {
            "scout_models": ["cosmos-reason2-8b-fp8"],
            "scout_prompt_hash": "abc123",
            "pipeline_version": "0.1.0",
            "is_synthetic": False,
            "reference_standards": ["PEGASUS 6-Layer Model (arXiv:2012.06319)"],
        },
    }


def _actor(actor_class: str) -> dict:
    return {
        "actor_class": actor_class,
        "state": "static",
        "distance_bucket": "far",
        "confidence": 0.5,
        "grounded_by_yolo26": False,
    }


def _fails(validator, doc: dict) -> bool:
    return bool(list(validator.iter_errors(doc)))


# ── Schema file integrity ─────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
def test_schema_file_is_valid_json():
    data = json.loads(SCHEMA_PATH.read_text())
    assert data["title"] == "Scenario DNA v0.1"
    assert data["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.schema
@pytest.mark.unit
def test_full_valid_document(validator):
    errors = list(validator.iter_errors(_valid_doc()))
    assert errors == [], [e.message for e in errors]


@pytest.mark.schema
@pytest.mark.unit
def test_full_valid_document_with_actor(validator):
    doc = _valid_doc()
    doc["actor_dynamics"] = [
        {
            "actor_class": "pedestrian",
            "state": "crossing",
            "distance_bucket": "near",
            "confidence": 0.85,
            "grounded_by_yolo26": True,
        }
    ]
    errors = list(validator.iter_errors(doc))
    assert errors == [], [e.message for e in errors]


# ── dna_version ───────────────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
def test_dna_version_correct(validator):
    doc = _valid_doc()
    doc["dna_version"] = "0.1.0"
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_dna_version_wrong_fails(validator):
    doc = _valid_doc()
    doc["dna_version"] = "0.2.0"
    assert _fails(validator, doc)


# ── timestamp_range ───────────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
def test_timestamp_range_valid(validator):
    doc = _valid_doc()
    doc["timestamp_range"] = {"start_s": 1.5, "end_s": 11.5}
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_timestamp_range_missing_end_fails(validator):
    doc = _valid_doc()
    doc["timestamp_range"] = {"start_s": 0.0}
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_timestamp_range_negative_fails(validator):
    doc = _valid_doc()
    doc["timestamp_range"] = {"start_s": -1.0, "end_s": 10.0}
    assert _fails(validator, doc)


# ── ODD — weather ─────────────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "clear", "overcast", "light_rain", "heavy_rain",
    "snow", "heavy_snow", "fog", "mist", "sleet",
])
def test_weather_valid(validator, value):
    doc = _valid_doc()
    doc["odd"]["weather"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_weather_invalid_fails(validator):
    doc = _valid_doc()
    doc["odd"]["weather"] = "hail"
    assert _fails(validator, doc)


# ── ODD — lighting ────────────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", ["day", "dawn", "dusk", "night", "tunnel", "overcast_day"])
def test_lighting_valid(validator, value):
    doc = _valid_doc()
    doc["odd"]["lighting"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_lighting_invalid_fails(validator):
    doc = _valid_doc()
    doc["odd"]["lighting"] = "evening"
    assert _fails(validator, doc)


# ── ODD — sensor_fidelity ─────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
def test_sensor_fidelity_empty_valid(validator):
    doc = _valid_doc()
    doc["odd"]["sensor_fidelity"] = []
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_sensor_fidelity_multiple_valid(validator):
    doc = _valid_doc()
    doc["odd"]["sensor_fidelity"] = ["lens_flare", "motion_blur"]
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_sensor_fidelity_invalid_item_fails(validator):
    doc = _valid_doc()
    doc["odd"]["sensor_fidelity"] = ["rain_on_lens"]
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_sensor_fidelity_duplicates_fail(validator):
    doc = _valid_doc()
    doc["odd"]["sensor_fidelity"] = ["lens_flare", "lens_flare"]
    assert _fails(validator, doc)


# ── Topology — road_type ──────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "motorway", "trunk", "primary", "secondary", "residential",
    "service", "rural", "parking", "walkway", "cycling",
])
def test_road_type_valid(validator, value):
    doc = _valid_doc()
    doc["topology"]["road_type"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_road_type_old_highway_fails(validator):
    """'highway' replaced by 'motorway' per OpenDRIVE e_roadType."""
    doc = _valid_doc()
    doc["topology"]["road_type"] = "highway"
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_road_type_old_urban_fails(validator):
    """'urban' split into primary/secondary per OpenDRIVE e_roadType."""
    doc = _valid_doc()
    doc["topology"]["road_type"] = "urban"
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_road_type_old_tunnel_road_fails(validator):
    """'tunnel_road' removed — tunnel is an OpenDRIVE object attribute."""
    doc = _valid_doc()
    doc["topology"]["road_type"] = "tunnel_road"
    assert _fails(validator, doc)


# ── Topology — lane_event ─────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "normal", "construction_divert", "lane_closed", "merge", "split", "unmarked",
])
def test_lane_event_valid(validator, value):
    doc = _valid_doc()
    doc["topology"]["lane_event"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_lane_event_invalid_fails(validator):
    doc = _valid_doc()
    doc["topology"]["lane_event"] = "blocked"
    assert _fails(validator, doc)


# ── Topology — intersection_type ──────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "none", "signalized", "unsignalized", "roundabout",
    "t_junction", "crosswalk", "direct_connection",
])
def test_intersection_type_valid(validator, value):
    doc = _valid_doc()
    doc["topology"]["intersection_type"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_intersection_type_invalid_fails(validator):
    doc = _valid_doc()
    doc["topology"]["intersection_type"] = "diamond_interchange"
    assert _fails(validator, doc)


# ── Actor Dynamics — actor_class ──────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "pedestrian", "cyclist", "motorcyclist",
    "vehicle_car", "vehicle_van", "vehicle_truck", "vehicle_bus",
    "vehicle_emergency", "vehicle_construction",
    "animal", "debris", "construction_object", "obstacle",
    "standup_scooter_rider", "e_bike_rider", "delivery_motorcycle",
    "wheelchair_user",
])
def test_actor_class_valid(validator, value):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor(value)]
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_actor_class_old_vehicle_sedan_fails(validator):
    """vehicle_sedan consolidated into vehicle_car (ASAM OSI TYPE_CAR)."""
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("vehicle_sedan")]
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_actor_class_old_vehicle_suv_fails(validator):
    """vehicle_suv consolidated into vehicle_car (ASAM OSI TYPE_CAR)."""
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("vehicle_suv")]
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_actor_class_old_escooter_rider_fails(validator):
    """escooter_rider renamed to standup_scooter_rider (ASAM OSI TYPE_STANDUP_SCOOTER)."""
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("escooter_rider")]
    assert _fails(validator, doc)


# ── Actor Dynamics — state & distance_bucket ──────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "crossing", "hesitating", "jaywalking", "cutin", "cutout",
    "stopped", "emerging", "tailing", "oncoming", "parked", "static",
])
def test_actor_state_valid(validator, value):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("pedestrian")]
    doc["actor_dynamics"][0]["state"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", ["near", "mid", "far"])
def test_actor_distance_bucket_valid(validator, value):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("pedestrian")]
    doc["actor_dynamics"][0]["distance_bucket"] = value
    assert not _fails(validator, doc)


# ── Actor Dynamics — confidence range ────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_actor_confidence_valid(validator, value):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("pedestrian")]
    doc["actor_dynamics"][0]["confidence"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_actor_confidence_above_one_fails(validator):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("pedestrian")]
    doc["actor_dynamics"][0]["confidence"] = 1.01
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_actor_confidence_negative_fails(validator):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("pedestrian")]
    doc["actor_dynamics"][0]["confidence"] = -0.1
    assert _fails(validator, doc)


# ── Planner Logic — ego_maneuver ──────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "cruise", "accelerate", "brake_soft", "brake_hard", "emergency_brake",
    "nudge_left", "nudge_right", "lane_change_left", "lane_change_right",
    "yield", "stop", "reverse", "swerve",
])
def test_ego_maneuver_valid(validator, value):
    doc = _valid_doc()
    doc["planner_logic"]["ego_maneuver"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_ego_maneuver_invalid_fails(validator):
    doc = _valid_doc()
    doc["planner_logic"]["ego_maneuver"] = "u_turn"
    assert _fails(validator, doc)


# ── Planner Logic — risk_level ────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("value", ["nominal", "elevated", "critical"])
def test_risk_level_valid(validator, value):
    doc = _valid_doc()
    doc["planner_logic"]["risk_level"] = value
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_risk_level_invalid_fails(validator):
    doc = _valid_doc()
    doc["planner_logic"]["risk_level"] = "high"
    assert _fails(validator, doc)


# ── Planner Logic — causal_trigger_actor_index ───────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
def test_causal_trigger_null_valid(validator):
    doc = _valid_doc()
    doc["planner_logic"]["causal_trigger_actor_index"] = None
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_causal_trigger_zero_valid(validator):
    doc = _valid_doc()
    doc["planner_logic"]["causal_trigger_actor_index"] = 0
    assert not _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_causal_trigger_string_fails(validator):
    doc = _valid_doc()
    doc["planner_logic"]["causal_trigger_actor_index"] = "actor_0"
    assert _fails(validator, doc)


# ── additionalProperties: false ───────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
def test_extra_top_level_field_fails(validator):
    doc = _valid_doc()
    doc["unknown_field"] = "oops"
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_extra_field_in_odd_fails(validator):
    doc = _valid_doc()
    doc["odd"]["humidity"] = 0.8
    assert _fails(validator, doc)


@pytest.mark.schema
@pytest.mark.unit
def test_extra_field_in_actor_fails(validator):
    doc = _valid_doc()
    doc["actor_dynamics"] = [_actor("pedestrian")]
    doc["actor_dynamics"][0]["speed_mps"] = 1.5
    assert _fails(validator, doc)


# ── Required fields ───────────────────────────────────────────────────────────

@pytest.mark.schema
@pytest.mark.unit
@pytest.mark.parametrize("field", [
    "dna_version", "clip_id", "timestamp_range",
    "odd", "topology", "actor_dynamics",
    "planner_logic", "confidence", "provenance",
])
def test_missing_required_top_level_field_fails(validator, field):
    doc = _valid_doc()
    del doc[field]
    assert _fails(validator, doc)
