"""Unit tests for the v0.2 re-curation DNA normalizer (hotfix)."""

from __future__ import annotations

import pytest

from my_curator.domain.scout.dna_normalizer import (
    ensure_managed_fields,
    normalize_dna,
)
from my_curator.domain.scout.dna_validator import DNAValidator


def _flattened_scout_output() -> dict:
    """A doc reproducing the measured production failure modes: dotted keys,
    bracketed actor array, missing envelope, 'urban', synonym drift, over-length
    text, and a bad safety_event null."""
    return {
        "clip_id": "00000000-0000-0000-0000-000000000000",
        "scene_description": "x" * 900,
        "odd.weather": "clear",
        "odd.lighting": "overcast",
        "odd.sensor_fidelity": ["clean"],
        "topology.road_type": "urban",
        "topology.lane_event": "normal",
        "topology.intersection_type": "none",
        "actor_dynamics[]": [
            {
                "actor_class": "person",
                "state": "crossing",
                "distance_bucket": "near",
                "confidence": 0.9,
                "grounded_by_yolo26": True,
            }
        ],
        "planner_logic": {
            "ego_maneuver": "nudge right",
            "risk_level": "elevated",
            "risk_level_rationale": "y" * 500,
            "safety_event": {
                "has_event": True,
                "event_type": "near_miss",
                "collision_type": "rear_end",  # must be nulled (event_type != collision)
                "severity_estimate": "minor",
            },
        },
        "confidence": {"overall": 0.9, "scout_agreement": 1.0, "hallucination_flags": []},
    }


@pytest.mark.unit
class TestDeflatten:
    def test_dotted_keys_become_nested(self):
        out = normalize_dna({"odd.weather": "clear", "odd.lighting": "day"})
        assert out["odd"] == {"weather": "clear", "lighting": "day"}
        assert "odd.weather" not in out

    def test_bracket_suffix_stripped(self):
        out = normalize_dna({"actor_dynamics[]": []})
        assert out["actor_dynamics"] == []
        assert "actor_dynamics[]" not in out

    def test_existing_nested_object_wins_over_flat(self):
        out = normalize_dna({"odd": {"weather": "fog"}, "odd.weather": "clear"})
        assert out["odd"]["weather"] == "fog"

    def test_input_not_mutated(self):
        src = {"odd.weather": "clear"}
        normalize_dna(src)
        assert src == {"odd.weather": "clear"}

    def test_bracket_in_intermediate_path_component_stripped(self):
        # "actor_dynamics[].actor_class" must not leave a broken "actor_dynamics[]" key.
        out = normalize_dna({"actor_dynamics[].actor_class": "vehicle_car"})
        assert "actor_dynamics" in out
        assert not any("[]" in k for k in out)


@pytest.mark.unit
class TestEnumCoercion:
    def test_urban_maps_to_primary(self):
        out = normalize_dna({"topology.road_type": "urban"})
        assert out["topology"]["road_type"] == "primary"

    def test_case_and_whitespace_canonicalised(self):
        out = normalize_dna({"planner_logic": {"ego_maneuver": "Nudge Right"}})
        assert out["planner_logic"]["ego_maneuver"] == "nudge_right"

    def test_actor_class_synonyms(self):
        out = normalize_dna(
            {
                "actor_dynamics": [
                    {"actor_class": "person", "state": "crossing", "distance_bucket": "near"},
                    {"actor_class": "bicycle", "state": "static", "distance_bucket": "far"},
                ]
            }
        )
        classes = [a["actor_class"] for a in out["actor_dynamics"]]
        assert classes == ["pedestrian", "cyclist"]

    def test_overcast_lighting_maps(self):
        out = normalize_dna({"odd": {"lighting": "overcast"}})
        assert out["odd"]["lighting"] == "overcast_day"

    def test_valid_enum_untouched(self):
        out = normalize_dna({"topology": {"road_type": "motorway"}})
        assert out["topology"]["road_type"] == "motorway"


@pytest.mark.unit
class TestSafetyEvent:
    def test_collision_type_nulled_when_not_collision(self):
        out = normalize_dna(
            {
                "planner_logic": {
                    "safety_event": {"event_type": "near_miss", "collision_type": "rear_end"}
                }
            }
        )
        assert out["planner_logic"]["safety_event"]["collision_type"] is None

    def test_severity_nulled_when_event_none(self):
        out = normalize_dna(
            {
                "planner_logic": {
                    "safety_event": {"event_type": "none", "severity_estimate": "minor"}
                }
            }
        )
        assert out["planner_logic"]["safety_event"]["severity_estimate"] is None

    def test_default_injected_when_missing(self):
        out = normalize_dna({"planner_logic": {"ego_maneuver": "cruise", "risk_level": "nominal"}})
        se = out["planner_logic"]["safety_event"]
        assert se == {
            "has_event": False,
            "event_type": "none",
            "collision_type": None,
            "severity_estimate": None,
        }

    def test_severity_not_fabricated_for_non_none(self):
        out = normalize_dna(
            {
                "planner_logic": {
                    "safety_event": {"event_type": "near_miss", "severity_estimate": None}
                }
            }
        )
        # normalization must not invent a severity for a real event
        assert out["planner_logic"]["safety_event"]["severity_estimate"] is None


@pytest.mark.unit
class TestTruncation:
    def test_scene_description_capped_at_500(self):
        out = normalize_dna({"scene_description": "a" * 900})
        assert len(out["scene_description"]) <= 500

    def test_rationale_capped_at_300(self):
        out = normalize_dna({"planner_logic": {"risk_level_rationale": "b" * 900}})
        assert len(out["planner_logic"]["risk_level_rationale"]) <= 300

    def test_short_text_untouched(self):
        out = normalize_dna({"scene_description": "short."})
        assert out["scene_description"] == "short."

    def test_sentence_boundary_preferred_when_past_midpoint(self):
        # A period past the halfway mark is used as the cut; earlier tiny-sentence
        # boundaries are ignored in favour of retaining more content.
        text = "A" * 400 + ". " + "z" * 300
        out = normalize_dna({"scene_description": text})["scene_description"]
        assert out.endswith(".")
        assert len(out) <= 500
        assert "z" not in out


@pytest.mark.unit
class TestEnsureManagedFields:
    def test_injects_envelope_when_absent(self):
        dna = ensure_managed_fields(
            {},
            clip_id="abc",
            start_s=1.0,
            end_s=6.0,
            scout_prompt_hash="h",
            pipeline_version="p2-6",
        )
        assert dna["dna_version"] == "0.2.0"
        assert dna["clip_id"] == "abc"
        assert dna["timestamp_range"] == {"start_s": 1.0, "end_s": 6.0}
        assert dna["provenance"]["scout_prompt_hash"] == "h"
        assert dna["provenance"]["pipeline_version"] == "p2-6"
        assert dna["provenance"]["scout_models"]
        assert dna["provenance"]["reference_standards"]

    def test_preserves_existing_nonzero_timestamp(self):
        dna = ensure_managed_fields(
            {"timestamp_range": {"start_s": 2.0, "end_s": 7.0}}, start_s=0.0, end_s=9.0
        )
        assert dna["timestamp_range"] == {"start_s": 2.0, "end_s": 7.0}


@pytest.mark.unit
class TestGenericFallback:
    def test_unknown_weather_defaults_to_clear(self):
        out = normalize_dna({"odd": {"weather": "partly_cloudy"}})
        assert out["odd"]["weather"] == "clear"

    def test_unknown_ego_maneuver_defaults_to_cruise(self):
        out = normalize_dna({"planner_logic": {"ego_maneuver": "follow"}})
        assert out["planner_logic"]["ego_maneuver"] == "cruise"

    def test_actor_with_unmappable_state_is_dropped(self):
        out = normalize_dna(
            {
                "actor_dynamics": [
                    {"actor_class": "vehicle_car", "state": "moving", "distance_bucket": "mid"},
                    {"actor_class": "vehicle_car", "state": "tailing", "distance_bucket": "mid"},
                ]
            }
        )
        assert len(out["actor_dynamics"]) == 1
        assert out["actor_dynamics"][0]["state"] == "tailing"

    def test_valid_values_untouched_by_fallback(self):
        out = normalize_dna({"topology": {"road_type": "motorway"}, "odd": {"weather": "fog"}})
        assert out["topology"]["road_type"] == "motorway"
        assert out["odd"]["weather"] == "fog"

    def test_absent_section_not_fabricated(self):
        # No 'topology' key at all -> fallback must not invent one.
        out = normalize_dna({"odd": {"weather": "clear"}})
        assert "topology" not in out

    def test_absent_required_field_not_fabricated(self):
        # planner_logic present but risk_level absent -> must NOT default to
        # "nominal" (that would mask risk); left missing so it routes to review.
        out = normalize_dna({"planner_logic": {"safety_event": {"event_type": "collision"}}})
        assert "risk_level" not in out["planner_logic"]
        assert "ego_maneuver" not in out["planner_logic"]


@pytest.mark.unit
@pytest.mark.schema
class TestFallbackSetsMatchSchema:
    """Guard the hardcoded fallback enum sets against the live v0.2 schema."""

    def _schema_enum(self, schema, *path):
        node = schema["properties"]
        for p in path[:-1]:
            node = node[p]["properties"]
        return set(node[path[-1]]["enum"])

    def test_enum_sets_and_defaults_match_schema(self):
        import json
        import pathlib

        from my_curator.domain.scout import dna_normalizer as dn

        schema = json.loads(
            (
                pathlib.Path(__file__).resolve().parents[2]
                / "schemas"
                / "scenario_dna_v0_2.schema.json"
            ).read_text()
        )
        for (parent, field), (valid, default) in dn._ENUM_DEFAULTS.items():
            assert valid == self._schema_enum(schema, parent, field), f"{parent}.{field} set drift"
            assert default in valid
        actor_props = schema["properties"]["actor_dynamics"]["items"]["properties"]
        assert set(actor_props["actor_class"]["enum"]) == dn._ACTOR_CLASS
        assert set(actor_props["state"]["enum"]) == dn._ACTOR_STATE
        assert set(actor_props["distance_bucket"]["enum"]) == dn._DISTANCE_BUCKET
        assert (
            set(schema["properties"]["odd"]["properties"]["sensor_fidelity"]["items"]["enum"])
            == dn._SENSOR_FIDELITY
        )


@pytest.mark.unit
@pytest.mark.schema
class TestEndToEndRepair:
    def test_flattened_output_becomes_valid_v02(self):
        raw = _flattened_scout_output()
        repaired = normalize_dna(raw)
        ensure_managed_fields(
            repaired,
            clip_id="11111111-1111-1111-1111-111111111111",
            start_s=0.0,
            end_s=5.0,
            scout_prompt_hash="h",
            pipeline_version="p2-6",
        )
        ok, errors = DNAValidator().validate(repaired)
        assert ok, errors
        # spot-check the repairs landed
        assert repaired["topology"]["road_type"] == "primary"
        assert repaired["odd"]["lighting"] == "overcast_day"
        assert repaired["actor_dynamics"][0]["actor_class"] == "pedestrian"
        assert repaired["planner_logic"]["ego_maneuver"] == "nudge_right"
        assert repaired["planner_logic"]["safety_event"]["collision_type"] is None
        assert len(repaired["scene_description"]) <= 500
