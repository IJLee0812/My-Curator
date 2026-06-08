"""Unit tests for src/scouts/dna_validator.py (P2-6).

Markers:
  @pytest.mark.unit    — all classes
  @pytest.mark.schema  — validation-related classes
"""

from __future__ import annotations

import copy
import json

import pytest

from my_curator.domain.scout.dna_validator import DNAValidator, _extract_last_object

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_DNA: dict = {
    "dna_version": "0.1.0",
    "clip_id": "12345678-1234-5678-1234-567812345678",
    "timestamp_range": {"start_s": 0.0, "end_s": 5.0},
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
    },
    "confidence": {
        "overall": 0.9,
        "scout_agreement": 1.0,
        "hallucination_flags": [],
    },
    "provenance": {
        "scout_models": ["cosmos-reason2-8b"],
        "scout_prompt_hash": "abcd1234ef567890",
        "pipeline_version": "p2-6",
        "is_synthetic": False,
        "reference_standards": ["ASAM OSI v3.x", "OpenDRIVE v1.5M"],
    },
}

_VALID_DNA_V02: dict = {
    **copy.deepcopy(_VALID_DNA),
    "dna_version": "0.2.0",
    "scene_description": (
        "Clear-day cruise on a two-lane primary road with no active intersection. "
        "Ego maintains a steady following gap with no notable actor interactions. "
        "Routine segment; no safety-relevant event observed."
    ),
    "planner_logic": {
        "ego_maneuver": "cruise",
        "risk_level": "nominal",
        "risk_level_rationale": "Clear primary road, no actors within 50 m, ego cruise within limit.",
        "safety_event": {
            "has_event": False,
            "event_type": "none",
            "collision_type": None,
            "severity_estimate": None,
        },
    },
}

_COT_PREFIX = (
    "Let me analyze the scene step by step.\n\n"
    "Weather: The sky appears clear with good visibility.\n"
    "Road: Two-lane primary road with no active intersection.\n"
    "Actors: No notable actors visible in this segment.\n"
    "Ego: Vehicle appears to be cruising at moderate speed.\n"
    "Risk: No elevated risk factors observed.\n\n"
)


@pytest.fixture
def validator() -> DNAValidator:
    return DNAValidator()


@pytest.fixture
def valid_dna() -> dict:
    return copy.deepcopy(_VALID_DNA)


@pytest.fixture
def valid_dna_v02() -> dict:
    return copy.deepcopy(_VALID_DNA_V02)


# ---------------------------------------------------------------------------
# TestExtractJson
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractJson:
    def test_extracts_from_code_fence(self, validator, valid_dna):
        text = f"```json\n{json.dumps(valid_dna)}\n```"
        result = validator.extract_json(text)
        assert result == valid_dna

    def test_last_fence_wins_with_cot_prefix(self, validator, valid_dna):
        """When CoT contains an incidental JSON example, the last fence is used."""
        early_json = json.dumps({"example": "earlier block"})
        text = (
            f"Here is an example:\n```json\n{early_json}\n```\n\n"
            f"My analysis:\n\n```json\n{json.dumps(valid_dna)}\n```"
        )
        result = validator.extract_json(text)
        assert result == valid_dna

    def test_last_fence_wins_multiple_fences(self, validator, valid_dna):
        first = json.dumps({"wrong": "first"})
        second = json.dumps({"wrong": "second"})
        text = (
            f"```json\n{first}\n```\n```json\n{second}\n```\n```json\n{json.dumps(valid_dna)}\n```"
        )
        result = validator.extract_json(text)
        assert result == valid_dna

    def test_falls_back_to_bare_json_when_no_fence(self, validator, valid_dna):
        text = f"Some reasoning text.\n\n{json.dumps(valid_dna)}\n\nEnd."
        result = validator.extract_json(text)
        assert result == valid_dna

    def test_unterminated_fence_falls_back_to_bare_json(self, validator, valid_dna):
        """Unterminated ```json fence fails stage 1; stage 2 finds bare object."""
        text = f"```json\n{json.dumps(valid_dna)}"  # no closing ```
        result = validator.extract_json(text)
        assert result == valid_dna

    def test_returns_none_on_plain_text(self, validator):
        result = validator.extract_json("No JSON here at all.")
        assert result is None

    def test_returns_none_on_empty_string(self, validator):
        result = validator.extract_json("")
        assert result is None

    def test_returns_none_on_invalid_json_in_fence(self, validator):
        text = "```json\n{not valid json}\n```"
        result = validator.extract_json(text)
        # Stage 1 fails; stage 2 also fails ({not valid json} is not parseable)
        assert result is None

    def test_returns_dict_not_list(self, validator):
        text = "```json\n[1, 2, 3]\n```"
        result = validator.extract_json(text)
        # Arrays are not dicts — should not be returned
        assert result is None

    def test_cot_prefix_with_code_fence(self, validator, valid_dna):
        text = _COT_PREFIX + f"```json\n{json.dumps(valid_dna)}\n```"
        result = validator.extract_json(text)
        assert result == valid_dna


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.schema
class TestValidate:
    def test_valid_dna_passes(self, validator, valid_dna):
        ok, errors = validator.validate(valid_dna)
        assert ok is True
        assert errors == []

    def test_missing_required_top_level_field_fails(self, validator, valid_dna):
        del valid_dna["odd"]
        ok, errors = validator.validate(valid_dna)
        assert ok is False
        assert any("odd" in e for e in errors)

    def test_missing_required_nested_field_fails(self, validator, valid_dna):
        del valid_dna["planner_logic"]["risk_level"]
        ok, errors = validator.validate(valid_dna)
        assert ok is False
        assert errors

    def test_invalid_weather_enum_fails(self, validator, valid_dna):
        valid_dna["odd"]["weather"] = "partly_cloudy"  # not in enum
        ok, errors = validator.validate(valid_dna)
        assert ok is False
        assert errors

    def test_invalid_road_type_enum_fails(self, validator, valid_dna):
        valid_dna["topology"]["road_type"] = "urban_road"  # old pre-P2-6 value
        ok, errors = validator.validate(valid_dna)
        assert ok is False

    def test_invalid_ego_maneuver_enum_fails(self, validator, valid_dna):
        valid_dna["planner_logic"]["ego_maneuver"] = "going_straight"  # old value
        ok, errors = validator.validate(valid_dna)
        assert ok is False

    def test_additional_properties_blocked(self, validator, valid_dna):
        valid_dna["scene_summary"] = "extra field not in schema"
        ok, errors = validator.validate(valid_dna)
        assert ok is False
        assert any("scene_summary" in e for e in errors)

    def test_error_messages_are_strings(self, validator, valid_dna):
        valid_dna["odd"]["weather"] = "rainy"
        ok, errors = validator.validate(valid_dna)
        assert not ok
        for msg in errors:
            assert isinstance(msg, str)

    def test_actor_with_valid_fields_passes(self, validator, valid_dna):
        valid_dna["actor_dynamics"] = [
            {
                "actor_class": "vehicle_car",
                "state": "tailing",
                "distance_bucket": "mid",
                "confidence": 0.85,
                "grounded_by_yolo26": True,
            }
        ]
        ok, errors = validator.validate(valid_dna)
        assert ok is True, errors

    def test_actor_with_invalid_class_fails(self, validator, valid_dna):
        valid_dna["actor_dynamics"] = [
            {
                "actor_class": "car",  # not in 17-class enum
                "state": "tailing",
                "distance_bucket": "mid",
                "confidence": 0.85,
                "grounded_by_yolo26": True,
            }
        ]
        ok, errors = validator.validate(valid_dna)
        assert ok is False

    def test_v01_doc_with_v01_version_passes(self, validator, valid_dna):
        # A well-formed v0.1 document validates under its own version.
        ok, _ = validator.validate(valid_dna)
        assert ok is True

    def test_v01_doc_mislabelled_as_v02_fails(self, validator, valid_dna):
        # A v0.1-shaped doc claiming dna_version 0.2.0 is dispatched to the v0.2
        # validator and fails on the missing v0.2-required fields.
        valid_dna["dna_version"] = "0.2.0"
        ok, _ = validator.validate(valid_dna)
        assert ok is False

    def test_confidence_out_of_range_fails(self, validator, valid_dna):
        valid_dna["confidence"]["overall"] = 1.5
        ok, errors = validator.validate(valid_dna)
        assert ok is False

    def test_sensor_fidelity_invalid_item_fails(self, validator, valid_dna):
        valid_dna["odd"]["sensor_fidelity"] = ["clean", "dirty_lens"]  # not in enum
        ok, errors = validator.validate(valid_dna)
        assert ok is False

    def test_multiple_errors_all_surfaced(self, validator, valid_dna):
        valid_dna["odd"]["weather"] = "bad_weather"
        valid_dna["topology"]["road_type"] = "urban_road"
        ok, errors = validator.validate(valid_dna)
        assert ok is False
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# TestVersionDispatch (P4-1 multi-version routing)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.schema
class TestVersionDispatch:
    def test_v01_doc_routes_to_v01_validator(self, validator, valid_dna):
        ok, errors = validator.validate(valid_dna)
        assert ok is True
        assert errors == []

    def test_v02_doc_routes_to_v02_validator(self, validator, valid_dna_v02):
        ok, errors = validator.validate(valid_dna_v02)
        assert ok is True
        assert errors == []

    def test_v02_fields_rejected_under_v01(self, validator, valid_dna):
        # v0.2-only fields on a v0.1-versioned doc violate additionalProperties:false.
        valid_dna["scene_description"] = "should not be allowed under v0.1"
        ok, _ = validator.validate(valid_dna)
        assert ok is False

    def test_unknown_version_is_explicit_error(self, validator, valid_dna_v02):
        valid_dna_v02["dna_version"] = "9.9.9"
        ok, errors = validator.validate(valid_dna_v02)
        assert ok is False
        assert any("dna_version" in e for e in errors)

    def test_missing_version_is_explicit_error(self, validator, valid_dna_v02):
        del valid_dna_v02["dna_version"]
        ok, errors = validator.validate(valid_dna_v02)
        assert ok is False
        assert any("dna_version" in e for e in errors)

    def test_non_dict_input_is_explicit_error(self, validator):
        ok, errors = validator.validate([])  # type: ignore[arg-type]
        assert ok is False
        assert any("dna_version" in e for e in errors)


# ---------------------------------------------------------------------------
# TestExtractLastObject (internal helper)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractLastObject:
    def test_finds_simple_object(self):
        text = 'prefix {"key": "value"} suffix'
        result = _extract_last_object(text)
        assert result == {"key": "value"}

    def test_finds_last_when_multiple(self):
        text = '{"first": 1} some text {"second": 2}'
        result = _extract_last_object(text)
        assert result == {"second": 2}

    def test_handles_nested_objects(self):
        obj = {"outer": {"inner": 42}}
        text = f"prefix {json.dumps(obj)} suffix"
        result = _extract_last_object(text)
        assert result == obj

    def test_returns_none_on_no_braces(self):
        result = _extract_last_object("no json here")
        assert result is None

    def test_returns_none_on_invalid_json(self):
        result = _extract_last_object("{not: valid}")
        assert result is None


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.schema
class TestIntegration:
    def test_full_cot_round_trip(self, validator, valid_dna):
        """CoT text → extract_json → validate — must pass end-to-end."""
        cot_text = _COT_PREFIX + f"```json\n{json.dumps(valid_dna, indent=2)}\n```"
        extracted = validator.extract_json(cot_text)
        assert extracted is not None
        ok, errors = validator.validate(extracted)
        assert ok is True, errors

    def test_schema_invalid_after_extraction_returns_errors(self, validator, valid_dna):
        valid_dna["odd"]["weather"] = "partly_cloudy"  # invalid enum
        cot_text = _COT_PREFIX + f"```json\n{json.dumps(valid_dna)}\n```"
        extracted = validator.extract_json(cot_text)
        assert extracted is not None
        ok, errors = validator.validate(extracted)
        assert ok is False
        assert errors

    def test_no_json_in_text_extract_returns_none(self, validator):
        cot_text = _COT_PREFIX + "I could not determine the scene structure."
        result = validator.extract_json(cot_text)
        assert result is None

    def test_incidental_json_in_cot_ignored(self, validator, valid_dna):
        """JSON fragments in reasoning are ignored; last fence is authoritative."""
        incidental = json.dumps({"odd": {"weather": "partly_cloudy"}})
        cot_text = (
            f"Consider this example: {incidental}\n\n"
            f"But my actual analysis is:\n```json\n{json.dumps(valid_dna)}\n```"
        )
        extracted = validator.extract_json(cot_text)
        ok, errors = validator.validate(extracted)
        assert ok is True, errors
