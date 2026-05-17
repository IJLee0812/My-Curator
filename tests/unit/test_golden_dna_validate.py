"""GT-2: golden DNA validator equivalence.

Locks the (valid, error_count) tuple produced by ``DNAValidator.validate()`` for
14 representative DNA dicts so that any change in either the validator
implementation or the underlying schema is caught immediately.

The golden table below was captured against the live validator at R-0; any drift
across the R-1..R-7 refactoring stages — where the validator moves from
``src/scouts/dna_validator.py`` to ``my_curator/domain/scout/dna_validator.py`` —
must produce the same outcome.

References:
  docs/refactoring_plan.md  §3.1 GT-2.
"""

from __future__ import annotations

import copy

import pytest

from my_curator.domain.scout.dna_validator import DNAValidator

_VALID_DNA: dict = {
    "dna_version": "0.1.0",
    "clip_id": "12345678-1234-5678-1234-567812345678",
    "timestamp_range": {"start_s": 0.0, "end_s": 5.0},
    "odd": {"weather": "clear", "lighting": "day", "sensor_fidelity": ["clean"]},
    "topology": {"road_type": "primary", "lane_event": "normal", "intersection_type": "none"},
    "actor_dynamics": [],
    "planner_logic": {"ego_maneuver": "cruise", "risk_level": "nominal"},
    "confidence": {"overall": 0.9, "scout_agreement": 1.0, "hallucination_flags": []},
    "provenance": {
        "scout_models": ["cosmos-reason2-8b"],
        "scout_prompt_hash": "abcd1234ef567890",
        "pipeline_version": "p2-6",
        "is_synthetic": False,
        "reference_standards": ["ASAM OSI v3.x"],
    },
}


def _with(patch: dict) -> dict:
    d = copy.deepcopy(_VALID_DNA)
    for path, value in patch.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return d


def _without(path: str) -> dict:
    d = copy.deepcopy(_VALID_DNA)
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    cur.pop(keys[-1], None)
    return d


# (case_name, dna_dict, expected_valid, expected_min_error_count)
_GOLDEN_CASES = [
    ("baseline_valid", copy.deepcopy(_VALID_DNA), True, 0),
    ("missing_odd", _without("odd"), False, 1),
    ("missing_risk_level", _without("planner_logic.risk_level"), False, 1),
    ("bad_weather_enum", _with({"odd.weather": "partly_cloudy"}), False, 1),
    ("bad_road_type_enum", _with({"topology.road_type": "urban_road"}), False, 1),
    ("bad_ego_maneuver", _with({"planner_logic.ego_maneuver": "going_straight"}), False, 1),
    ("dna_version_drift", _with({"dna_version": "0.2.0"}), False, 1),
    ("confidence_out_of_range", _with({"confidence.overall": 1.5}), False, 1),
    ("bad_sensor_fidelity", _with({"odd.sensor_fidelity": ["clean", "dirty_lens"]}), False, 1),
    (
        "multiple_errors",
        _with({"odd.weather": "partly_cloudy", "topology.road_type": "urban_road"}),
        False,
        2,
    ),
    (
        "valid_actor",
        _with(
            {
                "actor_dynamics": [
                    {
                        "actor_class": "vehicle_car",
                        "state": "tailing",
                        "distance_bucket": "mid",
                        "confidence": 0.85,
                        "grounded_by_yolo26": True,
                    }
                ]
            }
        ),
        True,
        0,
    ),
    (
        "bad_actor_class",
        _with(
            {
                "actor_dynamics": [
                    {
                        "actor_class": "car",
                        "state": "tailing",
                        "distance_bucket": "mid",
                        "confidence": 0.85,
                        "grounded_by_yolo26": True,
                    }
                ]
            }
        ),
        False,
        1,
    ),
    ("extra_top_level_field", _with({"clip_id": _VALID_DNA["clip_id"]}) | {"extra": "x"}, False, 1),
    ("bad_lighting_enum", _with({"odd.lighting": "twilight"}), False, 1),
]


@pytest.fixture(scope="module")
def validator() -> DNAValidator:
    return DNAValidator()


@pytest.mark.unit
@pytest.mark.schema
@pytest.mark.parametrize(
    "name,dna,expected_valid,min_errors", _GOLDEN_CASES, ids=[c[0] for c in _GOLDEN_CASES]
)
def test_golden_dna_validate(validator, name, dna, expected_valid, min_errors):
    ok, errors = validator.validate(dna)
    assert ok is expected_valid, (
        f"[{name}] validity drift: got valid={ok}, expected={expected_valid}; errors={errors}"
    )
    if not expected_valid:
        assert len(errors) >= min_errors, (
            f"[{name}] expected ≥{min_errors} errors, got {len(errors)}: {errors}"
        )


@pytest.mark.unit
def test_golden_case_count():
    """Hard-pin the case count so accidental deletion is caught."""
    assert len(_GOLDEN_CASES) == 14
