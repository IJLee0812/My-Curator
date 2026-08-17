"""Every v0.2 DNA enum value must have an explicit CARLA mapping.

``test_every_enum_value_is_mapped`` reads ``schemas/scenario_dna_v0_2.schema.json`` rather
than a hardcoded list, so adding an enum value to the schema without deciding how to stage
it breaks the build.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from my_curator.domain.sim import catalog as cat
from my_curator.domain.sim.reasons import DegradationCode, ExclusionReason

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "schemas" / "scenario_dna_v0_2.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _enum(schema: dict, *path: str) -> set[str]:
    node = schema["properties"]
    for key in path[:-1]:
        node = node[key]["properties"]
    return set(node[path[-1]]["enum"])


@pytest.mark.unit
@pytest.mark.schema
class TestCatalogsCoverSchema:
    """Exhaustiveness of every catalog against the live schema."""

    def test_every_enum_value_is_mapped(self, schema):
        actor_props = schema["properties"]["actor_dynamics"]["items"]["properties"]
        cases = [
            ("odd.weather", _enum(schema, "odd", "weather"), set(cat.WEATHER)),
            ("odd.lighting", _enum(schema, "odd", "lighting"), set(cat.LIGHTING)),
            (
                "odd.sensor_fidelity",
                set(schema["properties"]["odd"]["properties"]["sensor_fidelity"]["items"]["enum"]),
                set(cat.SENSOR_FIDELITY),
            ),
            ("topology.road_type", _enum(schema, "topology", "road_type"), set(cat.ROAD_TYPE)),
            ("topology.lane_event", _enum(schema, "topology", "lane_event"), set(cat.LANE_EVENT)),
            (
                "topology.intersection_type",
                _enum(schema, "topology", "intersection_type"),
                set(cat.INTERSECTION_TYPE),
            ),
            (
                "planner_logic.ego_maneuver",
                _enum(schema, "planner_logic", "ego_maneuver"),
                set(cat.EGO_MANEUVER),
            ),
            (
                "planner_logic.risk_level",
                _enum(schema, "planner_logic", "risk_level"),
                set(cat.RISK_LEVELS),
            ),
            (
                "actor_dynamics.actor_class",
                set(actor_props["actor_class"]["enum"]),
                set(cat.ACTOR_CLASS),
            ),
            ("actor_dynamics.state", set(actor_props["state"]["enum"]), set(cat.ACTOR_STATE)),
            (
                "actor_dynamics.distance_bucket",
                set(actor_props["distance_bucket"]["enum"]),
                set(cat.DISTANCE_BUCKETS),
            ),
        ]
        for field, schema_values, catalog_values in cases:
            assert not schema_values - catalog_values, f"{field}: unmapped enum value(s)"
            assert not catalog_values - schema_values, f"{field}: catalog value(s) not in schema"

    def test_safety_event_sets_match_schema(self, schema):
        safety = schema["properties"]["planner_logic"]["properties"]["safety_event"]["properties"]
        assert set(safety["event_type"]["enum"]) == cat.SAFETY_EVENT_TYPES
        assert {v for v in safety["collision_type"]["enum"] if v is not None} == cat.COLLISION_TYPES
        assert {
            v for v in safety["severity_estimate"]["enum"] if v is not None
        } == cat.SEVERITY_ESTIMATES

    def test_event_actor_priority_is_a_subset_of_actor_states(self):
        assert set(cat.EVENT_ACTOR_STATE_PRIORITY) <= set(cat.ACTOR_STATE)
        assert len(set(cat.EVENT_ACTOR_STATE_PRIORITY)) == len(cat.EVENT_ACTOR_STATE_PRIORITY)


@pytest.mark.unit
class TestCatalogInternalConsistency:
    """Guards on the catalog's own invariants, independent of the schema."""

    def test_every_town_reference_is_loadable(self):
        for name, mapping in cat.ROAD_TYPE.items():
            unknown = set(mapping.towns) - set(cat.LOADABLE_TOWNS)
            assert not unknown, f"road_type {name} references non-loadable town(s) {unknown}"
        for name, mapping in cat.INTERSECTION_TYPE.items():
            unknown = set(mapping.towns) - set(cat.LOADABLE_TOWNS)
            assert not unknown, f"intersection {name} references non-loadable town(s) {unknown}"

    def test_town_profiles_cover_the_loadable_set(self):
        assert set(cat.TOWN_PROFILES) == set(cat.LOADABLE_TOWNS)
        assert not set(cat.UNAVAILABLE_TOWNS) & set(cat.LOADABLE_TOWNS)

    def test_excluded_road_types_declare_no_towns(self):
        """An exclusion must not also advertise a staging target."""
        for name, mapping in cat.ROAD_TYPE.items():
            if mapping.exclusion is not None:
                assert mapping.towns == (), f"{name} is excluded but lists towns"

    def test_every_degrading_entry_explains_itself(self):
        """A degradation without an ``applied`` string would produce an unreadable report."""
        tables = (
            ("weather", cat.WEATHER),
            ("lighting", cat.LIGHTING),
            ("sensor_fidelity", cat.SENSOR_FIDELITY),
            ("road_type", cat.ROAD_TYPE),
            ("intersection_type", cat.INTERSECTION_TYPE),
            ("lane_event", cat.LANE_EVENT),
            ("actor_class", cat.ACTOR_CLASS),
            ("actor_state", cat.ACTOR_STATE),
            ("ego_maneuver", cat.EGO_MANEUVER),
        )
        for table_name, table in tables:
            for key, mapping in table.items():
                if getattr(mapping, "degradation", None) is not None:
                    assert mapping.applied, f"{table_name}.{key} degrades without an explanation"

    def test_reason_codes_all_carry_notes(self):
        for reason in ExclusionReason:
            assert reason.note
        for code in DegradationCode:
            assert code.note

    def test_road_speed_ranges_are_ordered_and_satisfiable(self):
        for name, mapping in cat.ROAD_TYPE.items():
            low, high = mapping.speed_kph_range
            assert low <= high, f"{name} has an inverted speed range"
            if mapping.exclusion is not None:
                continue
            satisfying = [
                town
                for town in mapping.towns
                if any(low <= s <= high for s in cat.TOWN_PROFILES[town].speed_kph)
                and cat.TOWN_PROFILES[town].max_driving_lanes >= mapping.min_driving_lanes
            ]
            assert satisfying, f"{name}: no listed town satisfies its own speed/lane requirement"

    def test_required_lane_types_exist_in_candidate_towns(self):
        for name, mapping in cat.ROAD_TYPE.items():
            if mapping.exclusion is not None:
                continue
            for lane_type in mapping.required_lane_types:
                assert any(lane_type in cat.TOWN_PROFILES[t].lane_types for t in mapping.towns), (
                    f"{name}: no candidate town has a '{lane_type}' lane"
                )

    def test_unsignalized_towns_actually_have_unsignalized_junctions(self):
        for town in cat.INTERSECTION_TYPE["unsignalized"].towns:
            assert cat.TOWN_PROFILES[town].unsignalized_junctions > 0

    def test_signalized_towns_actually_have_signals(self):
        for town in cat.INTERSECTION_TYPE["signalized"].towns:
            assert cat.TOWN_PROFILES[town].signalized_junctions > 0
