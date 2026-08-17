"""Reason codes for the DNA -> SimSpec mapping.

Two orthogonal vocabularies: :class:`ExclusionReason` means nothing can be staged and the
segment drops out of the coverage numerator; :class:`DegradationCode` means the segment is
staged but a compromise was made and recorded. Each code carries a ``note`` so the
coverage report explains itself without a lookup table.
"""

from __future__ import annotations

from enum import Enum


class ExclusionReason(str, Enum):
    """Why a segment cannot be re-staged at all."""

    DNA_INCOMPLETE = "dna_incomplete"
    UNKNOWN_ENUM_VALUE = "unknown_enum_value"
    UNSUPPORTED_ROAD_TYPE = "unsupported_road_type"
    UNSUPPORTED_INTERSECTION = "unsupported_intersection"
    UNSUPPORTED_EGO_MANEUVER = "unsupported_ego_maneuver"

    @property
    def note(self) -> str:
        return _EXCLUSION_NOTES[self]


_EXCLUSION_NOTES: dict[ExclusionReason, str] = {
    ExclusionReason.DNA_INCOMPLETE: (
        "a required v0.2 block or enum is absent — the segment never produced usable DNA "
        "(VLM degeneration quarantine), so there is nothing to reconstruct"
    ),
    ExclusionReason.UNKNOWN_ENUM_VALUE: (
        "the stored value is outside the v0.2 schema enum, so no mapping can be trusted"
    ),
    ExclusionReason.UNSUPPORTED_ROAD_TYPE: (
        "the road class is not ego-drivable in the built-in towns (no such carriageway exists)"
    ),
    ExclusionReason.UNSUPPORTED_INTERSECTION: ("no built-in town contains this junction form"),
    ExclusionReason.UNSUPPORTED_EGO_MANEUVER: (
        "the ego maneuver cannot be scripted against a static built-in road network"
    ),
}


class DegradationCode(str, Enum):
    """What was compromised while still staging the segment."""

    WEATHER_NO_SNOW = "weather_no_snow"
    WEATHER_APPROXIMATED = "weather_approximated"
    LIGHTING_APPROXIMATED = "lighting_approximated"
    SENSOR_EFFECT_UNAVAILABLE = "sensor_effect_unavailable"
    ROAD_TYPE_SUBSTITUTED = "road_type_substituted"
    INTERSECTION_SUBSTITUTED = "intersection_substituted"
    LANE_EVENT_PROP_STAGED = "lane_event_prop_staged"
    ACTOR_BLUEPRINT_SUBSTITUTED = "actor_blueprint_substituted"
    ACTOR_STATE_APPROXIMATED = "actor_state_approximated"
    ACTOR_DROPPED = "actor_dropped"
    EGO_MANEUVER_APPROXIMATED = "ego_maneuver_approximated"
    EVENT_ACTOR_INFERRED = "event_actor_inferred"
    COLLISION_NOT_STAGED = "collision_not_staged"

    @property
    def note(self) -> str:
        return _DEGRADATION_NOTES[self]


_DEGRADATION_NOTES: dict[DegradationCode, str] = {
    DegradationCode.WEATHER_NO_SNOW: (
        "CARLA 0.9.15 WeatherParameters has no snow channel (measured: 14 fields, none "
        "frozen-precipitation) — rendered as the closest heavy-precipitation overcast"
    ),
    DegradationCode.WEATHER_APPROXIMATED: (
        "no exact weather channel; composed from the nearest cloudiness/precipitation/fog mix"
    ),
    DegradationCode.LIGHTING_APPROXIMATED: (
        "lighting realized only through sun altitude; the exact light environment differs"
    ),
    DegradationCode.SENSOR_EFFECT_UNAVAILABLE: (
        "the RGB camera exposes no matching post-process attribute for this artifact"
    ),
    DegradationCode.ROAD_TYPE_SUBSTITUTED: (
        "the requested road class has no built-in equivalent; the nearest drivable class is used"
    ),
    DegradationCode.INTERSECTION_SUBSTITUTED: (
        "the junction form is substituted by the nearest available geometry"
    ),
    DegradationCode.LANE_EVENT_PROP_STAGED: (
        "the lane event is dressed with static props rather than an altered road network"
    ),
    DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED: (
        "no blueprint for this actor class; the nearest silhouette is spawned instead"
    ),
    DegradationCode.ACTOR_STATE_APPROXIMATED: (
        "the behavior has no exact maneuver template; the nearest template is scripted"
    ),
    DegradationCode.ACTOR_DROPPED: (
        "the actor has no representable blueprint at all and is omitted from the scene"
    ),
    DegradationCode.EGO_MANEUVER_APPROXIMATED: (
        "the ego maneuver is approximated by the nearest scriptable control template"
    ),
    DegradationCode.EVENT_ACTOR_INFERRED: (
        "planner_logic.causal_trigger_actor_index was absent, so the event actor was inferred "
        "from actor state and distance"
    ),
    DegradationCode.COLLISION_NOT_STAGED: (
        "a contact event is reproduced up to the pre-impact geometry only; no impact is forced"
    ),
}
