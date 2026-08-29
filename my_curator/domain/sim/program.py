"""The executable form of a compiled scenario.

``xosc_compiler`` writes an OpenSCENARIO document; the render stage has to read it back and
do what it says. These are the types that stand between the two — everything the executor
needs, with no XML and no CARLA in sight, so the reader and the executor can be tested
apart from each other.

Only the subset the compiler emits is represented. Anything else is an error rather than a
silent omission, which is what keeps the compiler from outgrowing the executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

#: Fog is stated as a visual range in OpenSCENARIO and as a density in CARLA. This is the
#: inverse of the compiler's mapping, and the constant has to match it.
_MAX_VISUAL_RANGE_M = 2000.0


class UnsupportedScenarioError(Exception):
    """The document asks for something this executor cannot stage."""


@dataclass(frozen=True)
class LanePlacement:
    road_id: int
    lane_id: int
    s: float
    offset: float = 0.0


#: DNA states in which the actor moves across the carriageway rather than along it.
CROSSING_STATES = frozenset({"crossing", "hesitating", "jaywalking", "emerging"})


@dataclass(frozen=True)
class EntityDef:
    name: str
    blueprint_filter: str
    role: str
    is_walker: bool
    state: str = ""

    @property
    def is_ego(self) -> bool:
        return self.role == "ego_vehicle"

    @property
    def crosses_the_road(self) -> bool:
        """Whether this entity's motion is across the lane instead of along it."""
        return self.state in CROSSING_STATES


@dataclass(frozen=True)
class SpeedAction:
    target_mps: float
    shape: str
    over_s: float


@dataclass(frozen=True)
class LaneChangeAction:
    relative_lane: int
    entity_ref: str
    over_s: float


@dataclass(frozen=True)
class LaneOffsetAction:
    offset_m: float


@dataclass(frozen=True)
class TeleportAction:
    placement: LanePlacement


# A runtime alias, not an annotation, so `from __future__ import annotations` cannot defer
# it and `X | Y` would fail on the py3.7 the simulator image runs.
Action = Union[SpeedAction, LaneChangeAction, LaneOffsetAction, TeleportAction]  # noqa: UP007


@dataclass(frozen=True)
class Trigger:
    """When an event fires: either off the clock or off ego closing in."""

    kind: str
    value: float
    entity_ref: str = ""

    @property
    def is_timed(self) -> bool:
        return self.kind == "time"


@dataclass(frozen=True)
class ScenarioEvent:
    name: str
    actions: tuple[Action, ...]
    trigger: Trigger


@dataclass(frozen=True)
class ManeuverGroup:
    entity: str
    events: tuple[ScenarioEvent, ...]


@dataclass(frozen=True)
class InitState:
    entity: str
    placement: LanePlacement
    speed_mps: float


@dataclass(frozen=True)
class Environment:
    cloudiness: float
    precipitation: float
    visual_range_m: float
    sun_elevation: float
    sun_azimuth: float
    friction: float

    def to_weather(self) -> dict[str, float]:
        """CARLA ``WeatherParameters`` field values, inverting the compiler's mapping."""
        density = 100.0 * (
            1.0 - min(self.visual_range_m, _MAX_VISUAL_RANGE_M) / _MAX_VISUAL_RANGE_M
        )
        wetness = max(0.0, min(100.0, (1.0 - self.friction) / 0.4 * 100.0))
        return {
            "cloudiness": self.cloudiness,
            "precipitation": self.precipitation,
            "precipitation_deposits": self.precipitation,
            "wetness": wetness,
            "fog_density": round(max(0.0, density), 3),
            "sun_altitude_angle": self.sun_elevation,
            "sun_azimuth_angle": self.sun_azimuth,
        }


@dataclass(frozen=True)
class ScenarioProgram:
    """One compiled scenario, ready to execute."""

    clip_id: str
    town: str
    description: str
    environment: Environment
    entities: tuple[EntityDef, ...]
    init: tuple[InitState, ...]
    groups: tuple[ManeuverGroup, ...]
    stop_time_s: float

    @property
    def ego(self) -> EntityDef:
        for entity in self.entities:
            if entity.is_ego:
                return entity
        raise UnsupportedScenarioError("scenario declares no ego vehicle")

    def init_for(self, entity: str) -> InitState | None:
        return next((state for state in self.init if state.entity == entity), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "town": self.town,
            "entities": [e.name for e in self.entities],
            "groups": {g.entity: [ev.name for ev in g.events] for g in self.groups},
            "stop_time_s": self.stop_time_s,
        }
