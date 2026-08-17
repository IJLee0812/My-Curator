"""``SimSpec`` — the engine-agnostic intermediate representation between DNA and CARLA.

Free of CARLA types so it stays host-testable and serializable: blueprint *filters* and
template *names* are strings, resolved late by the compiler and executor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from my_curator.domain.sim.reasons import DegradationCode

# The warm-up runs but is never recorded, so a synthetic clip and its source have
# identical duration for the embedding comparison.
DEFAULT_SEGMENT_S = 5.0
WARMUP_S = 3.0

# The source corpus has no recoverable camera intrinsics — mount, FOV and even
# forward-facing orientation vary per clip — so every render uses one fixed rig. Kept as
# fields so a future per-clip override is a data change only.
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
RENDER_FPS = 10
DEFAULT_FOV = 90.0

# (x, y, z, roll, pitch, yaw) in the ego's local frame, meters/degrees.
EGO_CAM_TRANSFORM = (1.6, 0.0, 1.5, 0.0, 0.0, 0.0)
CHASE_CAM_TRANSFORM = (-6.0, 0.0, 3.0, 0.0, -12.0, 0.0)


@dataclass(frozen=True)
class Degradation:
    """One recorded compromise: ``requested`` could not be staged, ``applied`` was."""

    code: DegradationCode
    field_path: str
    requested: str
    applied: str

    @property
    def note(self) -> str:
        return self.code.note

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "field_path": self.field_path,
            "requested": self.requested,
            "applied": self.applied,
            "note": self.note,
        }


@dataclass(frozen=True)
class CameraSpec:
    """One recorded view. Two per spec: first-person ego and third-person chase."""

    view: str
    transform: tuple[float, float, float, float, float, float]
    image_size_x: int = RENDER_WIDTH
    image_size_y: int = RENDER_HEIGHT
    fov: float = DEFAULT_FOV
    fps: int = RENDER_FPS
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoadQuery:
    """What the compiler must find in a town — a *query*, not a resolved road.

    Mapping asserts only that a satisfying road exists in at least one loadable town
    (``candidate_towns``); picking the specific road id is the compiler's job.
    """

    road_type: str
    intersection_type: str
    min_driving_lanes: int
    speed_kph_range: tuple[int, int]
    required_lane_types: tuple[str, ...]
    candidate_towns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PropSpec:
    """Static dressing used to express a lane event the road network cannot encode."""

    blueprint: str
    count: int
    placement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorldSpec:
    """Environment + road requirements."""

    weather: dict[str, float]
    road: RoadQuery
    props: tuple[PropSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "weather": dict(self.weather),
            "road": self.road.to_dict(),
            "props": [p.to_dict() for p in self.props],
        }


class ControlMode(str, Enum):
    """How an actor is driven during the render.

    * ``EVENT`` — the actor the safety event is about; its maneuver is timed *relative
      to ego*, which is the interaction the segment exists to show.
    * ``SCRIPTED`` — a deterministic, ego-independent behavior (hold position, cross,
      change lane on a fixed cue). Reproduces the stated state without choreography.
    * ``AMBIENT`` — genuine background traffic, handed to the traffic manager.

    Most states must be scripted: an autopilot cannot reproduce ``parked``/``stopped``
    (it drives away) or ``cutin`` (it never happens), and demoting them would discard the
    ``actor_dynamics[].state`` the DNA asserts.
    """

    EVENT = "event"
    SCRIPTED = "scripted"
    AMBIENT = "ambient"


@dataclass(frozen=True)
class ActorSpec:
    """One non-ego participant."""

    index: int
    actor_class: str
    blueprint_filter: str
    state: str
    maneuver_template: str
    distance_bucket: str
    distance_m: float
    control_mode: ControlMode = ControlMode.AMBIENT

    @property
    def is_event_actor(self) -> bool:
        """The actor the ``safety_event`` is about (at most one per spec)."""
        return self.control_mode is ControlMode.EVENT

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["control_mode"] = self.control_mode.value
        data["is_event_actor"] = self.is_event_actor
        return data


@dataclass(frozen=True)
class EgoSpec:
    maneuver: str
    control_template: str
    target_speed_kph: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyEventSpec:
    has_event: bool
    event_type: str
    collision_type: str | None
    severity_estimate: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimSpec:
    """A fully-resolved, engine-agnostic reconstruction plan for one segment."""

    clip_id: str
    dna_version: str
    duration_s: float
    warmup_s: float
    risk_level: str
    world: WorldSpec
    ego: EgoSpec
    actors: tuple[ActorSpec, ...]
    cameras: tuple[CameraSpec, ...]
    safety_event: SafetyEventSpec
    degradations: tuple[Degradation, ...] = ()

    @property
    def is_degraded(self) -> bool:
        return bool(self.degradations)

    @property
    def event_actor(self) -> ActorSpec | None:
        return next((a for a in self.actors if a.is_event_actor), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "dna_version": self.dna_version,
            "duration_s": self.duration_s,
            "warmup_s": self.warmup_s,
            "risk_level": self.risk_level,
            "world": self.world.to_dict(),
            "ego": self.ego.to_dict(),
            "actors": [a.to_dict() for a in self.actors],
            "cameras": [c.to_dict() for c in self.cameras],
            "safety_event": self.safety_event.to_dict(),
            "degradations": [d.to_dict() for d in self.degradations],
        }
