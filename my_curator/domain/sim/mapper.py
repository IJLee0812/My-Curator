"""Scenario DNA v0.2 -> :class:`SimSpec`.

Deterministic and side-effect free: the same DNA always yields the same spec, which is
what lets the coverage report be a stable artifact and the compiler cache its output.

Mapping is lenient about narrative, strict about staging. Fields that drive what gets
built (weather, road, junction, actors, ego maneuver) must resolve to a known enum or the
segment is excluded. Fields that only describe the outcome (``collision_type``,
``severity_estimate``) are carried through untouched — a value outside the schema is
recorded as an anomaly rather than thrown away, because it is evidence about Scout output
quality.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from my_curator.domain.sim import catalog as cat
from my_curator.domain.sim.reasons import DegradationCode, ExclusionReason
from my_curator.domain.sim.spec import (
    CHASE_CAM_TRANSFORM,
    DEFAULT_SEGMENT_S,
    EGO_CAM_TRANSFORM,
    WARMUP_S,
    ActorSpec,
    CameraSpec,
    ControlMode,
    Degradation,
    EgoSpec,
    PropSpec,
    RoadQuery,
    SafetyEventSpec,
    SimSpec,
    WorldSpec,
)

#: Capped so a 5 s scene stays legible and spawn-collision free; the deployed corpus
#: never exceeds 3 actors per segment anyway.
MAX_ACTORS = 6


@dataclass(frozen=True)
class MappingResult:
    """Outcome for exactly one segment."""

    clip_id: str
    spec: SimSpec | None
    exclusions: tuple[tuple[ExclusionReason, str], ...] = ()
    anomalies: tuple[str, ...] = ()

    @property
    def mapped(self) -> bool:
        return self.spec is not None

    @property
    def degradations(self) -> tuple[Degradation, ...]:
        return self.spec.degradations if self.spec else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "mapped": self.mapped,
            "exclusions": [
                {"reason": r.value, "detail": d, "note": r.note} for r, d in self.exclusions
            ],
            "anomalies": list(self.anomalies),
            "spec": self.spec.to_dict() if self.spec else None,
        }


def _blank(value: Any) -> bool:
    """The VLM-degeneration quarantine leaves required enums present but empty."""
    return value is None or (isinstance(value, str) and not value.strip())


def map_dna(dna: dict[str, Any]) -> MappingResult:
    """Map one v0.2 Scenario DNA document to a :class:`SimSpec`."""
    clip_id = str(dna.get("clip_id", ""))
    odd = dna.get("odd") or {}
    topology = dna.get("topology") or {}
    planner = dna.get("planner_logic") or {}

    exclusions: list[tuple[ExclusionReason, str]] = []
    anomalies: list[str] = []
    degradations: list[Degradation] = []

    required = {
        "odd.weather": odd.get("weather"),
        "odd.lighting": odd.get("lighting"),
        "topology.road_type": topology.get("road_type"),
        "topology.lane_event": topology.get("lane_event"),
        "topology.intersection_type": topology.get("intersection_type"),
        "planner_logic.ego_maneuver": planner.get("ego_maneuver"),
        "planner_logic.risk_level": planner.get("risk_level"),
    }
    missing = sorted(path for path, value in required.items() if _blank(value))
    if missing:
        return MappingResult(
            clip_id,
            None,
            ((ExclusionReason.DNA_INCOMPLETE, "absent or empty: " + ", ".join(missing)),),
        )

    tables: list[tuple[str, str, dict[str, Any]]] = [
        ("odd.weather", required["odd.weather"], cat.WEATHER),
        ("odd.lighting", required["odd.lighting"], cat.LIGHTING),
        ("topology.road_type", required["topology.road_type"], cat.ROAD_TYPE),
        ("topology.lane_event", required["topology.lane_event"], cat.LANE_EVENT),
        (
            "topology.intersection_type",
            required["topology.intersection_type"],
            cat.INTERSECTION_TYPE,
        ),
        ("planner_logic.ego_maneuver", required["planner_logic.ego_maneuver"], cat.EGO_MANEUVER),
    ]
    unknown = [f"{path}={value!r}" for path, value, table in tables if value not in table]
    if required["planner_logic.risk_level"] not in cat.RISK_LEVELS:
        unknown.append(f"planner_logic.risk_level={required['planner_logic.risk_level']!r}")
    if unknown:
        return MappingResult(
            clip_id,
            None,
            ((ExclusionReason.UNKNOWN_ENUM_VALUE, "; ".join(sorted(unknown))),),
        )

    weather_key = required["odd.weather"]
    lighting_key = required["odd.lighting"]
    road_key = required["topology.road_type"]
    lane_event_key = required["topology.lane_event"]
    intersection_key = required["topology.intersection_type"]
    ego_key = required["planner_logic.ego_maneuver"]

    road = cat.ROAD_TYPE[road_key]
    intersection = cat.INTERSECTION_TYPE[intersection_key]
    ego_map = cat.EGO_MANEUVER[ego_key]
    for path, value, reason in (
        ("topology.road_type", road_key, road.exclusion),
        ("topology.intersection_type", intersection_key, intersection.exclusion),
        ("planner_logic.ego_maneuver", ego_key, ego_map.exclusion),
    ):
        if reason is not None:
            exclusions.append((reason, f"{path}={value}"))
    if exclusions:
        return MappingResult(clip_id, None, tuple(exclusions))

    # --- environment -------------------------------------------------------------------
    weather_map = cat.WEATHER[weather_key]
    lighting_map = cat.LIGHTING[lighting_key]
    weather_params: dict[str, float] = {**weather_map.params, **lighting_map.params}
    for path, requested, mapping in (
        ("odd.weather", weather_key, weather_map),
        ("odd.lighting", lighting_key, lighting_map),
    ):
        if mapping.degradation is not None:
            degradations.append(Degradation(mapping.degradation, path, requested, mapping.applied))

    # --- road / junction ---------------------------------------------------------------
    towns = tuple(t for t in road.towns if t in intersection.towns)
    if not towns:
        # Every loadable town that has this road class signalizes (or lacks) the junction
        # form the DNA asked for. Keep the road environment, relax the junction.
        towns = road.towns
        degradations.append(
            Degradation(
                DegradationCode.INTERSECTION_SUBSTITUTED,
                "topology.intersection_type",
                intersection_key,
                f"nearest junction on a {road_key} road ({', '.join(towns)})",
            )
        )
    elif intersection.degradation is not None:
        degradations.append(
            Degradation(
                intersection.degradation,
                "topology.intersection_type",
                intersection_key,
                intersection.applied,
            )
        )
    if road.degradation is not None:
        degradations.append(
            Degradation(road.degradation, "topology.road_type", road_key, road.applied)
        )

    lane_event = cat.LANE_EVENT[lane_event_key]
    if lane_event.degradation is not None:
        degradations.append(
            Degradation(
                lane_event.degradation, "topology.lane_event", lane_event_key, lane_event.applied
            )
        )
    props = tuple(PropSpec(bp, n, placement) for bp, n, placement in lane_event.props)

    world = WorldSpec(
        weather=weather_params,
        road=RoadQuery(
            road_type=road_key,
            intersection_type=intersection_key,
            min_driving_lanes=road.min_driving_lanes,
            speed_kph_range=road.speed_kph_range,
            required_lane_types=road.required_lane_types,
            candidate_towns=towns,
        ),
        props=props,
    )

    # --- ego ---------------------------------------------------------------------------
    if ego_map.degradation is not None:
        degradations.append(
            Degradation(ego_map.degradation, "planner_logic.ego_maneuver", ego_key, ego_map.applied)
        )
    ego = EgoSpec(
        maneuver=ego_key,
        control_template=ego_map.template,
        target_speed_kph=round(road.target_speed_kph * ego_map.speed_factor, 1),
    )

    # --- actors ------------------------------------------------------------------------
    actors, actor_degradations, actor_anomalies = _map_actors(dna.get("actor_dynamics") or [])
    degradations.extend(actor_degradations)
    anomalies.extend(actor_anomalies)

    # --- safety event ------------------------------------------------------------------
    safety_raw = planner.get("safety_event") or {}
    event_type = safety_raw.get("event_type") or "none"
    if event_type not in cat.SAFETY_EVENT_TYPES:
        anomalies.append(f"planner_logic.safety_event.event_type={event_type!r} outside schema")
        event_type = "none"
    collision_type = safety_raw.get("collision_type")
    if collision_type is not None and collision_type not in cat.COLLISION_TYPES:
        anomalies.append(
            f"planner_logic.safety_event.collision_type={collision_type!r} outside schema"
        )
    severity = safety_raw.get("severity_estimate")
    if severity is not None and severity not in cat.SEVERITY_ESTIMATES:
        anomalies.append(
            f"planner_logic.safety_event.severity_estimate={severity!r} outside schema"
        )
    has_event = bool(safety_raw.get("has_event")) or event_type != "none"

    actors = _mark_event_actor(actors, has_event, planner.get("causal_trigger_actor_index"))
    if has_event and any(a.is_event_actor for a in actors):
        if planner.get("causal_trigger_actor_index") is None:
            degradations.append(
                Degradation(
                    DegradationCode.EVENT_ACTOR_INFERRED,
                    "planner_logic.causal_trigger_actor_index",
                    "absent",
                    "highest-priority actor state at the nearest distance bucket",
                )
            )
    if event_type in cat.COLLISION_STAGED_AS_NEAR_MISS:
        degradations.append(
            Degradation(
                DegradationCode.COLLISION_NOT_STAGED,
                "planner_logic.safety_event.event_type",
                event_type,
                "pre-impact geometry only; no impact is forced",
            )
        )

    # --- cameras -----------------------------------------------------------------------
    cam_attrs, cam_degradations = _camera_attributes(odd.get("sensor_fidelity") or [])
    degradations.extend(cam_degradations)
    cameras = (
        CameraSpec(view="ego", transform=EGO_CAM_TRANSFORM, attributes=cam_attrs),
        CameraSpec(view="chase", transform=CHASE_CAM_TRANSFORM, attributes=cam_attrs),
    )

    spec = SimSpec(
        clip_id=clip_id,
        dna_version=str(dna.get("dna_version", "")),
        duration_s=DEFAULT_SEGMENT_S,
        warmup_s=WARMUP_S,
        risk_level=required["planner_logic.risk_level"],
        world=world,
        ego=ego,
        actors=actors,
        cameras=cameras,
        safety_event=SafetyEventSpec(has_event, event_type, collision_type, severity),
        degradations=tuple(degradations),
    )
    return MappingResult(clip_id, spec, (), tuple(anomalies))


def _map_actors(
    raw_actors: list[dict[str, Any]],
) -> tuple[tuple[ActorSpec, ...], list[Degradation], list[str]]:
    """Map ``actor_dynamics`` entries, dropping only what has no representation at all."""
    specs: list[ActorSpec] = []
    degradations: list[Degradation] = []
    anomalies: list[str] = []

    for index, raw in enumerate(raw_actors[:MAX_ACTORS]):
        actor_class = raw.get("actor_class")
        state = raw.get("state")
        bucket = raw.get("distance_bucket")
        path = f"actor_dynamics[{index}]"

        if actor_class not in cat.ACTOR_CLASS:
            anomalies.append(f"{path}.actor_class={actor_class!r} outside schema")
            degradations.append(
                Degradation(
                    DegradationCode.ACTOR_DROPPED,
                    f"{path}.actor_class",
                    str(actor_class),
                    "omitted — unrecognized actor class",
                )
            )
            continue
        actor_map = cat.ACTOR_CLASS[actor_class]
        if actor_map.degradation is DegradationCode.ACTOR_DROPPED:
            degradations.append(
                Degradation(
                    DegradationCode.ACTOR_DROPPED,
                    f"{path}.actor_class",
                    actor_class,
                    actor_map.applied,
                )
            )
            continue
        if actor_map.degradation is not None:
            degradations.append(
                Degradation(
                    actor_map.degradation, f"{path}.actor_class", actor_class, actor_map.applied
                )
            )

        if state not in cat.ACTOR_STATE:
            anomalies.append(f"{path}.state={state!r} outside schema")
            state = "static"
        state_map = cat.ACTOR_STATE[state]
        if state_map.degradation is not None:
            degradations.append(
                Degradation(state_map.degradation, f"{path}.state", state, state_map.applied)
            )

        if bucket not in cat.DISTANCE_BUCKETS:
            anomalies.append(f"{path}.distance_bucket={bucket!r} outside schema")
            bucket = "mid"

        specs.append(
            ActorSpec(
                index=index,
                actor_class=actor_class,
                blueprint_filter=actor_map.blueprint_filter,
                state=state,
                maneuver_template=state_map.template,
                distance_bucket=bucket,
                distance_m=cat.DISTANCE_M[bucket],
                control_mode=state_map.control_mode,
            )
        )
    return tuple(specs), degradations, anomalies


def _mark_event_actor(
    actors: tuple[ActorSpec, ...], has_event: bool, declared_index: Any
) -> tuple[ActorSpec, ...]:
    """Promote the one actor the safety event is about to ``ControlMode.EVENT``.

    ``causal_trigger_actor_index`` is honored when present; Scout never emits it, so the
    fallback ranks actors by how strongly their state implies causation, then by
    proximity. Actors that are not promoted keep the control mode their state implies —
    demoting them would drop behavior the DNA asserts.
    """
    if not has_event or not actors:
        return actors

    chosen: ActorSpec | None = None
    if isinstance(declared_index, int):
        chosen = next((a for a in actors if a.index == declared_index), None)
    if chosen is None:

        def rank(actor: ActorSpec) -> tuple[int, float]:
            try:
                priority = cat.EVENT_ACTOR_STATE_PRIORITY.index(actor.state)
            except ValueError:
                priority = len(cat.EVENT_ACTOR_STATE_PRIORITY)
            return (priority, actor.distance_m)

        chosen = min(actors, key=rank)

    return tuple(
        replace(a, control_mode=ControlMode.EVENT) if a.index == chosen.index else a for a in actors
    )


def _camera_attributes(
    sensor_fidelity: list[str],
) -> tuple[dict[str, str], list[Degradation]]:
    """Fold ``sensor_fidelity[]`` into RGB-camera post-process attributes."""
    attributes: dict[str, str] = {}
    degradations: list[Degradation] = []
    for artifact in sensor_fidelity:
        mapping = cat.SENSOR_FIDELITY.get(artifact)
        if mapping is None:
            continue
        attributes.update(mapping.attributes)
        if mapping.degradation is not None:
            degradations.append(
                Degradation(mapping.degradation, "odd.sensor_fidelity[]", artifact, mapping.applied)
            )
    return attributes, degradations
