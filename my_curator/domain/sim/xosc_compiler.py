"""``SimSpec`` + a resolved road, compiled into an OpenSCENARIO 1.0 document.

The output is a standard scenario file, not a CARLA script: it names entities, states the
environment, places everything on a concrete road and declares what each participant does
and when. That makes it readable by any OpenSCENARIO 1.0 tool, which is the point — it is
how a reconstruction claim can be checked by something other than the code that made it.

Determinism is a requirement, so the document carries no wall-clock timestamp and every
float is formatted at fixed precision: recompiling unchanged input reproduces the same
bytes. ``FileHeader/@date`` is therefore a constant, not the time of the run.

Stdlib only — ``xml.etree.ElementTree`` builds the tree; validation and writing are the
adapter's job, since the XSD validator is an external dependency ``domain/`` may not take.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from my_curator.domain.sim.catalog import ACTOR_CLASS
from my_curator.domain.sim.road_index import RoadSelection
from my_curator.domain.sim.spec import ActorSpec, ControlMode, SimSpec
from my_curator.domain.sim.templates import (
    TemplateContext,
    build_events,
    el,
    speed_action,
    time_trigger,
)

AUTHOR = "My-Curator real-to-sim"

#: Fixed so recompiling unchanged input is byte-identical. The scenario's provenance is
#: the clip id it carries, not the moment the file happened to be written.
HEADER_DATE = "2026-01-01T00:00:00"

EGO_NAME = "ego"
EGO_BLUEPRINT = "vehicle.tesla.model3"

KPH_TO_MPS = 1.0 / 3.6

#: OpenSCENARIO requires full vehicle dynamics; CARLA takes them from the blueprint, so
#: these are permissive placeholders that never bind before the blueprint's own limits.
_PERFORMANCE = {"maxSpeed": 69.4, "maxAcceleration": 8.0, "maxDeceleration": 10.0}
_VEHICLE_BOX = {"width": 2.0, "length": 4.6, "height": 1.6}
_WALKER_BOX = {"width": 0.6, "length": 0.6, "height": 1.8}
_AXLE = {
    "maxSteering": 0.5,
    "positionX": 1.4,
    "positionZ": 0.3,
    "trackWidth": 1.6,
    "wheelDiameter": 0.6,
}

#: actor_class -> OpenSCENARIO vehicleCategory. Classes staged as walkers or as static
#: props are absent and handled separately.
_VEHICLE_CATEGORY: dict[str, str] = {
    "cyclist": "bicycle",
    "e_bike_rider": "bicycle",
    "motorcyclist": "motorbike",
    "standup_scooter_rider": "motorbike",
    "delivery_motorcycle": "motorbike",
    "vehicle_car": "car",
    "vehicle_van": "van",
    "vehicle_truck": "truck",
    "vehicle_bus": "bus",
    "vehicle_emergency": "car",
    "vehicle_construction": "truck",
}

_WALKER_CLASSES = frozenset({"pedestrian", "wheelchair_user"})


def _is_walker(actor: ActorSpec) -> bool:
    return actor.actor_class in _WALKER_CLASSES or actor.blueprint_filter.startswith("walker.")


def _is_prop(actor: ActorSpec) -> bool:
    mapping = ACTOR_CLASS.get(actor.actor_class)
    return bool(getattr(mapping, "is_static_prop", False)) or actor.blueprint_filter.startswith(
        "static."
    )


# --- environment -------------------------------------------------------------------


def _cloud_state(weather: dict[str, float]) -> str:
    if weather.get("precipitation", 0.0) > 0.0:
        return "rainy"
    cloudiness = weather.get("cloudiness", 0.0)
    if cloudiness < 10.0:
        return "free"
    return "cloudy" if cloudiness < 60.0 else "overcast"


def _visual_range_m(weather: dict[str, float]) -> float:
    """Fog density as a visual range; OpenSCENARIO states visibility, CARLA states density."""
    density = max(0.0, min(100.0, weather.get("fog_density", 0.0)))
    return round(max(20.0, 2000.0 * (1.0 - density / 100.0)), 1)


def _environment(spec: SimSpec) -> ET.Element:
    weather = spec.world.weather
    action = el("GlobalAction")
    env_action = el("EnvironmentAction", action)
    env = el("Environment", env_action, name="reconstructed_odd")
    el("TimeOfDay", env, animation=False, dateTime=HEADER_DATE)
    weather_el = el("Weather", env, cloudState=_cloud_state(weather))
    el(
        "Sun",
        weather_el,
        intensity=max(0.0, weather.get("sun_altitude_angle", 45.0)) / 90.0,
        azimuth=weather.get("sun_azimuth_angle", 0.0),
        elevation=weather.get("sun_altitude_angle", 45.0),
    )
    el("Fog", weather_el, visualRange=_visual_range_m(weather))
    precipitation = weather.get("precipitation", 0.0)
    el(
        "Precipitation",
        weather_el,
        precipitationType="rain" if precipitation > 0.0 else "dry",
        intensity=round(precipitation / 100.0, 3),
    )
    el(
        "RoadCondition",
        env,
        frictionScaleFactor=round(1.0 - 0.4 * weather.get("wetness", 0.0) / 100.0, 3),
    )
    return action


# --- entities ----------------------------------------------------------------------


def _bounding_box(parent: ET.Element, dims: dict[str, float]) -> None:
    box = el("BoundingBox", parent)
    el("Center", box, x=1.4, y=0.0, z=0.9)
    el("Dimensions", box, **dims)


def _properties(parent: ET.Element, values: dict[str, str]) -> None:
    props = el("Properties", parent)
    for name, value in values.items():
        el("Property", props, name=name, value=value)


def _vehicle(parent: ET.Element, name: str, blueprint: str, category: str, role: str) -> None:
    vehicle = el("Vehicle", parent, name=blueprint, vehicleCategory=category)
    _bounding_box(vehicle, _VEHICLE_BOX)
    el("Performance", vehicle, **_PERFORMANCE)
    axles = el("Axles", vehicle)
    el("FrontAxle", axles, **_AXLE)
    el("RearAxle", axles, **{**_AXLE, "maxSteering": 0.0, "positionX": 0.0})
    _properties(vehicle, {"type": role, "blueprint_filter": blueprint, "entity": name})


def _pedestrian(parent: ET.Element, name: str, blueprint: str) -> None:
    walker = el(
        "Pedestrian",
        parent,
        name=blueprint,
        model=blueprint,
        mass=80.0,
        pedestrianCategory="pedestrian",
    )
    _bounding_box(walker, _WALKER_BOX)
    _properties(walker, {"type": "walker", "blueprint_filter": blueprint, "entity": name})


def _entities(spec: SimSpec, actors: list[tuple[str, ActorSpec]]) -> ET.Element:
    entities = el("Entities")
    ego_object = el("ScenarioObject", entities, name=EGO_NAME)
    _vehicle(ego_object, EGO_NAME, EGO_BLUEPRINT, "car", "ego_vehicle")

    for name, actor in actors:
        obj = el("ScenarioObject", entities, name=name)
        if _is_walker(actor):
            _pedestrian(obj, name, actor.blueprint_filter)
        else:
            category = _VEHICLE_CATEGORY.get(actor.actor_class, "car")
            _vehicle(obj, name, actor.blueprint_filter, category, "simulation")
    return entities


# --- placement ---------------------------------------------------------------------


def _lane_position(
    parent: ET.Element, road_id: int, lane_id: int, s: float, offset: float = 0.0
) -> None:
    position = el("Position", parent)
    el("LanePosition", position, roadId=road_id, laneId=lane_id, s=round(s, 3), offset=offset)


def _teleport(road_id: int, lane_id: int, s: float, offset: float = 0.0) -> ET.Element:
    action = el("PrivateAction")
    teleport = el("TeleportAction", action)
    _lane_position(teleport, road_id, lane_id, s, offset)
    return action


def _actor_placement(actor: ActorSpec, road: RoadSelection, ego_s: float) -> tuple[int, float]:
    """Where an actor starts, relative to ego along the same road.

    Oncoming and cut-in traffic sit ahead of ego; a follower sits behind it. The lane is
    the neighbouring one where the maneuver needs somewhere to come from.
    """
    lane = road.candidate.lane_id
    if actor.state in {"cutin", "cutout", "oncoming"}:
        lane = lane + 1 if lane < 0 else lane - 1
    if actor.state == "tailing":
        return lane, max(0.0, ego_s - actor.distance_m)
    return lane, ego_s + actor.distance_m


# --- storyboard --------------------------------------------------------------------


def _init(spec: SimSpec, road: RoadSelection, actors: list[tuple[str, ActorSpec]]) -> ET.Element:
    init = el("Init")
    actions = el("Actions", init)
    actions.append(_environment(spec))

    candidate = road.candidate
    ego_s = candidate.entry_s
    private = el("Private", actions, entityRef=EGO_NAME)
    private.append(_teleport(candidate.road_id, candidate.lane_id, ego_s))
    private.append(_speed_init(spec.ego.target_speed_kph * KPH_TO_MPS))

    for name, actor in actors:
        lane, s = _actor_placement(actor, road, ego_s)
        actor_private = el("Private", actions, entityRef=name)
        actor_private.append(_teleport(candidate.road_id, lane, s))
        actor_private.append(_speed_init(_initial_speed_mps(actor, spec)))
    return init


def _speed_init(value_mps: float) -> ET.Element:
    return speed_action(value_mps, shape="step", over_s=0.0)


def _initial_speed_mps(actor: ActorSpec, spec: SimSpec) -> float:
    if actor.state in {"stopped", "parked", "static"}:
        return 0.0
    if _is_walker(actor):
        return 0.0
    return spec.ego.target_speed_kph * KPH_TO_MPS


def _maneuver_group(name: str, entity: str, events: list[ET.Element]) -> ET.Element:
    group = el("ManeuverGroup", maximumExecutionCount=1, name=f"{name}_group")
    actors_el = el("Actors", group, selectTriggeringEntities=False)
    el("EntityRef", actors_el, entityRef=entity)
    maneuver = el("Maneuver", group, name=f"{name}_maneuver")
    for event_el in events:
        maneuver.append(event_el)
    return group


def _story(spec: SimSpec, actors: list[tuple[str, ActorSpec]]) -> ET.Element:
    story = el("Story", name="reconstructed_segment")
    act = el("Act", story, name="main")

    ego_ctx = TemplateContext(
        entity=EGO_NAME,
        ego=EGO_NAME,
        target_speed_mps=spec.ego.target_speed_kph * KPH_TO_MPS,
        trigger_distance_m=0.0,
        event_timed=False,
    )
    ego_events = build_events(spec.ego.control_template, ego_ctx)
    if ego_events:
        act.append(_maneuver_group(EGO_NAME, EGO_NAME, ego_events))

    for name, actor in actors:
        if actor.control_mode is ControlMode.AMBIENT:
            # Genuine background traffic: the traffic manager drives it, so the scenario
            # states only that it exists, not what it does.
            continue
        ctx = TemplateContext(
            entity=name,
            ego=EGO_NAME,
            target_speed_mps=spec.ego.target_speed_kph * KPH_TO_MPS,
            trigger_distance_m=actor.distance_m,
            event_timed=actor.is_event_actor,
        )
        events = build_events(actor.maneuver_template, ctx)
        if events:
            act.append(_maneuver_group(name, name, events))

    act.append(time_trigger(0.0, name="act_start"))
    return story


def _stop_trigger(spec: SimSpec) -> ET.Element:
    trigger = time_trigger(spec.warmup_s + spec.duration_s, name="segment_elapsed")
    trigger.tag = "StopTrigger"
    return trigger


def _actor_names(spec: SimSpec) -> list[tuple[str, ActorSpec]]:
    """Name and order the actors that become scenario entities.

    Static props are dressing rather than participants: they are placed by the render
    stage from ``world.props`` and are not entities in the scenario.
    """
    named = []
    for actor in spec.actors:
        if _is_prop(actor) or not actor.blueprint_filter:
            continue
        named.append((f"adversary_{actor.index}", actor))
    return named


def compile_scenario(spec: SimSpec, road: RoadSelection) -> ET.Element:
    """Build the OpenSCENARIO document for one segment staged on one resolved road."""
    actors = _actor_names(spec)
    root = el("OpenSCENARIO")
    el(
        "FileHeader",
        root,
        revMajor=1,
        revMinor=0,
        date=HEADER_DATE,
        description=f"My-Curator reconstruction of segment {spec.clip_id} "
        f"(risk={spec.risk_level}, dna={spec.dna_version})",
        author=AUTHOR,
    )
    el("CatalogLocations", root)
    road_network = el("RoadNetwork", root)
    el("LogicFile", road_network, filepath=road.candidate.town)
    root.append(_entities(spec, actors))

    storyboard = el("Storyboard", root)
    storyboard.append(_init(spec, road, actors))
    storyboard.append(_story(spec, actors))
    storyboard.append(_stop_trigger(spec))
    return root
