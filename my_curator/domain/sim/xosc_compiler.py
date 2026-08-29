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
from my_curator.domain.sim.road_index import RoadCandidate, RoadSelection
from my_curator.domain.sim.spec import ActorSpec, ControlMode, SimSpec
from my_curator.domain.sim.templates import (
    TemplateContext,
    build_events,
    distance_trigger,
    el,
    event,
    lane_change_action,
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


def _vehicle(
    parent: ET.Element, name: str, blueprint: str, category: str, role: str, state: str = ""
) -> None:
    vehicle = el("Vehicle", parent, name=blueprint, vehicleCategory=category)
    _bounding_box(vehicle, _VEHICLE_BOX)
    el("Performance", vehicle, **_PERFORMANCE)
    axles = el("Axles", vehicle)
    el("FrontAxle", axles, **_AXLE)
    el("RearAxle", axles, **{**_AXLE, "maxSteering": 0.0, "positionX": 0.0})
    _properties(
        vehicle, {"type": role, "blueprint_filter": blueprint, "entity": name, "state": state}
    )


def _pedestrian(parent: ET.Element, name: str, blueprint: str, state: str = "") -> None:
    walker = el(
        "Pedestrian",
        parent,
        name=blueprint,
        model=blueprint,
        mass=80.0,
        pedestrianCategory="pedestrian",
    )
    _bounding_box(walker, _WALKER_BOX)
    # The state travels with the entity because the executor cannot infer it: a pedestrian
    # who crosses the road and one who walks along it differ only by their DNA state.
    _properties(
        walker, {"type": "walker", "blueprint_filter": blueprint, "entity": name, "state": state}
    )


def _entities(spec: SimSpec, actors: list[tuple[str, ActorSpec]]) -> ET.Element:
    entities = el("Entities")
    ego_object = el("ScenarioObject", entities, name=EGO_NAME)
    _vehicle(ego_object, EGO_NAME, EGO_BLUEPRINT, "car", "ego_vehicle")

    for name, actor in actors:
        obj = el("ScenarioObject", entities, name=name)
        if _is_walker(actor):
            _pedestrian(obj, name, actor.blueprint_filter, actor.state)
        else:
            category = _VEHICLE_CATEGORY.get(actor.actor_class, "car")
            _vehicle(obj, name, actor.blueprint_filter, category, "simulation", actor.state)
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


#: States staged in a lane other than ego's, because the maneuver needs somewhere to come
#: from. ``cutout`` is deliberately absent: leaving ego's lane requires starting in it.
_NEIGHBOURING_LANE_STATES = frozenset({"cutin", "oncoming"})

#: States in which the actor enters ego's path from outside it. A walker crosses the lane
#: on cue; a vehicle cannot cross sideways, so it waits in the neighbouring lane and pulls
#: into ego's instead.
_CROSSING_STATES = frozenset({"crossing", "hesitating", "jaywalking", "emerging"})

#: Minimum spacing between two placements on one lane. Identical positions spawn one actor
#: on top of another, which then scatters under physics.
_PLACEMENT_SPACING_M = 7.0

#: Fastest speed a class is asked to hold; a cyclist cannot reach a car's target and falls
#: out of frame trying.
_CLASS_SPEED_CAP_MPS: dict[str, float] = {"cyclist": 7.0}


def _neighbour_lane(candidate: RoadCandidate) -> int | None:
    """A drivable lane beside ego's, travelling the same way — ``None`` if there is none.

    Lane 0 is the reference line and is never a driving lane, so the search steps inward
    first, then outward while the carriageway still has lanes. It never crosses the centre
    line: the opposing carriageway is oncoming traffic, not a neighbour.
    """
    lane = candidate.lane_id
    inward = lane + 1 if lane < 0 else lane - 1
    if inward != 0:
        return inward
    outward = lane - 1 if lane < 0 else lane + 1
    if abs(outward) <= candidate.driving_lanes:
        return outward
    return None


def _ego_at_record_s(ego_s: float, candidate: RoadCandidate, spec: SimSpec) -> float:
    """Where ego will be when the cameras start.

    Ego drives its Init speed through the whole unrecorded warm-up while adversaries are
    held still, so the DNA's distance — the gap at the first recorded frame — is measured
    from here rather than from ego's staged ``s``.
    """
    travel = spec.ego.target_speed_kph * KPH_TO_MPS * spec.warmup_s
    return ego_s + candidate.travel_direction * travel


def _actor_placement(
    actor: ActorSpec, road: RoadSelection, ego_s: float, spec: SimSpec
) -> tuple[int, float]:
    """Where an actor starts, so that the DNA's gap holds at the first recorded frame.

    Lane first: oncoming traffic takes the innermost opposing lane, cut-ins and
    lane-entering vehicles a same-direction neighbour (ego's own lane when the carriageway
    has none), everything else ego's lane. Then the gap, forward from
    :func:`_ego_at_record_s` — backward only for a follower — clamped to leave a vehicle
    length at either end of the section, whose edge is often unspawnable.
    """
    candidate = road.candidate
    forward = candidate.travel_direction
    lane = candidate.lane_id
    if actor.state == "oncoming":
        # Innermost opposing lane, not ego's mirror: on a multi-lane road the mirror lane
        # is a whole carriageway away and the pass never reaches the camera.
        lane = 1 if lane < 0 else -1
    elif actor.state in _NEIGHBOURING_LANE_STATES or (
        actor.state in _CROSSING_STATES and not _is_walker(actor)
    ):
        # With no same-direction neighbour the actor stages in ego's lane as slower traffic
        # ahead — less than the DNA states, but not the head-on pass it never described.
        # The executor drops the lane change that is then meaningless.
        lane = _neighbour_lane(candidate) or candidate.lane_id

    gap = max(actor.distance_m, 0.0)
    if lane * candidate.lane_id < 0:
        # Head-on pairs close at their combined speed, so a gap set for the first frame is
        # spent within a few of them; staged to meet mid-segment the approach stays on film.
        # The DNA's distance is a bucket for the moment of interest, not for frame zero.
        gap += spec.ego.target_speed_kph * KPH_TO_MPS * spec.duration_s
    offset = -gap if actor.state == "tailing" else gap

    s = _ego_at_record_s(ego_s, candidate, spec) + forward * offset
    low = candidate.lane_section_s + _PLACEMENT_SPACING_M
    high = max(low, candidate.lane_section_end_s - _PLACEMENT_SPACING_M)
    return lane, round(min(max(s, low), high), 3)


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

    taken: list[tuple[int, float]] = [(candidate.lane_id, ego_s)]
    for name, actor in actors:
        lane, s = _actor_placement(actor, road, ego_s, spec)
        s = _deconflict(lane, s, taken, candidate)
        taken.append((lane, s))
        actor_private = el("Private", actions, entityRef=name)
        actor_private.append(_teleport(candidate.road_id, lane, s))
        actor_private.append(_speed_init(_initial_speed_mps(actor, spec)))
    return init


def _deconflict(
    lane: int, s: float, taken: list[tuple[int, float]], candidate: RoadCandidate
) -> float:
    """Push a placement along its lane until it overlaps nothing already placed there."""
    step = candidate.travel_direction * _PLACEMENT_SPACING_M
    while any(
        other_lane == lane and abs(other_s - s) < _PLACEMENT_SPACING_M
        for other_lane, other_s in taken
    ):
        s = min(max(s + step, candidate.lane_section_s), candidate.lane_section_end_s)
        if s in (candidate.lane_section_s, candidate.lane_section_end_s):
            break  # out of road — an edge overlap beats an infinite loop
    return round(s, 3)


def _speed_init(value_mps: float) -> ET.Element:
    return speed_action(value_mps, shape="step", over_s=0.0)


def _initial_speed_mps(actor: ActorSpec, spec: SimSpec) -> float:
    if actor.state in {"stopped", "parked", "static"}:
        return 0.0
    if actor.state in _CROSSING_STATES:
        # Waits for its cue; driven through the warm-up it leaves before recording starts.
        return 0.0
    if actor.state == "cutin":
        # Rolls alongside rather than waiting — from rest a fast ego overtakes it before
        # the cue and the cut-in happens behind the camera.
        return min(
            spec.ego.target_speed_kph * KPH_TO_MPS,
            _CLASS_SPEED_CAP_MPS.get(actor.actor_class, float("inf")),
        )
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

    # Ego reacts to the event actor at the DNA distance rather than on a fixed clock: the
    # driver in the source clip reacted to the actor. Oncoming traffic is excluded — it
    # passes in the opposing lane, so the longitudinal gap is never reached and the
    # maneuver would never fire.
    event_actor = next(
        (
            (name, actor)
            for name, actor in actors
            if actor.is_event_actor and actor.state != "oncoming"
        ),
        None,
    )
    ego_ctx = TemplateContext(
        entity=event_actor[0] if event_actor else EGO_NAME,
        ego=EGO_NAME,
        target_speed_mps=spec.ego.target_speed_kph * KPH_TO_MPS,
        trigger_distance_m=event_actor[1].distance_m if event_actor else 0.0,
        event_timed=event_actor is not None,
        warmup_s=spec.warmup_s,
        duration_s=spec.duration_s,
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
            warmup_s=spec.warmup_s,
            duration_s=spec.duration_s,
        )
        events = build_events(actor.maneuver_template, ctx)
        if actor.state in _CROSSING_STATES and not _is_walker(actor):
            # The vehicle waits in the neighbouring lane (see the placement); entering
            # ego's path is a lane change, which the crossing templates — written for
            # walkers, who cross by walking — do not state.
            trigger = distance_trigger(ctx) if ctx.event_timed else time_trigger(ctx.cue_s())
            events.append(
                event(
                    f"{name}_pulls_into_lane",
                    [lane_change_action(-1, entity=name, over_s=2.0)],
                    trigger,
                )
            )
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
