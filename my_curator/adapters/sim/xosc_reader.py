"""Read a compiled OpenSCENARIO document back into an executable program.

The inverse of ``domain.sim.xosc_compiler``, and deliberately no more general than it: the
compiler emits four action types and two trigger types, and this reader implements exactly
those. Anything else raises :class:`UnsupportedScenarioError` rather than being skipped, so
a compiler that grows a new construct fails loudly here instead of rendering a scenario
that quietly omits it.

Stdlib only, so the render stage can read a scenario without a simulator attached.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from my_curator.domain.sim.program import (
    Action,
    EntityDef,
    Environment,
    InitState,
    LaneChangeAction,
    LaneOffsetAction,
    LanePlacement,
    ManeuverGroup,
    ScenarioEvent,
    ScenarioProgram,
    SpeedAction,
    TeleportAction,
    Trigger,
    UnsupportedScenarioError,
)

#: The compiler states cloud cover as an OpenSCENARIO bucket, which cannot carry the
#: original percentage. These are the representative values it is read back as.
_CLOUD_STATE_PCT = {"free": 5.0, "cloudy": 35.0, "overcast": 80.0, "rainy": 80.0}

_CLIP_ID = re.compile(r"segment (\S+) \(risk=")


def _require(parent: ET.Element, path: str, what: str) -> ET.Element:
    found = parent.find(path)
    if found is None:
        raise UnsupportedScenarioError(f"{what} is missing ({path})")
    return found


def _float(element: ET.Element, attr: str, default: float | None = None) -> float:
    raw = element.get(attr)
    if raw is None:
        if default is None:
            raise UnsupportedScenarioError(f"<{element.tag}> has no {attr}")
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise UnsupportedScenarioError(f"<{element.tag}> {attr}={raw!r} is not a number") from exc


def _properties(obj: ET.Element) -> dict[str, str]:
    return {
        prop.get("name", ""): prop.get("value", "")
        for prop in obj.iter("Property")
        if prop.get("name")
    }


def _entities(root: ET.Element) -> tuple[EntityDef, ...]:
    entities = []
    for obj in root.findall("Entities/ScenarioObject"):
        name = obj.get("name", "")
        props = _properties(obj)
        role = props.get("type", "")
        if not name or not role:
            raise UnsupportedScenarioError(f"entity {name!r} carries no role property")
        entities.append(
            EntityDef(
                name=name,
                blueprint_filter=props.get("blueprint_filter", ""),
                role=role,
                is_walker=obj.find("Pedestrian") is not None,
                state=props.get("state", ""),
            )
        )
    if not entities:
        raise UnsupportedScenarioError("scenario declares no entities")
    return tuple(entities)


def _environment(root: ET.Element) -> Environment:
    env = _require(
        root,
        "Storyboard/Init/Actions/GlobalAction/EnvironmentAction/Environment",
        "environment",
    )
    weather = _require(env, "Weather", "weather")
    sun = _require(weather, "Sun", "sun")
    precipitation = _require(weather, "Precipitation", "precipitation")
    cloud_state = weather.get("cloudState", "free")
    if cloud_state not in _CLOUD_STATE_PCT:
        raise UnsupportedScenarioError(f"unknown cloudState {cloud_state!r}")
    road_condition = env.find("RoadCondition")
    return Environment(
        cloudiness=_CLOUD_STATE_PCT[cloud_state],
        precipitation=round(_float(precipitation, "intensity", 0.0) * 100.0, 3),
        visual_range_m=_float(_require(weather, "Fog", "fog"), "visualRange", 2000.0),
        sun_elevation=_float(sun, "elevation", 45.0),
        sun_azimuth=_float(sun, "azimuth", 0.0),
        friction=_float(road_condition, "frictionScaleFactor", 1.0) if road_condition else 1.0,
    )


def _lane_placement(position: ET.Element) -> LanePlacement:
    lane = _require(position, "LanePosition", "lane position")
    return LanePlacement(
        road_id=int(_float(lane, "roadId")),
        lane_id=int(_float(lane, "laneId")),
        s=_float(lane, "s"),
        offset=_float(lane, "offset", 0.0),
    )


def _speed_action(node: ET.Element) -> SpeedAction:
    dynamics = _require(node, "SpeedActionDynamics", "speed dynamics")
    target = _require(node, "SpeedActionTarget/AbsoluteTargetSpeed", "absolute target speed")
    return SpeedAction(
        target_mps=_float(target, "value"),
        shape=dynamics.get("dynamicsShape", "linear"),
        over_s=_float(dynamics, "value", 0.0),
    )


def _lane_change_action(node: ET.Element) -> LaneChangeAction:
    dynamics = _require(node, "LaneChangeActionDynamics", "lane-change dynamics")
    target = _require(node, "LaneChangeTarget/RelativeTargetLane", "relative target lane")
    return LaneChangeAction(
        relative_lane=int(_float(target, "value")),
        entity_ref=target.get("entityRef", ""),
        over_s=_float(dynamics, "value", 2.0),
    )


def _lane_offset_action(node: ET.Element) -> LaneOffsetAction:
    target = _require(node, "LaneOffsetTarget/AbsoluteTargetLaneOffset", "target lane offset")
    return LaneOffsetAction(offset_m=_float(target, "value"))


def _private_action(action: ET.Element) -> Action:
    speed = action.find("LongitudinalAction/SpeedAction")
    if speed is not None:
        return _speed_action(speed)
    lane_change = action.find("LateralAction/LaneChangeAction")
    if lane_change is not None:
        return _lane_change_action(lane_change)
    lane_offset = action.find("LateralAction/LaneOffsetAction")
    if lane_offset is not None:
        return _lane_offset_action(lane_offset)
    teleport = action.find("TeleportAction")
    if teleport is not None:
        return TeleportAction(_lane_placement(_require(teleport, "Position", "position")))
    children = [child.tag for child in action]
    raise UnsupportedScenarioError(f"unsupported PrivateAction: {children}")


def _trigger(node: ET.Element, what: str) -> Trigger:
    condition = _require(node, "ConditionGroup/Condition", f"{what} condition")
    simulation_time = condition.find("ByValueCondition/SimulationTimeCondition")
    if simulation_time is not None:
        return Trigger(kind="time", value=_float(simulation_time, "value"))
    distance = condition.find("ByEntityCondition/EntityCondition/RelativeDistanceCondition")
    if distance is not None:
        return Trigger(
            kind="distance",
            value=_float(distance, "value"),
            entity_ref=distance.get("entityRef", ""),
        )
    raise UnsupportedScenarioError(f"unsupported {what}: {condition.get('name')!r}")


def _init(root: ET.Element) -> tuple[InitState, ...]:
    states = []
    for private in root.findall("Storyboard/Init/Actions/Private"):
        entity = private.get("entityRef", "")
        placement: LanePlacement | None = None
        speed = 0.0
        for action in private.findall("PrivateAction"):
            parsed = _private_action(action)
            if isinstance(parsed, TeleportAction):
                placement = parsed.placement
            elif isinstance(parsed, SpeedAction):
                speed = parsed.target_mps
        if placement is None:
            raise UnsupportedScenarioError(f"entity {entity!r} is never placed")
        states.append(InitState(entity=entity, placement=placement, speed_mps=speed))
    return tuple(states)


def _events(maneuver: ET.Element) -> tuple[ScenarioEvent, ...]:
    events = []
    for node in maneuver.findall("Event"):
        actions = tuple(
            _private_action(_require(wrapper, "PrivateAction", "action"))
            for wrapper in node.findall("Action")
        )
        events.append(
            ScenarioEvent(
                name=node.get("name", ""),
                actions=actions,
                trigger=_trigger(_require(node, "StartTrigger", "event trigger"), "start trigger"),
            )
        )
    return tuple(events)


def _groups(root: ET.Element) -> tuple[ManeuverGroup, ...]:
    groups = []
    for group in root.findall("Storyboard/Story/Act/ManeuverGroup"):
        entity_ref = _require(group, "Actors/EntityRef", "maneuver group actor")
        maneuver = _require(group, "Maneuver", "maneuver")
        groups.append(
            ManeuverGroup(entity=entity_ref.get("entityRef", ""), events=_events(maneuver))
        )
    return tuple(groups)


def _stop_time_s(root: ET.Element) -> float:
    stop = _require(root, "Storyboard/StopTrigger", "stop trigger")
    return _trigger(stop, "stop trigger").value


def parse_program(root: ET.Element) -> ScenarioProgram:
    """Turn a compiled scenario document into an executable program."""
    if root.tag != "OpenSCENARIO":
        raise UnsupportedScenarioError(f"root element is <{root.tag}>, not <OpenSCENARIO>")
    header = _require(root, "FileHeader", "file header")
    description = header.get("description", "")
    match = _CLIP_ID.search(description)
    program = ScenarioProgram(
        clip_id=match.group(1) if match else "",
        town=_require(root, "RoadNetwork/LogicFile", "road network").get("filepath", ""),
        description=description,
        environment=_environment(root),
        entities=_entities(root),
        init=_init(root),
        groups=_groups(root),
        stop_time_s=_stop_time_s(root),
    )
    known = {entity.name for entity in program.entities}
    for group in program.groups:
        if group.entity not in known:
            raise UnsupportedScenarioError(f"maneuver group drives unknown entity {group.entity!r}")
    return program


def read_program(path: str | Path) -> ScenarioProgram:
    return parse_program(ET.parse(str(path)).getroot())
