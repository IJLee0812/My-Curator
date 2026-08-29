"""The 24 maneuver templates, compiled into OpenSCENARIO 1.0 story structure.

The mapper carries maneuvers as names — ``vehicle_lane_change_into_ego_lane``,
``ego_emergency_brake`` — decided by the DNA but not yet expressed. This module turns each
name into the ``Event`` elements that state it: an action, and a trigger saying when.

Depth is tiered, per the phase decision. The four actor states that dominate the corpus —
``cutin``, ``crossing``, ``stopped``, ``cutout`` — are triggered *relative to ego*, so the
interaction happens at the distance the DNA recorded. Everything else fires on a fixed
simulation-time cue, which reproduces the stated behavior without choreographing it.

Only structure is emitted here. Executing it — route planning, speed tracking, the lateral
controller — belongs to the render stage, which reads these actions back.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

# typing, not collections.abc: TemplateBuilder below is evaluated at runtime, and neither
# builtin nor abc generics are subscriptable on the py3.7 the simulator image runs.
from typing import Callable, List  # noqa: UP035

from my_curator.domain.sim.spec import DEFAULT_SEGMENT_S, WARMUP_S

#: When a scripted cue fires, as a point in the *recorded* segment: this many seconds into
#: a nominal DEFAULT_SEGMENT_S recording, scaled to the segment's real length. Simulation
#: time 0 is the start of the unrecorded warm-up, so :meth:`TemplateContext.cue_s` turns
#: this into the absolute time the document needs — a bare 2.0 s would fire before the
#: cameras start.
SCRIPTED_CUE_S = 2.0

#: Ego lane-change and offset directions. OpenSCENARIO counts lanes leftward from the
#: entity's own lane, matching OpenDRIVE's signed lane ids.
_LEFT = 1
_RIGHT = -1

NUDGE_OFFSET_M = 0.8
SWERVE_OFFSET_M = 1.6

WALK_SPEED_MPS = 1.4
HESITANT_WALK_SPEED_MPS = 0.7


def el(tag: str, parent: ET.Element | None = None, **attrs: object) -> ET.Element:
    """Create an element, dropping ``None`` attributes and stringifying the rest."""
    node = ET.Element(tag) if parent is None else ET.SubElement(parent, tag)
    for key, value in attrs.items():
        if value is not None:
            node.set(key, _fmt(value))
    return node


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Fixed precision keeps recompilation byte-identical across platforms.
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return str(value)


@dataclass(frozen=True)
class TemplateContext:
    """Everything a template needs to express one entity's behavior."""

    entity: str
    ego: str
    target_speed_mps: float
    #: Distance to ego at which an event-timed maneuver fires.
    trigger_distance_m: float
    #: Event-timed maneuvers trigger off ego; scripted ones off the clock.
    event_timed: bool
    #: The unrecorded settling time before the cameras start; simulation time 0 is warm-up.
    warmup_s: float = WARMUP_S
    #: Length of the recorded segment, which scripted cue times scale to.
    duration_s: float = DEFAULT_SEGMENT_S

    def cue_s(self, offset_s: float = 0.0) -> float:
        """Absolute simulation time of a scripted cue *offset_s* after the base cue.

        Cue times are stated relative to a nominal DEFAULT_SEGMENT_S recording and scaled
        to the real one, so a multi-phase maneuver keeps its proportions on a short
        segment instead of running off the end of it.
        """
        return self.warmup_s + (SCRIPTED_CUE_S + offset_s) * self.duration_s / DEFAULT_SEGMENT_S


# --- OpenSCENARIO vocabulary -------------------------------------------------------


def _transition(shape: str, dimension: str, value: float) -> ET.Element:
    return el("TransitionDynamics", dynamicsShape=shape, dynamicsDimension=dimension, value=value)


def speed_action(value_mps: float, *, shape: str = "linear", over_s: float = 1.0) -> ET.Element:
    action = el("PrivateAction")
    longitudinal = el("LongitudinalAction", action)
    speed = el("SpeedAction", longitudinal)
    dynamics = _transition(shape, "time", over_s)
    dynamics.tag = "SpeedActionDynamics"
    speed.append(dynamics)
    target = el("SpeedActionTarget", speed)
    el("AbsoluteTargetSpeed", target, value=value_mps)
    return action


def lane_change_action(relative_lane: int, *, entity: str, over_s: float = 2.0) -> ET.Element:
    action = el("PrivateAction")
    lateral = el("LateralAction", action)
    change = el("LaneChangeAction", lateral)
    dynamics = _transition("sinusoidal", "time", over_s)
    dynamics.tag = "LaneChangeActionDynamics"
    change.append(dynamics)
    target = el("LaneChangeTarget", change)
    el("RelativeTargetLane", target, entityRef=entity, value=relative_lane)
    return action


def lane_offset_action(offset_m: float) -> ET.Element:
    action = el("PrivateAction")
    lateral = el("LateralAction", action)
    offset = el("LaneOffsetAction", lateral, continuous=False)
    el("LaneOffsetActionDynamics", offset, dynamicsShape="sinusoidal", maxLateralAcc=2.0)
    target = el("LaneOffsetTarget", offset)
    el("AbsoluteTargetLaneOffset", target, value=offset_m)
    return action


def time_trigger(at_s: float, *, name: str = "cue") -> ET.Element:
    trigger = el("StartTrigger")
    group = el("ConditionGroup", trigger)
    condition = el("Condition", group, name=name, delay=0.0, conditionEdge="rising")
    by_value = el("ByValueCondition", condition)
    el("SimulationTimeCondition", by_value, value=at_s, rule="greaterThan")
    return trigger


def distance_trigger(ctx: TemplateContext, *, name: str = "ego_within_range") -> ET.Element:
    """Fire when ego closes to the distance the DNA recorded for this actor."""
    trigger = el("StartTrigger")
    group = el("ConditionGroup", trigger)
    condition = el("Condition", group, name=name, delay=0.0, conditionEdge="rising")
    by_entity = el("ByEntityCondition", condition)
    triggering = el("TriggeringEntities", by_entity, triggeringEntitiesRule="any")
    el("EntityRef", triggering, entityRef=ctx.ego)
    entity_condition = el("EntityCondition", by_entity)
    el(
        "RelativeDistanceCondition",
        entity_condition,
        entityRef=ctx.entity,
        freespace=True,
        relativeDistanceType="longitudinal",
        rule="lessThan",
        value=ctx.trigger_distance_m,
    )
    return trigger


def event(name: str, actions: list[ET.Element], trigger: ET.Element) -> ET.Element:
    node = el("Event", name=name, priority="overwrite")
    for i, action in enumerate(actions):
        wrapper = el("Action", node, name=f"{name}_action_{i}")
        wrapper.append(action)
    node.append(trigger)
    return node


def _trigger_for(ctx: TemplateContext) -> ET.Element:
    return distance_trigger(ctx) if ctx.event_timed else time_trigger(ctx.cue_s())


# --- actor_dynamics[].state templates ----------------------------------------------


def _walker_cross(ctx: TemplateContext, name: str, speed: float) -> list[ET.Element]:
    # Off ego's approach, never off the clock: a cruising ego covers the DNA's 8 m in
    # under a second, so a mid-clip cue would fire after it had already passed.
    trigger = distance_trigger(ctx) if ctx.trigger_distance_m > 0.0 else _trigger_for(ctx)
    return [event(name, [speed_action(speed, shape="step", over_s=0.5)], trigger)]


def walker_cross_at_crosswalk(ctx: TemplateContext) -> list[ET.Element]:
    return _walker_cross(ctx, "walker_cross_at_crosswalk", WALK_SPEED_MPS)


def walker_cross_midblock(ctx: TemplateContext) -> list[ET.Element]:
    return _walker_cross(ctx, "walker_cross_midblock", WALK_SPEED_MPS)


def walker_cross_stop_go(ctx: TemplateContext) -> list[ET.Element]:
    """Hesitation: step out, stop, then continue — the built-in walker AI has no such state."""
    return [
        event(
            "walker_step_out",
            [speed_action(HESITANT_WALK_SPEED_MPS, over_s=0.5)],
            _trigger_for(ctx),
        ),
        event(
            "walker_hesitate",
            [speed_action(0.0, shape="step", over_s=0.1)],
            time_trigger(ctx.cue_s(1.5), name="hesitate"),
        ),
        event(
            "walker_resume",
            [speed_action(WALK_SPEED_MPS, over_s=0.5)],
            time_trigger(ctx.cue_s(2.5), name="resume"),
        ),
    ]


def vehicle_lane_change_into_ego_lane(ctx: TemplateContext) -> list[ET.Element]:
    """Pull into ego's lane and settle there slower than ego.

    The actor rolls alongside until the cue, then slows while changing lanes. The
    below-ego target speed keeps the maneuver on camera: at ego's speed the car simply
    recedes into the distance instead of crowding it.
    """
    return [
        event(
            "vehicle_cut_in",
            [
                speed_action(ctx.target_speed_mps * 0.5, over_s=2.0),
                lane_change_action(_RIGHT, entity=ctx.entity, over_s=1.5),
            ],
            _trigger_for(ctx),
        )
    ]


def vehicle_lane_change_out_of_ego_lane(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "vehicle_cut_out",
            [lane_change_action(_LEFT, entity=ctx.entity, over_s=2.0)],
            _trigger_for(ctx),
        )
    ]


def actor_hold_position(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "actor_hold_position",
            [speed_action(0.0, shape="step", over_s=0.1)],
            time_trigger(0.0, name="immediate"),
        )
    ]


def actor_emerge_from_occluder(ctx: TemplateContext) -> list[ET.Element]:
    """Spawned behind the nearest static obstruction — the real occluder is unknown."""
    return [
        event(
            "actor_emerge",
            [speed_action(max(ctx.target_speed_mps, WALK_SPEED_MPS), shape="step", over_s=0.5)],
            _trigger_for(ctx),
        )
    ]


def vehicle_follow_ego(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "vehicle_follow_ego",
            [speed_action(ctx.target_speed_mps)],
            time_trigger(0.0, name="immediate"),
        )
    ]


def vehicle_oncoming_lane(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "vehicle_oncoming",
            [speed_action(ctx.target_speed_mps)],
            time_trigger(0.0, name="immediate"),
        )
    ]


def actor_parked_no_autopilot(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "actor_parked",
            [speed_action(0.0, shape="step", over_s=0.1)],
            time_trigger(0.0, name="immediate"),
        )
    ]


def actor_static(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "actor_static",
            [speed_action(0.0, shape="step", over_s=0.1)],
            time_trigger(0.0, name="immediate"),
        )
    ]


# --- planner_logic.ego_maneuver templates ------------------------------------------


def _ego_speed(name: str, factor: float, *, shape: str, over_s: float):
    def build(ctx: TemplateContext) -> list[ET.Element]:
        return [
            event(
                name,
                [speed_action(ctx.target_speed_mps * factor, shape=shape, over_s=over_s)],
                _trigger_for(ctx),
            )
        ]

    return build


def _ego_offset(name: str, offset_m: float):
    def build(ctx: TemplateContext) -> list[ET.Element]:
        return [event(name, [lane_offset_action(offset_m)], _trigger_for(ctx))]

    return build


def _ego_lane_change(name: str, direction: int):
    def build(ctx: TemplateContext) -> list[ET.Element]:
        return [
            event(
                name,
                [lane_change_action(direction, entity=ctx.ego, over_s=2.0)],
                _trigger_for(ctx),
            )
        ]

    return build


def ego_swerve(ctx: TemplateContext) -> list[ET.Element]:
    """Out and back — a swerve that does not return is a lane change."""
    return [
        event(
            "ego_swerve_out",
            [lane_offset_action(SWERVE_OFFSET_M)],
            time_trigger(ctx.cue_s(), name="swerve_out"),
        ),
        event(
            "ego_swerve_back",
            [lane_offset_action(0.0)],
            time_trigger(ctx.cue_s(1.5), name="swerve_back"),
        ),
    ]


def ego_reverse(ctx: TemplateContext) -> list[ET.Element]:
    return [
        event(
            "ego_reverse",
            [speed_action(-abs(ctx.target_speed_mps), shape="linear", over_s=2.0)],
            _trigger_for(ctx),
        )
    ]


# Evaluated at runtime, so it cannot use a subscripted builtin: the render stage imports
# this package under the simulator image's py3.7.
TemplateBuilder = Callable[[TemplateContext], List[ET.Element]]  # noqa: UP006

#: Every ``maneuver_template`` and ``control_template`` name the catalog can emit.
TEMPLATES: dict[str, TemplateBuilder] = {
    "walker_cross_at_crosswalk": walker_cross_at_crosswalk,
    "walker_cross_midblock": walker_cross_midblock,
    "walker_cross_stop_go": walker_cross_stop_go,
    "vehicle_lane_change_into_ego_lane": vehicle_lane_change_into_ego_lane,
    "vehicle_lane_change_out_of_ego_lane": vehicle_lane_change_out_of_ego_lane,
    "actor_hold_position": actor_hold_position,
    "actor_emerge_from_occluder": actor_emerge_from_occluder,
    "vehicle_follow_ego": vehicle_follow_ego,
    "vehicle_oncoming_lane": vehicle_oncoming_lane,
    "actor_parked_no_autopilot": actor_parked_no_autopilot,
    "actor_static": actor_static,
    "ego_constant_speed": _ego_speed("ego_constant_speed", 1.0, shape="linear", over_s=1.0),
    "ego_accelerate": _ego_speed("ego_accelerate", 1.3, shape="linear", over_s=3.0),
    "ego_brake_soft": _ego_speed("ego_brake_soft", 0.5, shape="linear", over_s=2.5),
    "ego_brake_hard": _ego_speed("ego_brake_hard", 0.2, shape="linear", over_s=1.2),
    "ego_emergency_brake": _ego_speed("ego_emergency_brake", 0.0, shape="step", over_s=0.5),
    "ego_lateral_offset_left": _ego_offset("ego_lateral_offset_left", NUDGE_OFFSET_M),
    "ego_lateral_offset_right": _ego_offset("ego_lateral_offset_right", -NUDGE_OFFSET_M),
    "ego_lane_change_left": _ego_lane_change("ego_lane_change_left", _LEFT),
    "ego_lane_change_right": _ego_lane_change("ego_lane_change_right", _RIGHT),
    "ego_yield_at_junction": _ego_speed("ego_yield_at_junction", 0.15, shape="linear", over_s=2.0),
    "ego_stop": _ego_speed("ego_stop", 0.0, shape="linear", over_s=2.0),
    "ego_reverse": ego_reverse,
    "ego_swerve": ego_swerve,
}


def build_events(template: str, ctx: TemplateContext) -> list[ET.Element]:
    """Expand *template* into its events, or return nothing if the name is unknown."""
    builder = TEMPLATES.get(template)
    return builder(ctx) if builder else []
