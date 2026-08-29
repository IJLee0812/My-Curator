"""Stage a compiled scenario in a running simulator and drive it.

Reads a :class:`ScenarioProgram` — never a ``SimSpec`` — so what executes is the document
that was compiled, validated and archived. Entities are placed on the OpenDRIVE lane
positions the scenario names, then each maneuver group's events fire as their triggers
come true.

Two constraints from earlier phases shape this module. The simulator segfaults on a
runtime map switch, so the town is *verified*, never loaded. And control is split by role:
*ego* is driven by CARLA's ``BasicAgent``, while every *adversary* is moved kinematically
along its lane's waypoints. An adversary under an autonomous agent re-plans its own route,
declines lane changes and brakes on its own judgement; a reconstruction needs the DNA's
choreography executed exactly.

CARLA is imported lazily so this module stays importable on a host with no simulator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from my_curator.domain.sim.catalog import parse_blueprint_filter
from my_curator.domain.sim.program import (
    LaneChangeAction,
    LaneOffsetAction,
    LanePlacement,
    ScenarioProgram,
    SpeedAction,
    TeleportAction,
    Trigger,
)
from my_curator.domain.sim.reasons import RenderFailure

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2000
CONNECT_TIMEOUT_S = 20.0

#: Heights above the lane to try, in order. OpenDRIVE waypoints in some towns report z=0
#: where the road surface is metres higher, so a spawn at the nominal height lands inside
#: the terrain and is rejected. The first height that takes wins.
SPAWN_LIFTS_M = (0.3, 0.6, 1.0, 2.0, 3.0)

#: How far ahead the route is planned. Long enough that a five-second segment never runs
#: off the end of it at any speed the corpus asks for.
ROUTE_AHEAD_M = 400.0
_ROUTE_STEP_M = 5.0

MPS_TO_KPH = 3.6


class ExecutionError(RuntimeError):
    """Staging or running the scenario failed, with the code the ledger records."""

    def __init__(self, reason: RenderFailure, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class _SpeedRamp:
    """A speed target approached over time, honouring the action's stated dynamics.

    ``BasicAgent`` only knows an instantaneous target, so a hard brake handed to it as a
    bare number decays on the PID's schedule rather than the scenario's. Interpolating the
    *target* makes the controller trace the stated ramp.
    """

    start_mps: float
    target_mps: float
    start_s: float
    over_s: float

    def value(self, elapsed_s: float) -> float:
        if self.over_s <= 0.0:
            return self.target_mps
        fraction = min(1.0, max(0.0, (elapsed_s - self.start_s) / self.over_s))
        return self.start_mps + (self.target_mps - self.start_mps) * fraction


@dataclass
class StagedEntity:
    name: str
    actor: Any
    is_walker: bool
    is_ego: bool
    agent: Any = None
    ramp: _SpeedRamp | None = None
    #: Adversaries are moved kinematically along their lane instead of being driven.
    kinematic: bool = False
    lane_wp: Any = None
    #: An in-flight kinematic lane change: {"side", "progress", "over_s"}.
    change: Any = None
    cruise_mps: float = 0.0
    heading: Any = None
    #: Toward the centre of the road: in right-hand traffic the opposing carriageway is
    #: always left of the direction of travel, so a walker crosses left, not onto the kerb.
    crossing: Any = None
    crosses: bool = False

    @property
    def walk_direction(self) -> Any:
        """Which way a walker moves when it is given a speed.

        A pedestrian who crosses the road moves across the lane, not along it. Both
        vectors come from the lane the walker was placed on, so the crossing is
        perpendicular to the carriageway rather than to the world axes.
        """
        return self.crossing if self.crosses else self.heading


@dataclass
class _PendingEvent:
    entity: str
    name: str
    actions: tuple
    trigger: Trigger
    fired: bool = False


@dataclass
class ExecutionResult:
    town: str
    entities: int
    events_fired: list[str] = field(default_factory=list)


def connect(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> tuple[Any, Any]:
    import carla

    try:
        client = carla.Client(host, port)
        client.set_timeout(CONNECT_TIMEOUT_S)
        return client, client.get_world()
    except RuntimeError as exc:
        raise ExecutionError(
            RenderFailure.SIMULATOR_UNREACHABLE, f"no simulator at {host}:{port} — {exc}"
        ) from exc


def loaded_town(world: Any) -> str:
    return str(world.get_map().name).rsplit("/", 1)[-1]


def ensure_town(world: Any, town: str) -> None:
    """Refuse to run on the wrong town rather than switching to the right one."""
    current = loaded_town(world)
    if current != town:
        raise ExecutionError(
            RenderFailure.WRONG_TOWN_LOADED,
            f"server is on {current}, scenario needs {town}; boot it with CARLA_MAP={town}",
        )


def _left_of(transform: Any) -> Any:
    """The unit vector to the left of *transform*'s facing."""
    import carla

    right = transform.get_right_vector()
    return carla.Vector3D(-right.x, -right.y, 0.0)


def apply_weather(world: Any, weather: dict[str, float]) -> None:
    parameters = world.get_weather()
    for name, value in weather.items():
        if hasattr(parameters, name):
            setattr(parameters, name, float(value))
    world.set_weather(parameters)


class ScenarioExecutor:
    """One scenario, from spawn to stop trigger."""

    def __init__(self, world: Any, program: ScenarioProgram, *, tick_s: float) -> None:
        self._world = world
        self._program = program
        self._tick_s = tick_s
        self._entities: dict[str, StagedEntity] = {}
        self._events: list[_PendingEvent] = []
        self._original_settings: Any = None

    # -- setup ---------------------------------------------------------------------

    def __enter__(self) -> ScenarioExecutor:
        self._enter_synchronous_mode()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def _is_night(self) -> bool:
        return self._program.environment.sun_elevation < 0.0

    def _enter_synchronous_mode(self) -> None:
        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._tick_s
        self._world.apply_settings(settings)

    def _waypoint(self, placement: LanePlacement) -> Any:
        waypoint = self._world.get_map().get_waypoint_xodr(
            placement.road_id, placement.lane_id, placement.s
        )
        if waypoint is None:
            raise ExecutionError(
                RenderFailure.SPAWN_REJECTED,
                f"road {placement.road_id} lane {placement.lane_id} has no point at "
                f"s={placement.s}",
            )
        return waypoint

    def _blueprint(self, filter_expression: str) -> Any:
        """Resolve a catalog filter to one blueprint, deterministically."""
        patterns, attributes = parse_blueprint_filter(filter_expression)
        library = self._world.get_blueprint_library()

        matches: dict[str, Any] = {}
        for pattern in patterns:
            for blueprint in library.filter(pattern):
                matches[blueprint.id] = blueprint
        for name, value in attributes.items():
            matches = {
                key: bp
                for key, bp in matches.items()
                if bp.has_attribute(name) and bp.get_attribute(name) == value
            }
        if not matches:
            raise ExecutionError(
                RenderFailure.BLUEPRINT_MISSING, f"no blueprint matches {filter_expression!r}"
            )
        return matches[sorted(matches)[0]]

    def stage(self) -> None:
        """Spawn every entity where the scenario's Init places it."""
        import carla

        for definition in self._program.entities:
            state = self._program.init_for(definition.name)
            if state is None:
                raise ExecutionError(
                    RenderFailure.SPAWN_REJECTED, f"entity {definition.name!r} has no Init state"
                )
            waypoint = self._waypoint(state.placement)
            blueprint = self._blueprint(definition.blueprint_filter)
            actor = None
            for lift in SPAWN_LIFTS_M:
                transform = carla.Transform(
                    waypoint.transform.location + carla.Location(z=lift),
                    waypoint.transform.rotation,
                )
                actor = self._world.try_spawn_actor(blueprint, transform)
                if actor is not None:
                    break
            if actor is None:
                raise ExecutionError(
                    RenderFailure.SPAWN_REJECTED,
                    f"{definition.name!r} could not be placed on road "
                    f"{state.placement.road_id} lane {state.placement.lane_id}",
                )
            if not definition.is_walker and self._is_night:
                # A night scene is lit by headlights; a vehicle driving dark renders as a
                # silhouette the source video never shows.
                actor.set_light_state(
                    carla.VehicleLightState(
                        carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position
                    )
                )
            kinematic = not definition.is_walker and not definition.is_ego
            if kinematic:
                # Choreography, not autonomy: physics off, moved along the lane by hand.
                # This also keeps two-wheelers upright — under physics they fall over.
                actor.set_simulate_physics(False)
            self._entities[definition.name] = StagedEntity(
                name=definition.name,
                actor=actor,
                is_walker=definition.is_walker,
                is_ego=definition.is_ego,
                kinematic=kinematic,
                lane_wp=waypoint,
                cruise_mps=float(state.speed_mps),
                heading=waypoint.transform.get_forward_vector(),
                crossing=_left_of(waypoint.transform),
                crosses=definition.crosses_the_road,
            )

        self._world.tick()
        for definition in self._program.entities:
            state = self._program.init_for(definition.name)
            self._apply_initial_speed(self._entities[definition.name], state.speed_mps)
        self._plan_routes()
        self._collect_events()

    def entity(self, name: str) -> StagedEntity:
        if name not in self._entities:
            raise ExecutionError(RenderFailure.SPAWN_REJECTED, f"{name!r} was never staged")
        return self._entities[name]

    def _apply_initial_speed(self, entity: StagedEntity, speed_mps: float) -> None:
        import carla

        if entity.is_walker:
            entity.actor.apply_control(
                carla.WalkerControl(direction=entity.walk_direction, speed=float(speed_mps))
            )
            return
        if entity.kinematic:
            return  # cruise_mps drives it; a velocity means nothing with physics off
        heading = entity.heading
        entity.actor.set_target_velocity(
            carla.Vector3D(heading.x * speed_mps, heading.y * speed_mps, 0.0)
        )

    def _plan_routes(self) -> None:
        from agents.navigation.basic_agent import BasicAgent

        carla_map = self._world.get_map()
        for entity in self._entities.values():
            if entity.is_walker or entity.kinematic:
                continue
            state = self._program.init_for(entity.name)
            agent = BasicAgent(entity.actor, target_speed=state.speed_mps * MPS_TO_KPH)
            # A scripted cue has to happen when the scenario says it does, not when the
            # junction lets it; ambient traffic is the only thing that obeys the lights.
            agent.ignore_traffic_lights(True)
            agent.ignore_stop_signs(True)
            agent.ignore_vehicles(True)
            agent.set_destination(self._destination(carla_map, entity.actor.get_location()))
            entity.agent = agent

    def _destination(self, carla_map: Any, location: Any) -> Any:
        waypoint = carla_map.get_waypoint(location)
        travelled = 0.0
        while travelled < ROUTE_AHEAD_M:
            following = waypoint.next(_ROUTE_STEP_M)
            if not following:
                break
            waypoint = following[0]
            travelled += _ROUTE_STEP_M
        return waypoint.transform.location

    def _collect_events(self) -> None:
        for group in self._program.groups:
            if group.entity not in self._entities:
                raise ExecutionError(
                    RenderFailure.SCENARIO_UNSUPPORTED,
                    f"maneuver group drives unstaged entity {group.entity!r}",
                )
            for event in group.events:
                self._events.append(
                    _PendingEvent(
                        entity=group.entity,
                        name=event.name,
                        actions=event.actions,
                        trigger=event.trigger,
                    )
                )

    # -- run -----------------------------------------------------------------------

    def _fires(self, pending: _PendingEvent, elapsed_s: float) -> bool:
        trigger = pending.trigger
        if trigger.is_timed:
            return elapsed_s >= trigger.value
        subject = self._entities.get(trigger.entity_ref)
        ego = self._entities.get(self._program.ego.name)
        if subject is None or ego is None:
            return False
        return ego.actor.get_location().distance(subject.actor.get_location()) < trigger.value

    def _apply(self, entity: StagedEntity, action: Any, elapsed_s: float) -> None:
        import carla

        if isinstance(action, SpeedAction):
            if entity.is_walker:
                entity.actor.apply_control(
                    carla.WalkerControl(
                        direction=entity.walk_direction, speed=abs(action.target_mps)
                    )
                )
            else:
                if entity.kinematic:
                    current = entity.ramp.value(elapsed_s) if entity.ramp else entity.cruise_mps
                else:
                    velocity = entity.actor.get_velocity()
                    current = (velocity.x**2 + velocity.y**2) ** 0.5
                entity.ramp = _SpeedRamp(current, abs(action.target_mps), elapsed_s, action.over_s)
        elif isinstance(action, LaneChangeAction):
            side = self._lane_change_direction(entity, action)
            if not side:
                return  # already in ego's lane; there is nothing to pull into
            if entity.kinematic:
                entity.change = {"side": side, "progress": 0.0, "over_s": max(0.5, action.over_s)}
            elif entity.agent is not None:
                entity.agent.lane_change(side, lane_change_time=max(0.5, action.over_s))
        elif isinstance(action, LaneOffsetAction):
            if entity.agent is not None:
                entity.agent.set_offset(action.offset_m)
        elif isinstance(action, TeleportAction):
            waypoint = self._waypoint(action.placement)
            entity.actor.set_transform(waypoint.transform)
        else:
            raise ExecutionError(
                RenderFailure.SCENARIO_UNSUPPORTED, f"cannot execute {type(action).__name__}"
            )

    def _lane_change_direction(self, entity: StagedEntity, action: Any) -> str:
        """Which way a lane change actually goes.

        For an adversary the maneuver means "toward ego's lane", and whether that is left
        or right depends on which side it was staged — which only the live world knows.
        Ego's own lane changes keep the document's direction. An adversary already sharing
        ego's lane gets ``""``: there is nothing to pull into.
        """
        fallback = "left" if action.relative_lane > 0 else "right"
        if entity.is_ego:
            return fallback
        ego = self._entities.get(self._program.ego.name)
        if ego is None:
            return fallback
        carla_map = self._world.get_map()
        waypoint = carla_map.get_waypoint(entity.actor.get_location())
        ego_waypoint = carla_map.get_waypoint(ego.actor.get_location())
        if waypoint.lane_id == ego_waypoint.lane_id and waypoint.road_id == ego_waypoint.road_id:
            return ""
        if waypoint.road_id != ego_waypoint.road_id:
            return fallback
        for side, neighbour in (
            ("left", waypoint.get_left_lane()),
            ("right", waypoint.get_right_lane()),
        ):
            if neighbour is not None and neighbour.lane_id == ego_waypoint.lane_id:
                return side
        return fallback

    def _drive(self, elapsed_s: float, advance: bool = True) -> None:
        # Kinematic adversaries hold still through the warm-up (``advance=False``): their
        # motion is the position update, so a full-speed start on the first recorded tick
        # shows no seam, and the staging math only has to account for ego's own travel.
        for entity in self._entities.values():
            if entity.kinematic:
                if advance:
                    self._advance(entity, elapsed_s)
            elif entity.agent is not None:
                if entity.ramp is not None:
                    entity.agent.set_target_speed(entity.ramp.value(elapsed_s) * MPS_TO_KPH)
                entity.actor.apply_control(entity.agent.run_step())

    def _advance(self, entity: StagedEntity, elapsed_s: float) -> None:
        """Move a kinematic adversary one tick along its lane.

        The lane's waypoint chain is the trajectory; a lane change is a lateral blend
        between the source and target lane centres while the chain keeps advancing. The
        road's end simply holds the actor in place.
        """
        import carla

        speed = entity.ramp.value(elapsed_s) if entity.ramp is not None else entity.cruise_mps
        step = speed * self._tick_s
        waypoint = entity.lane_wp
        if step > 1e-3:
            following = waypoint.next(step)
            if following:
                waypoint = following[0]
                entity.lane_wp = waypoint
        location = waypoint.transform.location
        rotation = waypoint.transform.rotation
        if entity.change is not None:
            change = entity.change
            change["progress"] = min(1.0, change["progress"] + self._tick_s / change["over_s"])
            target = (
                waypoint.get_left_lane() if change["side"] == "left" else waypoint.get_right_lane()
            )
            if target is None:
                entity.change = None  # nowhere to go from this section; stay in lane
            else:
                fraction = change["progress"]
                other = target.transform.location
                location = carla.Location(
                    location.x + (other.x - location.x) * fraction,
                    location.y + (other.y - location.y) * fraction,
                    location.z + (other.z - location.z) * fraction,
                )
                if fraction >= 1.0:
                    entity.lane_wp = target
                    entity.change = None
        entity.actor.set_transform(
            carla.Transform(carla.Location(location.x, location.y, location.z + 0.05), rotation)
        )

    def step(self, elapsed_s: float, fire: bool = True) -> list[str]:
        """Fire whatever is due at *elapsed_s* and advance every controller one step.

        ``fire=False`` marks a warm-up tick: the world settles but is not yet recorded.
        Timed triggers still fire, since the compiler authors them warmup-aware, but
        distance triggers are held back — the DNA stages an event actor *at* its trigger
        distance, so the condition is true from tick 0 and the maneuver would play out
        before the cameras start.
        """
        fired = []
        for pending in self._events:
            if not fire and not pending.trigger.is_timed:
                continue
            if not pending.fired and self._fires(pending, elapsed_s):
                for action in pending.actions:
                    self._apply(self._entities[pending.entity], action, elapsed_s)
                pending.fired = True
                fired.append(pending.name)
        self._drive(elapsed_s, advance=fire)
        return fired

    def staged_entities(self) -> dict[str, StagedEntity]:
        """A live view of everything on stage, for measurement by the caller."""
        return dict(self._entities)

    def close(self) -> None:
        for entity in self._entities.values():
            try:
                entity.actor.destroy()
            except RuntimeError:
                log.debug("actor %s already gone", entity.name)
        self._entities.clear()
        if self._original_settings is not None:
            self._world.apply_settings(self._original_settings)
            self._original_settings = None
