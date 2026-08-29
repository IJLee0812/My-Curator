"""Road-level resolution: a ``RoadQuery`` to one concrete road, deterministically.

The mapper asserts only that *some* road in *some* loadable town satisfies a query. This
module picks the road, which is where the reconstruction's fidelity actually comes from: 302 of
349 mapped segments resolve to the same four-town candidate set, so the town says very
little while the road — one of 1,282 across the six towns — says a great deal.

Selection is a relaxation ladder. Every hard constraint is applied first; if nothing
survives, constraints are dropped one at a time in a fixed order and each drop is recorded
as a degradation, so a query never fails outright and never silently loses a requirement.
The final tie-break is a stable hash of the segment identifier, which keeps the choice
reproducible without collapsing every segment onto the same road.

Pure logic: candidates are supplied by the caller, never fetched here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from my_curator.domain.sim.reasons import DegradationCode
from my_curator.domain.sim.spec import Degradation, RoadQuery

#: How far into a lane section ego is placed. Far enough that a follower fits behind it
#: (a vehicle is about 4.6 m long) and that ego is off the section boundary; short enough
#: to leave the section usable ahead.
ENTRY_MARGIN_M = 8.0


@dataclass(frozen=True)
class RoadCandidate:
    """One driving lane of one lane section — the unit a scenario is staged on."""

    town: str
    road_id: int
    lane_id: int
    lane_section_s: float
    lane_section_end_s: float
    driving_lanes: int
    speed_kph: float
    lane_types: frozenset[str]
    junction_forms: frozenset[str]
    in_junction: bool

    @property
    def usable_length_m(self) -> float:
        return max(0.0, self.lane_section_end_s - self.lane_section_s)

    @property
    def travel_direction(self) -> int:
        """``+1`` if the lane is driven with increasing ``s``, ``-1`` if against it.

        OpenDRIVE numbers lanes outward from the reference line — negative to its right,
        positive to its left — and the left-hand lanes carry traffic the other way, so
        every placement along the lane has to be signed by this.
        """
        return -1 if self.lane_id > 0 else 1

    @property
    def entry_s(self) -> float:
        """Where along the road an entity is placed, leaving room in front of it."""
        margin = min(ENTRY_MARGIN_M, self.usable_length_m / 2.0)
        if self.travel_direction < 0:
            return self.lane_section_end_s - margin
        return self.lane_section_s + margin

    def s_ahead(self, metres: float) -> float:
        """``metres`` in front of :attr:`entry_s`, clamped to the lane section."""
        target = self.entry_s + self.travel_direction * metres
        return min(max(target, self.lane_section_s), self.lane_section_end_s)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lane_types"] = sorted(self.lane_types)
        data["junction_forms"] = sorted(self.junction_forms)
        data["entry_s"] = self.entry_s
        data["travel_direction"] = self.travel_direction
        return data


@dataclass(frozen=True)
class RoadSelection:
    """The chosen road plus whatever had to be given up to choose it."""

    candidate: RoadCandidate
    degradations: tuple[Degradation, ...] = ()

    @property
    def is_degraded(self) -> bool:
        return bool(self.degradations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "road": self.candidate.to_dict(),
            "degradations": [d.to_dict() for d in self.degradations],
        }


def _speed_ok(c: RoadCandidate, q: RoadQuery) -> bool:
    low, high = q.speed_kph_range
    return low <= c.speed_kph <= high


def _town_ok(c: RoadCandidate, q: RoadQuery) -> bool:
    return c.town in q.candidate_towns


def _intersection_ok(c: RoadCandidate, q: RoadQuery) -> bool:
    if q.intersection_type == "none":
        return not c.in_junction
    return q.intersection_type in c.junction_forms


def _lane_types_ok(c: RoadCandidate, q: RoadQuery) -> bool:
    return set(q.required_lane_types) <= c.lane_types


def _lane_count_ok(c: RoadCandidate, q: RoadQuery) -> bool:
    return c.driving_lanes >= q.min_driving_lanes


class _Constraint:
    """One droppable requirement, with the compromise its drop represents."""

    def __init__(
        self,
        name: str,
        predicate: Callable[[RoadCandidate, RoadQuery], bool],
        code: DegradationCode,
        applied: str,
    ) -> None:
        self.name = name
        self.predicate = predicate
        self.code = code
        self.applied = applied


#: Dropped left to right. Speed goes first because the target speed is ours to set anyway;
#: the town goes next because it is only a proxy for the physical properties below it;
#: lane count goes last because a maneuver that needs an adjacent lane cannot be staged
#: on a road that has none.
_CONSTRAINTS: tuple[_Constraint, ...] = (
    _Constraint(
        "speed",
        _speed_ok,
        DegradationCode.ROAD_TYPE_SUBSTITUTED,
        "road outside the requested speed range; ego target speed is set explicitly",
    ),
    _Constraint(
        "town",
        _town_ok,
        DegradationCode.ROAD_TYPE_SUBSTITUTED,
        "road taken from a town outside the mapped candidate set",
    ),
    _Constraint(
        "intersection",
        _intersection_ok,
        DegradationCode.INTERSECTION_SUBSTITUTED,
        "no road offers the requested junction form; nearest available geometry used",
    ),
    _Constraint(
        "lane_types",
        _lane_types_ok,
        DegradationCode.ROAD_TYPE_SUBSTITUTED,
        "road lacks a required lane type (sidewalk, parking or shoulder)",
    ),
    _Constraint(
        "lane_count",
        _lane_count_ok,
        DegradationCode.ROAD_TYPE_SUBSTITUTED,
        "road has fewer driving lanes than the road class implies",
    ),
)


def _sort_key(c: RoadCandidate) -> tuple:
    """Total order over candidates, independent of input order."""
    return (c.town, c.road_id, c.lane_section_s, c.lane_id)


def _stable_pick(pool: Sequence[RoadCandidate], seed: str) -> RoadCandidate:
    """Pick one candidate by a stable hash of *seed*.

    ``hash()`` is salted per process, so it cannot be used: the same segment has to
    resolve to the same road across runs and machines.
    """
    ordered = sorted(pool, key=_sort_key)
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    return ordered[int.from_bytes(digest, "big") % len(ordered)]


def select_road(
    query: RoadQuery,
    candidates: Iterable[RoadCandidate],
    seed: str,
    min_length_m: float = 0.0,
) -> RoadSelection | None:
    """Resolve *query* to one road, relaxing constraints only as far as needed.

    Returns ``None`` only when *candidates* is empty — with any candidate at all the
    ladder bottoms out at "no constraints", which always matches.
    """
    pool = list(candidates)
    if not pool:
        return None

    # Widen the search one constraint at a time, softest first, until something matches.
    active = list(_CONSTRAINTS)
    matched: list[RoadCandidate] = []
    while True:
        matched = [c for c in pool if all(k.predicate(c, query) for k in active)]
        if matched or not active:
            break
        active.pop(0)
    if not matched:
        matched = pool

    # Length is a preference, not a constraint: the route planner can continue onto a
    # connected road, so a short section costs nothing beyond a less convenient start.
    roomy = [c for c in matched if c.usable_length_m >= min_length_m]
    chosen = _stable_pick(roomy or matched, seed)

    # Report what the chosen road actually gives up, not which constraints the search
    # happened to relax on the way: widening the search past a constraint the winner
    # satisfies anyway is not a compromise, and recording it would overstate the damage.
    degradations = tuple(
        Degradation(
            code=k.code,
            field_path=f"topology.road.{k.name}",
            requested=_requested_value(k.name, query),
            applied=k.applied,
        )
        for k in _CONSTRAINTS
        if not k.predicate(chosen, query)
    )
    return RoadSelection(candidate=chosen, degradations=degradations)


def _requested_value(name: str, q: RoadQuery) -> str:
    if name == "speed":
        return f"{q.speed_kph_range[0]}-{q.speed_kph_range[1]} kph"
    if name == "town":
        return "|".join(q.candidate_towns) or "(none)"
    if name == "intersection":
        return q.intersection_type
    if name == "lane_types":
        return "+".join(q.required_lane_types) or "(none)"
    return f"{q.min_driving_lanes} driving lane(s)"
