"""Parse CARLA's built-in OpenDRIVE town networks into ``RoadCandidate`` rows.

Reads the ``.xodr`` files shipped inside the CARLA image and flattens them to one row per
driving lane per lane section — the unit a scenario is staged on. Runs offline against the
files alone: no CARLA client, no simulator, no GPU.

Two properties of CARLA's OpenDRIVE drive the shape of this module:

* **Speed limits are sparse.** Only 20-25% of roads carry ``<type><speed>``; the rest —
  overwhelmingly junction connecting roads — inherit it. Speed is therefore propagated
  across road links before candidates are emitted, falling back to the town's slowest
  observed limit and finally to CARLA's own 30 km/h default.
* **Junction form is a property of the junction, not the road.** Signalization, arm count
  and roundabout identity are resolved per junction first, then attached to every road
  that touches it.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from my_curator.domain.sim.catalog import LOADABLE_TOWNS, ROUNDABOUT_JUNCTIONS
from my_curator.domain.sim.road_index import RoadCandidate

log = logging.getLogger(__name__)

#: In-container path; copy the files out to run this on the host.
CARLA_OPENDRIVE_DIR = "/home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive"

#: CARLA's default when a road declares no speed limit.
DEFAULT_SPEED_KPH = 30.0

_MPH_TO_KPH = 1.609344

#: Rounded to the nearest 10 so parsed limits land on the same values the capability
#: catalog was measured with; the raw conversion (25 mph = 40.23) would miss every
#: closed speed range the catalog declares.
_SPEED_QUANTUM = 10

_NO_JUNCTION = "-1"


def _speed_kph(type_el: ET.Element | None) -> float | None:
    if type_el is None:
        return None
    speed = type_el.find("speed")
    if speed is None or speed.get("max") is None:
        return None
    raw = float(speed.get("max", "0"))
    if speed.get("unit", "m/s") == "mph":
        raw *= _MPH_TO_KPH
    elif speed.get("unit") == "m/s":
        raw *= 3.6
    return float(round(raw / _SPEED_QUANTUM) * _SPEED_QUANTUM)


def _linked_road_ids(road: ET.Element) -> set[str]:
    ids = set()
    for tag in ("predecessor", "successor"):
        el = road.find(f"link/{tag}")
        if el is not None and el.get("elementType") == "road":
            ids.add(el.get("elementId", ""))
    return {i for i in ids if i}


def _adjacent_junction_ids(road: ET.Element, arms: dict[str, set[str]]) -> set[str]:
    """Junctions this road touches, from both directions of the relationship.

    A road usually names the junction it runs into via ``<link>``, but the junction's own
    ``incomingRoad`` list is the authoritative statement of which roads are its arms, so
    both are consulted: a road that appears only in the junction still gets its form.
    """
    ids = set()
    own = road.get("junction", _NO_JUNCTION)
    if own != _NO_JUNCTION:
        ids.add(own)
    for tag in ("predecessor", "successor"):
        el = road.find(f"link/{tag}")
        if el is not None and el.get("elementType") == "junction":
            ids.add(el.get("elementId", ""))
    road_id = road.get("id", "")
    ids |= {jid for jid, incoming in arms.items() if road_id in incoming}
    return {i for i in ids if i}


def _propagate_speeds(
    roads: dict[str, ET.Element], junctions: dict[str, set[str]]
) -> dict[str, float]:
    """Fill in the speed of every road, spreading declared limits across links."""
    known = {rid: s for rid, r in roads.items() if (s := _speed_kph(r.find("type"))) is not None}
    if not known:
        return dict.fromkeys(roads, DEFAULT_SPEED_KPH)

    # A connecting road takes the limit of the junction it belongs to, i.e. of the roads
    # that feed it; everything else takes it from whatever it links to.
    for _ in range(len(LOADABLE_TOWNS)):
        added = False
        for rid, road in roads.items():
            if rid in known:
                continue
            own = road.get("junction", _NO_JUNCTION)
            neighbours = (
                junctions.get(own, set()) if own != _NO_JUNCTION else _linked_road_ids(road)
            )
            speeds = [known[n] for n in neighbours if n in known]
            if speeds:
                known[rid] = min(speeds)
                added = True
        if not added:
            break

    fallback = min(known.values(), default=DEFAULT_SPEED_KPH)
    return {rid: known.get(rid, fallback) for rid in roads}


def _junction_forms(root: ET.Element, town: str) -> dict[str, frozenset[str]]:
    """Classify every junction: signalization, arm count and roundabout identity."""
    signal_roads = {r.get("id") for r in root.findall("road") if r.find(".//signal") is not None}
    roundabouts = ROUNDABOUT_JUNCTIONS.get(town, frozenset())

    forms: dict[str, frozenset[str]] = {}
    for junction in root.findall("junction"):
        jid = junction.get("id", "")
        arms = {c.get("incomingRoad") for c in junction.findall("connection")}
        tags = {"signalized"} if arms & signal_roads else {"unsignalized"}
        if len(arms) == 3:
            tags.add("t_junction")
        elif len(arms) <= 2:
            # A two-arm junction is a merge or diverge, not a crossing.
            tags.add("direct_connection")
        if jid.isdigit() and int(jid) in roundabouts:
            tags.add("roundabout")
        forms[jid] = frozenset(tags)
    return forms


def _junction_arms(root: ET.Element) -> dict[str, set[str]]:
    return {
        j.get("id", ""): {c.get("incomingRoad", "") for c in j.findall("connection")}
        for j in root.findall("junction")
    }


def _has_crosswalk(road: ET.Element) -> bool:
    return any(o.get("type") == "crosswalk" for o in road.iter("object"))


def _lane_sections(road: ET.Element) -> Iterator[tuple[ET.Element, float, float]]:
    """Yield each lane section with the s-range it spans."""
    sections = road.findall("lanes/laneSection")
    road_length = float(road.get("length", "0"))
    for i, section in enumerate(sections):
        start = float(section.get("s", "0"))
        end = float(sections[i + 1].get("s", "0")) if i + 1 < len(sections) else road_length
        yield section, start, end


def parse_town(path: Path, town: str) -> list[RoadCandidate]:
    """Flatten one town's OpenDRIVE into candidates, one per driving lane per section."""
    root = ET.parse(path).getroot()
    roads = {r.get("id", ""): r for r in root.findall("road")}
    arms = _junction_arms(root)
    forms = _junction_forms(root, town)
    speeds = _propagate_speeds(roads, arms)

    candidates: list[RoadCandidate] = []
    for road_id, road in roads.items():
        if not road_id.lstrip("-").isdigit():
            continue
        in_junction = road.get("junction", _NO_JUNCTION) != _NO_JUNCTION
        junction_forms: set[str] = set()
        for jid in _adjacent_junction_ids(road, arms):
            junction_forms |= forms.get(jid, frozenset())
        if _has_crosswalk(road):
            junction_forms.add("crosswalk")
        frozen_forms = frozenset(junction_forms)
        speed = speeds.get(road_id, DEFAULT_SPEED_KPH)

        for section, start, end in _lane_sections(road):
            lane_types = frozenset(ln.get("type", "none") for ln in section.iter("lane"))
            for side in ("left", "right"):
                driving = [
                    ln
                    for ln in section.findall(f"{side}/lane")
                    if ln.get("type") == "driving" and ln.get("id", "0").lstrip("-").isdigit()
                ]
                candidates.extend(
                    RoadCandidate(
                        town=town,
                        road_id=int(road_id),
                        lane_id=int(lane.get("id", "0")),
                        lane_section_s=start,
                        lane_section_end_s=end,
                        driving_lanes=len(driving),
                        speed_kph=speed,
                        lane_types=lane_types,
                        junction_forms=frozen_forms,
                        in_junction=in_junction,
                    )
                    for lane in driving
                )
    return candidates


def parse_towns(
    opendrive_dir: str | Path = CARLA_OPENDRIVE_DIR,
    towns: tuple[str, ...] = LOADABLE_TOWNS,
) -> list[RoadCandidate]:
    """Parse every loadable town, skipping any whose ``.xodr`` is absent."""
    base = Path(opendrive_dir)
    out: list[RoadCandidate] = []
    for town in towns:
        path = base / f"{town}.xodr"
        if not path.is_file():
            log.warning("OpenDRIVE for %s not found at %s — skipping", town, path)
            continue
        town_rows = parse_town(path, town)
        log.info("%s: %d road candidate(s)", town, len(town_rows))
        out.extend(town_rows)
    return out
