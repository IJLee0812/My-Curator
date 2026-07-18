"""DNA → natural-language text for the narrative-text embedding (P4-7).

Pure domain logic (stdlib only): turns a scenario DNA dict into the string fed
to the Cosmos-Embed1 text tower.  Shared by the ``/v1/ingest`` text path and the
corpus re-embed script so both towers see identical text.

v0.2 change: the stale top-level ``scene_summary`` (empty on v0.2 DNA) is
dropped; the builder now includes the structured ODD / topology / maneuver
tokens plus the v0.2 narrative signals — ``scene_description``,
``planner_logic.risk_level_rationale`` and a meaningful
``planner_logic.safety_event.event_type`` (``none`` is skipped as noise).
"""

from __future__ import annotations

_NULL_TOKENS = frozenset({"none", "unknown", ""})


def dna_to_text(dna: dict) -> str:
    """Assemble the text-tower input string from a scenario DNA dict.

    Robust to missing / null nested blocks; returns ``"driving scene"`` when the
    DNA yields no usable tokens.
    """
    parts: list[str] = []

    odd = dna.get("odd") or {}
    if weather := odd.get("weather"):
        parts.append(weather)
    if lighting := odd.get("lighting"):
        parts.append(lighting)

    topology = dna.get("topology") or {}
    if road := topology.get("road_type"):
        parts.append(road)
    if lane := topology.get("lane_event"):
        parts.append(lane)

    planner = dna.get("planner_logic") or {}
    if maneuver := planner.get("ego_maneuver"):
        parts.append(maneuver)

    safety = planner.get("safety_event") or {}
    event_type = safety.get("event_type")
    if event_type and event_type not in _NULL_TOKENS:
        parts.append(event_type)

    if rationale := planner.get("risk_level_rationale"):
        parts.append(rationale)
    if description := dna.get("scene_description"):
        parts.append(description)

    return " ".join(parts) if parts else "driving scene"
