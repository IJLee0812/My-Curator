"""Post-extraction Scenario DNA repair (v0.2 re-curation hotfix).

Pure-domain (stdlib only), shared by the publisher (pre-validate) and consumer
(pre-store). Repairs the measured Scout v0.2 failure modes so a valid doc can be
stored: de-flatten dotted/bracket keys, coerce enum drift, force safety_event
null-invariants, truncate over-long text. The last-resort enum fallback defaults
a present-but-invalid required enum and drops unmappable actors; it never
fabricates an injury severity or an absent required field (those route to review).
"""

from __future__ import annotations

import copy
from typing import Any

SCENE_DESCRIPTION_MAX = 500
RISK_RATIONALE_MAX = 300

# Enum-drift synonym maps (canonicalised drift value -> v0.2 enum).
# "urban" -> "primary" is the approved default for the v0.2 primary/secondary split.
_ROAD_TYPE_MAP = {"urban": "primary"}
_LIGHTING_MAP = {"overcast": "overcast_day"}
_ACTOR_CLASS_MAP = {
    "person": "pedestrian",
    "bicycle": "cyclist",
    "bike": "cyclist",
    "car": "vehicle_car",
    "sedan": "vehicle_car",
    "suv": "vehicle_car",
    "vehicle_sedan": "vehicle_car",
    "vehicle_suv": "vehicle_car",
    "van": "vehicle_van",
    "bus": "vehicle_bus",
    "truck": "vehicle_truck",
    "motorcycle": "motorcyclist",
    "motorbike": "motorcyclist",
    "vehicle_motorcycle": "motorcyclist",
    "escooter_rider": "standup_scooter_rider",
}

# Fills an absent provenance block only.
DEFAULT_REFERENCE_STANDARDS = [
    "ASAM OSI v3.x",
    "OpenDRIVE v1.5M",
    "WOD-E2E",
    "ISO 21448:2022",
    "NVIDIA CDS (2025-Q4)",
    "VLM-AutoDrive (arXiv:2603.18178)",
]

# Valid v0.2 enum sets for the last-resort fallback; mirror
# schemas/scenario_dna_v0_2.schema.json (drift caught by a cross-check test).
_SENSOR_FIDELITY = {
    "clean",
    "lens_flare",
    "droplets_on_lens",
    "motion_blur",
    "low_contrast",
    "overexposed",
}
_ACTOR_CLASS = {
    "pedestrian",
    "cyclist",
    "motorcyclist",
    "vehicle_car",
    "vehicle_van",
    "vehicle_truck",
    "vehicle_bus",
    "vehicle_emergency",
    "vehicle_construction",
    "animal",
    "debris",
    "construction_object",
    "obstacle",
    "standup_scooter_rider",
    "e_bike_rider",
    "delivery_motorcycle",
    "wheelchair_user",
}
_ACTOR_STATE = {
    "crossing",
    "hesitating",
    "jaywalking",
    "cutin",
    "cutout",
    "stopped",
    "emerging",
    "tailing",
    "oncoming",
    "parked",
    "static",
}
_DISTANCE_BUCKET = {"near", "mid", "far"}
# (parent, field) -> (valid enum set, conservative default)
_ENUM_DEFAULTS: dict[tuple[str, str], tuple[set[str], str]] = {
    ("odd", "weather"): (
        {
            "clear",
            "overcast",
            "light_rain",
            "heavy_rain",
            "snow",
            "heavy_snow",
            "fog",
            "mist",
            "sleet",
        },
        "clear",
    ),
    ("odd", "lighting"): ({"day", "dawn", "dusk", "night", "tunnel", "overcast_day"}, "day"),
    ("topology", "road_type"): (
        {
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "residential",
            "service",
            "rural",
            "parking",
            "walkway",
            "cycling",
        },
        "primary",
    ),
    ("topology", "lane_event"): (
        {"normal", "construction_divert", "lane_closed", "merge", "split", "unmarked"},
        "normal",
    ),
    ("topology", "intersection_type"): (
        {
            "none",
            "signalized",
            "unsignalized",
            "roundabout",
            "t_junction",
            "crosswalk",
            "direct_connection",
        },
        "none",
    ),
    ("planner_logic", "ego_maneuver"): (
        {
            "cruise",
            "accelerate",
            "brake_soft",
            "brake_hard",
            "emergency_brake",
            "nudge_left",
            "nudge_right",
            "lane_change_left",
            "lane_change_right",
            "yield",
            "stop",
            "reverse",
            "swerve",
        },
        "cruise",
    ),
    ("planner_logic", "risk_level"): ({"nominal", "elevated", "critical"}, "nominal"),
}


def normalize_dna(dna: dict[str, Any]) -> dict[str, Any]:
    """Return a structurally-repaired copy of *dna* (no field injection).

    A non-dict or ``{"raw_text": ...}`` fallback is returned unchanged.
    """
    if not isinstance(dna, dict):
        return dna
    out = copy.deepcopy(dna)
    _deflatten(out)
    _coerce_enums(out)
    _fallback_enums(out)
    _default_safety_event(out)
    _normalize_safety_event(out)
    _truncate_lengths(out)
    return out


def ensure_managed_fields(
    dna: dict[str, Any],
    *,
    dna_version: str = "0.2.0",
    clip_id: object | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    scout_prompt_hash: str = "",
    pipeline_version: str = "",
    scout_models: list[str] | None = None,
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """Author the pipeline-managed envelope in place and return *dna*.

    The Scout omits these (the prompt says the pipeline owns them), so
    ``dna_version`` and ``provenance`` are set authoritatively rather than
    trusted; ``timestamp_range`` is set only when the model left it absent/zero.
    """
    if not isinstance(dna, dict):
        return dna

    dna["dna_version"] = dna_version
    if clip_id is not None:
        dna["clip_id"] = str(clip_id)

    if start_s is not None and end_s is not None:
        tr = dna.get("timestamp_range")
        if not isinstance(tr, dict) or not tr.get("end_s"):
            dna["timestamp_range"] = {"start_s": float(start_s), "end_s": float(end_s)}

    prov = dna.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
    prov["scout_models"] = scout_models or prov.get("scout_models") or ["cosmos-reason2-8b"]
    prov["scout_prompt_hash"] = scout_prompt_hash or prov.get("scout_prompt_hash", "")
    prov["pipeline_version"] = pipeline_version or prov.get("pipeline_version", "")
    if "is_synthetic" not in prov:
        prov["is_synthetic"] = is_synthetic
    prov["reference_standards"] = prov.get("reference_standards") or list(
        DEFAULT_REFERENCE_STANDARDS
    )
    dna["provenance"] = prov
    return dna


def _deflatten(dna: dict[str, Any]) -> None:
    """Move dotted/bracket root keys into nested objects; existing nesting wins."""
    for key in list(dna.keys()):
        base = key[:-2] if key.endswith("[]") else key
        if "." not in base:
            if base != key:
                dna.setdefault(base, dna.pop(key))
            continue
        value = dna.pop(key)
        # Strip "[]" from every path part so "actor_dynamics[].actor_class" nests.
        parts = [p[:-2] if p.endswith("[]") else p for p in base.split(".")]
        cursor = dna
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor.setdefault(parts[-1], value)


def _canon(value: Any) -> Any:
    """Strip/lowercase an enum string and map spaces/hyphens to ``_`` (non-str passes through)."""
    if not isinstance(value, str):
        return value
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _coerce_enums(dna: dict[str, Any]) -> None:
    """Canonicalise enum-shaped strings and apply the synonym maps, in place."""
    odd = dna.get("odd")
    if isinstance(odd, dict):
        if "weather" in odd:
            odd["weather"] = _canon(odd["weather"])
        if "lighting" in odd:
            lit = _canon(odd["lighting"])
            odd["lighting"] = _LIGHTING_MAP.get(lit, lit)
        sf = odd.get("sensor_fidelity")
        if isinstance(sf, list):
            odd["sensor_fidelity"] = [_canon(x) for x in sf]

    topo = dna.get("topology")
    if isinstance(topo, dict):
        if "road_type" in topo:
            rt = _canon(topo["road_type"])
            topo["road_type"] = _ROAD_TYPE_MAP.get(rt, rt)
        if "lane_event" in topo:
            topo["lane_event"] = _canon(topo["lane_event"])
        if "intersection_type" in topo:
            topo["intersection_type"] = _canon(topo["intersection_type"])

    actors = dna.get("actor_dynamics")
    if isinstance(actors, list):
        for a in actors:
            if not isinstance(a, dict):
                continue
            if "actor_class" in a:
                ac = _canon(a["actor_class"])
                a["actor_class"] = _ACTOR_CLASS_MAP.get(ac, ac)
            if "state" in a:
                a["state"] = _canon(a["state"])
            if "distance_bucket" in a:
                a["distance_bucket"] = _canon(a["distance_bucket"])

    planner = dna.get("planner_logic")
    if isinstance(planner, dict):
        if "ego_maneuver" in planner:
            planner["ego_maneuver"] = _canon(planner["ego_maneuver"])
        if "risk_level" in planner:
            planner["risk_level"] = _canon(planner["risk_level"])
        se = planner.get("safety_event")
        if isinstance(se, dict):
            if "event_type" in se:
                se["event_type"] = _canon(se["event_type"])
            if isinstance(se.get("collision_type"), str):
                se["collision_type"] = _canon(se["collision_type"])
            if isinstance(se.get("severity_estimate"), str):
                se["severity_estimate"] = _canon(se["severity_estimate"])


def _fallback_enums(dna: dict[str, Any]) -> None:
    """Last-resort repair for enum drift the synonym map misses.

    A *present-but-invalid* required enum is set to a conservative default and
    actors with an unmappable class/state/distance are dropped. An absent field
    is left missing so the doc fails validation and routes to review — never
    defaulted, so an omitted risk_level can't be silently masked as "nominal".
    """
    for (parent, field), (valid, default) in _ENUM_DEFAULTS.items():
        obj = dna.get(parent)
        if isinstance(obj, dict) and field in obj and obj[field] not in valid:
            obj[field] = default

    odd = dna.get("odd")
    if isinstance(odd, dict) and isinstance(odd.get("sensor_fidelity"), list):
        odd["sensor_fidelity"] = [x for x in odd["sensor_fidelity"] if x in _SENSOR_FIDELITY] or [
            "clean"
        ]

    actors = dna.get("actor_dynamics")
    if isinstance(actors, list):
        dna["actor_dynamics"] = [
            a
            for a in actors
            if isinstance(a, dict)
            and a.get("actor_class") in _ACTOR_CLASS
            and a.get("state") in _ACTOR_STATE
            and a.get("distance_bucket") in _DISTANCE_BUCKET
        ]


def _default_safety_event(dna: dict[str, Any]) -> None:
    """Inject the conservative nominal safety_event when the model omitted it."""
    planner = dna.get("planner_logic")
    if not isinstance(planner, dict):
        return
    if not isinstance(planner.get("safety_event"), dict):
        planner["safety_event"] = {
            "has_event": False,
            "event_type": "none",
            "collision_type": None,
            "severity_estimate": None,
        }


def _normalize_safety_event(dna: dict[str, Any]) -> None:
    """Force the schema null-invariants; never invents a missing severity_estimate."""
    planner = dna.get("planner_logic")
    if not isinstance(planner, dict):
        return
    se = planner.get("safety_event")
    if not isinstance(se, dict):
        return
    event_type = se.get("event_type")
    if event_type != "collision":
        se["collision_type"] = None
    if event_type == "none":
        se["severity_estimate"] = None


def _truncate_lengths(dna: dict[str, Any]) -> None:
    sd = dna.get("scene_description")
    if isinstance(sd, str):
        dna["scene_description"] = _truncate(sd, SCENE_DESCRIPTION_MAX)
    planner = dna.get("planner_logic")
    if isinstance(planner, dict):
        rationale = planner.get("risk_level_rationale")
        if isinstance(rationale, str):
            planner["risk_level_rationale"] = _truncate(rationale, RISK_RATIONALE_MAX)


def _truncate(text: str, limit: int) -> str:
    """Truncate to <= *limit*, at the last '.' within the limit if it keeps half the budget."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = window.rfind(".")
    if cut >= limit // 2:
        return text[: cut + 1]
    return window.rstrip()
