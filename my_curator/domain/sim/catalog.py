"""Scenario DNA v0.2 enum -> CARLA 0.9.15 capability catalogs.

Every table is grounded in a measurement of the deployed CARLA 0.9.15 image (2026-08-17),
not in documentation: the blueprint library and ``sensor.camera.rgb`` attributes were
queried off the running server, and the town road networks parsed from the shipped
OpenDRIVE files. Where a measured limit costs fidelity, the entry carries a
``DegradationCode`` and an ``applied`` string naming what was substituted and why.

Each table is exhaustive over its schema enum; ``tests/unit/test_sim_catalog.py``
cross-checks that against ``schemas/scenario_dna_v0_2.schema.json`` so schema drift
breaks the build.
"""

from __future__ import annotations

from typing import NamedTuple

from my_curator.domain.sim.reasons import DegradationCode, ExclusionReason
from my_curator.domain.sim.spec import ControlMode

# --- Towns ---

#: Maps actually loadable in the deployed image (verified via ``get_available_maps()``).
LOADABLE_TOWNS: tuple[str, ...] = (
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town10HD",
)

#: Present as ``.xodr`` but NOT loadable — additional-maps package is not installed.
UNAVAILABLE_TOWNS: tuple[str, ...] = ("Town06", "Town07")

#: Hand-curated: OpenDRIVE junction ids that are roundabouts, keyed by town.
#:
#: Not detected. A geometric detector was tried and discarded — the turning arcs of an
#: ordinary four-arm junction sum to the same total curvature as a ring, so curvature
#: cannot separate them. Town03's junction 861 was identified by extent instead, and it
#: is unambiguous: 30.2 m across against 26.1 m for the next largest, over 5 arms and 21
#: connecting roads totalling 623 m where a normal junction of that town spans 3-4 arms
#: and under 430 m. It is the only roundabout in the loadable map set.
ROUNDABOUT_JUNCTIONS: dict[str, frozenset[int]] = {
    "Town03": frozenset({861}),
}


class TownProfile(NamedTuple):
    """Measured capability of one town (parsed from its OpenDRIVE).

    ``max_driving_lanes`` counts lanes *in one direction* on a single lane section — the
    figure a lane-change maneuver actually needs. Counting a road's lanes across all of
    its lane sections instead, as the first pass did, sums lanes that follow one another
    rather than running side by side and overstates the width of any road that changes
    its cross-section partway along.
    """

    speed_kph: tuple[int, ...]
    max_driving_lanes: int
    lane_types: frozenset[str]
    signalized_junctions: int
    unsignalized_junctions: int


TOWN_PROFILES: dict[str, TownProfile] = {
    "Town01": TownProfile((40,), 1, frozenset({"driving", "sidewalk", "shoulder"}), 12, 0),
    "Town02": TownProfile((40,), 1, frozenset({"driving", "sidewalk", "shoulder"}), 8, 0),
    "Town03": TownProfile(
        (40, 80, 90),
        2,
        frozenset({"driving", "sidewalk", "shoulder", "parking", "bidirectional", "median"}),
        2,
        33,
    ),
    "Town04": TownProfile(
        (50, 60, 90, 100), 4, frozenset({"driving", "sidewalk", "shoulder"}), 4, 23
    ),
    "Town05": TownProfile(
        (60, 90, 100), 3, frozenset({"driving", "sidewalk", "shoulder", "parking"}), 6, 15
    ),
    "Town10HD": TownProfile(
        (60, 80), 2, frozenset({"driving", "sidewalk", "shoulder", "median"}), 2, 7
    ),
}

# --- odd.weather -> carla.WeatherParameters kwargs ---


class WeatherMapping(NamedTuple):
    params: dict[str, float]
    degradation: DegradationCode | None = None
    applied: str = ""


WEATHER: dict[str, WeatherMapping] = {
    "clear": WeatherMapping(
        {
            "cloudiness": 5.0,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wetness": 0.0,
            "fog_density": 0.0,
            "wind_intensity": 5.0,
        }
    ),
    "overcast": WeatherMapping(
        {
            "cloudiness": 85.0,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wetness": 0.0,
            "fog_density": 3.0,
            "wind_intensity": 15.0,
        }
    ),
    "light_rain": WeatherMapping(
        {
            "cloudiness": 65.0,
            "precipitation": 30.0,
            "precipitation_deposits": 30.0,
            "wetness": 40.0,
            "fog_density": 5.0,
            "wind_intensity": 25.0,
        }
    ),
    "heavy_rain": WeatherMapping(
        {
            "cloudiness": 95.0,
            "precipitation": 90.0,
            "precipitation_deposits": 85.0,
            "wetness": 90.0,
            "fog_density": 12.0,
            "wind_intensity": 65.0,
        }
    ),
    "snow": WeatherMapping(
        {
            "cloudiness": 100.0,
            "precipitation": 30.0,
            "precipitation_deposits": 70.0,
            "wetness": 60.0,
            "fog_density": 25.0,
            "wind_intensity": 30.0,
        },
        DegradationCode.WEATHER_NO_SNOW,
        "saturated overcast + heavy ground deposits",
    ),
    "heavy_snow": WeatherMapping(
        {
            "cloudiness": 100.0,
            "precipitation": 60.0,
            "precipitation_deposits": 95.0,
            "wetness": 80.0,
            "fog_density": 55.0,
            "wind_intensity": 60.0,
        },
        DegradationCode.WEATHER_NO_SNOW,
        "saturated overcast + maximal deposits + dense fog",
    ),
    "fog": WeatherMapping(
        {
            "cloudiness": 70.0,
            "precipitation": 0.0,
            "precipitation_deposits": 10.0,
            "wetness": 20.0,
            "fog_density": 85.0,
            "fog_distance": 8.0,
            "fog_falloff": 1.0,
        }
    ),
    "mist": WeatherMapping(
        {
            "cloudiness": 55.0,
            "precipitation": 0.0,
            "precipitation_deposits": 5.0,
            "wetness": 15.0,
            "fog_density": 30.0,
            "fog_distance": 35.0,
            "fog_falloff": 0.5,
        }
    ),
    "sleet": WeatherMapping(
        {
            "cloudiness": 100.0,
            "precipitation": 70.0,
            "precipitation_deposits": 75.0,
            "wetness": 85.0,
            "fog_density": 20.0,
            "wind_intensity": 55.0,
        },
        DegradationCode.WEATHER_NO_SNOW,
        "heavy rain + heavy deposits (no frozen-precipitation channel)",
    ),
}

# --- odd.lighting -> sun altitude ---


class LightingMapping(NamedTuple):
    params: dict[str, float]
    degradation: DegradationCode | None = None
    applied: str = ""


LIGHTING: dict[str, LightingMapping] = {
    "day": LightingMapping({"sun_altitude_angle": 70.0, "sun_azimuth_angle": 250.0}),
    "dawn": LightingMapping({"sun_altitude_angle": 6.0, "sun_azimuth_angle": 90.0}),
    "dusk": LightingMapping({"sun_altitude_angle": -2.0, "sun_azimuth_angle": 270.0}),
    "night": LightingMapping({"sun_altitude_angle": -25.0, "sun_azimuth_angle": 270.0}),
    "overcast_day": LightingMapping({"sun_altitude_angle": 45.0, "sun_azimuth_angle": 200.0}),
    "tunnel": LightingMapping(
        {"sun_altitude_angle": 15.0, "sun_azimuth_angle": 180.0},
        DegradationCode.LIGHTING_APPROXIMATED,
        "low-sun shadowed corridor (no guaranteed tunnel geometry)",
    ),
}

# --- odd.sensor_fidelity -> sensor.camera.rgb attributes ---


class SensorMapping(NamedTuple):
    attributes: dict[str, str]
    degradation: DegradationCode | None = None
    applied: str = ""


SENSOR_FIDELITY: dict[str, SensorMapping] = {
    "clean": SensorMapping({}),
    "lens_flare": SensorMapping({"lens_flare_intensity": "0.6"}),
    "droplets_on_lens": SensorMapping(
        {"lens_circle_multiplier": "1.5", "chromatic_aberration_intensity": "0.6"},
        DegradationCode.SENSOR_EFFECT_UNAVAILABLE,
        "lens distortion + chromatic aberration stand-in (no droplet shader)",
    ),
    "motion_blur": SensorMapping(
        {"motion_blur_intensity": "0.85", "motion_blur_max_distortion": "0.5"}
    ),
    "low_contrast": SensorMapping({"slope": "0.55", "toe": "0.2", "black_clip": "0.05"}),
    "overexposed": SensorMapping({"exposure_compensation": "1.8", "white_clip": "0.7"}),
}

# --- topology.road_type -> road query ---


class RoadMapping(NamedTuple):
    min_driving_lanes: int
    speed_kph_range: tuple[int, int]
    required_lane_types: tuple[str, ...]
    towns: tuple[str, ...]
    target_speed_kph: float
    exclusion: ExclusionReason | None = None
    degradation: DegradationCode | None = None
    applied: str = ""


ROAD_TYPE: dict[str, RoadMapping] = {
    "motorway": RoadMapping(3, (90, 100), ("driving",), ("Town04", "Town05"), 95.0),
    "trunk": RoadMapping(2, (80, 100), ("driving",), ("Town03", "Town04", "Town05"), 80.0),
    "primary": RoadMapping(
        1, (50, 90), ("driving",), ("Town03", "Town04", "Town05", "Town10HD"), 55.0
    ),
    "secondary": RoadMapping(
        1, (40, 60), ("driving",), ("Town01", "Town02", "Town03", "Town10HD"), 45.0
    ),
    # Town03 is included because Town01/Town02 signalize *every* junction they own, so a
    # residential road with an unsignalized junction would otherwise be unstageable.
    "residential": RoadMapping(
        1, (40, 40), ("driving", "sidewalk"), ("Town01", "Town02", "Town03"), 35.0
    ),
    "service": RoadMapping(
        1,
        (40, 40),
        ("driving",),
        ("Town01", "Town02", "Town03"),
        25.0,
        degradation=DegradationCode.ROAD_TYPE_SUBSTITUTED,
        applied="residential grid road (no service-road class in the built-in towns)",
    ),
    "rural": RoadMapping(
        1,
        (40, 60),
        ("driving",),
        ("Town01", "Town02", "Town03"),
        50.0,
        degradation=DegradationCode.ROAD_TYPE_SUBSTITUTED,
        applied="low-speed town road — Town07 (rural) is not in the loadable map set",
    ),
    "parking": RoadMapping(1, (40, 60), ("driving", "parking"), ("Town03", "Town05"), 10.0),
    "walkway": RoadMapping(0, (0, 0), (), (), 0.0, exclusion=ExclusionReason.UNSUPPORTED_ROAD_TYPE),
    "cycling": RoadMapping(0, (0, 0), (), (), 0.0, exclusion=ExclusionReason.UNSUPPORTED_ROAD_TYPE),
}

# --- topology.intersection_type -> junction requirement ---


class IntersectionMapping(NamedTuple):
    towns: tuple[str, ...]
    exclusion: ExclusionReason | None = None
    degradation: DegradationCode | None = None
    applied: str = ""


INTERSECTION_TYPE: dict[str, IntersectionMapping] = {
    "none": IntersectionMapping(LOADABLE_TOWNS),
    "signalized": IntersectionMapping(LOADABLE_TOWNS),
    "unsignalized": IntersectionMapping(("Town03", "Town04", "Town05", "Town10HD")),
    "roundabout": IntersectionMapping(("Town03",)),
    "t_junction": IntersectionMapping(LOADABLE_TOWNS),
    "crosswalk": IntersectionMapping(
        ("Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD")
    ),
    "direct_connection": IntersectionMapping(
        ("Town04",),
        degradation=DegradationCode.INTERSECTION_SUBSTITUTED,
        applied="Town04 interchange only — Town06 (highway town) is not loadable",
    ),
}

# --- topology.lane_event -> static prop dressing ---


class LaneEventMapping(NamedTuple):
    props: tuple[tuple[str, int, str], ...]
    degradation: DegradationCode | None = None
    applied: str = ""


LANE_EVENT: dict[str, LaneEventMapping] = {
    "normal": LaneEventMapping(()),
    "construction_divert": LaneEventMapping(
        (
            ("static.prop.constructioncone", 8, "taper_into_adjacent_lane"),
            ("static.prop.warningconstruction", 1, "lane_head"),
        ),
        DegradationCode.LANE_EVENT_PROP_STAGED,
        "cone taper + warning sign (the road network itself is unchanged)",
    ),
    "lane_closed": LaneEventMapping(
        (
            ("static.prop.trafficcone01", 10, "lane_block"),
            ("static.prop.streetbarrier", 2, "lane_head"),
        ),
        DegradationCode.LANE_EVENT_PROP_STAGED,
        "cone line + barrier across the lane",
    ),
    "merge": LaneEventMapping(()),
    "split": LaneEventMapping(()),
    "unmarked": LaneEventMapping(
        (),
        DegradationCode.LANE_EVENT_PROP_STAGED,
        "marked lane retained — town lane paint is not runtime-editable",
    ),
}

# --- actor_dynamics[].actor_class -> blueprint ---


class ActorMapping(NamedTuple):
    blueprint_filter: str
    is_static_prop: bool = False
    degradation: DegradationCode | None = None
    applied: str = ""


ACTOR_CLASS: dict[str, ActorMapping] = {
    "pedestrian": ActorMapping("walker.pedestrian.*"),
    "cyclist": ActorMapping(
        "vehicle.diamondback.century|vehicle.gazelle.omafiets|vehicle.bh.crossbike"
    ),
    "motorcyclist": ActorMapping(
        "vehicle.harley-davidson.low_rider|vehicle.kawasaki.ninja|vehicle.yamaha.yzf"
    ),
    "vehicle_car": ActorMapping("vehicle.*|base_type=car"),
    "vehicle_van": ActorMapping("vehicle.*|base_type=van"),
    "vehicle_truck": ActorMapping("vehicle.*|base_type=truck"),
    "vehicle_bus": ActorMapping("vehicle.mitsubishi.fusorosa"),
    "vehicle_emergency": ActorMapping("vehicle.*|special_type=emergency"),
    "vehicle_construction": ActorMapping(
        "vehicle.carlamotors.carlacola",
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="cargo truck (no construction-vehicle blueprint in 0.9.15)",
    ),
    "animal": ActorMapping(
        "",
        degradation=DegradationCode.ACTOR_DROPPED,
        applied="omitted — no animal blueprint exists in this build",
    ),
    "debris": ActorMapping(
        "static.prop.dirtdebris01|static.prop.trashbag",
        is_static_prop=True,
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="static debris prop (not a physics actor)",
    ),
    "construction_object": ActorMapping(
        "static.prop.constructioncone|static.prop.warningconstruction", is_static_prop=True
    ),
    "obstacle": ActorMapping(
        "static.prop.trafficwarning|static.prop.streetbarrier",
        is_static_prop=True,
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="generic road-obstacle prop",
    ),
    "standup_scooter_rider": ActorMapping(
        "vehicle.vespa.zx125",
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="scooter blueprint (no stand-up kick-scooter rider exists)",
    ),
    "e_bike_rider": ActorMapping(
        "vehicle.diamondback.century",
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="pedal bicycle (no e-bike blueprint)",
    ),
    "delivery_motorcycle": ActorMapping(
        "vehicle.vespa.zx125",
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="scooter blueprint without delivery box",
    ),
    "wheelchair_user": ActorMapping(
        "walker.pedestrian.*",
        degradation=DegradationCode.ACTOR_BLUEPRINT_SUBSTITUTED,
        applied="walking pedestrian at reduced speed (no wheelchair blueprint)",
    ),
}

# --- actor_dynamics[].state -> maneuver template ---


class StateMapping(NamedTuple):
    template: str
    degradation: DegradationCode | None = None
    applied: str = ""
    #: Default control mode for the state. Only ``oncoming``/``tailing`` are genuinely
    #: reproducible by the traffic manager; everything else either interacts with ego or
    #: must hold a pose an autopilot would immediately abandon.
    control_mode: ControlMode = ControlMode.SCRIPTED


ACTOR_STATE: dict[str, StateMapping] = {
    "crossing": StateMapping("walker_cross_at_crosswalk"),
    "hesitating": StateMapping(
        "walker_cross_stop_go",
        DegradationCode.ACTOR_STATE_APPROXIMATED,
        "scripted stop-go crossing (the built-in walker AI has no hesitation behavior)",
    ),
    "jaywalking": StateMapping("walker_cross_midblock"),
    "cutin": StateMapping("vehicle_lane_change_into_ego_lane"),
    "cutout": StateMapping("vehicle_lane_change_out_of_ego_lane"),
    "stopped": StateMapping("actor_hold_position"),
    "emerging": StateMapping(
        "actor_emerge_from_occluder",
        DegradationCode.ACTOR_STATE_APPROXIMATED,
        "spawned behind the nearest parked vehicle — the original occluder is unknown",
    ),
    "tailing": StateMapping("vehicle_follow_ego", control_mode=ControlMode.AMBIENT),
    "oncoming": StateMapping("vehicle_oncoming_lane", control_mode=ControlMode.AMBIENT),
    "parked": StateMapping("actor_parked_no_autopilot"),
    "static": StateMapping("actor_static"),
}

#: States that make an actor a plausible cause of a safety event, most-implicated first.
#: Used to infer the event actor, because ``causal_trigger_actor_index`` is absent from
#: every row of the deployed corpus.
EVENT_ACTOR_STATE_PRIORITY: tuple[str, ...] = (
    "cutin",
    "jaywalking",
    "emerging",
    "crossing",
    "hesitating",
    "cutout",
    "oncoming",
    "tailing",
    "stopped",
)

DISTANCE_M: dict[str, float] = {"near": 8.0, "mid": 25.0, "far": 60.0}

# --- planner_logic.ego_maneuver -> control template ---


class EgoMapping(NamedTuple):
    template: str
    speed_factor: float
    exclusion: ExclusionReason | None = None
    degradation: DegradationCode | None = None
    applied: str = ""


EGO_MANEUVER: dict[str, EgoMapping] = {
    "cruise": EgoMapping("ego_constant_speed", 1.0),
    "accelerate": EgoMapping("ego_accelerate", 1.0),
    "brake_soft": EgoMapping("ego_brake_soft", 0.9),
    "brake_hard": EgoMapping("ego_brake_hard", 0.9),
    "emergency_brake": EgoMapping("ego_emergency_brake", 1.0),
    "nudge_left": EgoMapping("ego_lateral_offset_left", 0.9),
    "nudge_right": EgoMapping("ego_lateral_offset_right", 0.9),
    "lane_change_left": EgoMapping("ego_lane_change_left", 0.9),
    "lane_change_right": EgoMapping("ego_lane_change_right", 0.9),
    "yield": EgoMapping("ego_yield_at_junction", 0.5),
    "stop": EgoMapping("ego_stop", 0.4),
    "reverse": EgoMapping("ego_reverse", 0.15),
    "swerve": EgoMapping("ego_swerve", 0.9),
}

# --- planner_logic.safety_event ---

#: Contact events are staged up to the pre-impact geometry only — forcing an impact would
#: need trajectory-level fitting, which is out of scope.
COLLISION_STAGED_AS_NEAR_MISS: frozenset[str] = frozenset({"collision"})

SAFETY_EVENT_TYPES: frozenset[str] = frozenset(
    {"none", "near_miss", "hard_brake", "evasive_swerve", "collision"}
)
COLLISION_TYPES: frozenset[str] = frozenset(
    {"head_on", "rear_end", "t_bone", "sideswipe", "single_vehicle", "vru_struck", "none"}
)
SEVERITY_ESTIMATES: frozenset[str] = frozenset({"no_harm", "minor", "major", "fatal"})
RISK_LEVELS: frozenset[str] = frozenset({"nominal", "elevated", "critical"})
DISTANCE_BUCKETS: frozenset[str] = frozenset(DISTANCE_M)
