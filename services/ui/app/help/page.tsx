"use client";

import { useEffect, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  Cloud,
  Eye,
  Layers,
  Map,
  Navigation,
  Users,
  Zap,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

type EnumRow = { value: string; desc: string };
type FieldDef = { name: string; type: string; description: string; source?: string; rows: EnumRow[] };

// ── EN data ───────────────────────────────────────────────────────────────────

const TOC_ITEMS_EN = [
  { id: "overview",       label: "Platform Overview" },
  { id: "workflow",       label: "How to Use" },
  { id: "odd",            label: "Layer 1 · ODD" },
  { id: "topology",       label: "Layer 2 · Topology" },
  { id: "actors",         label: "Layer 3 · Actors" },
  { id: "planner",        label: "Layer 4 · Planner" },
  { id: "review-states",  label: "Review States" },
  { id: "glossary",       label: "Glossary" },
];

const ODD_FIELDS_EN: FieldDef[] = [
  {
    name: "weather",
    type: "enum · required",
    description: "Precipitation and atmospheric state at clip time.",
    source: "ASAM OpenSCENARIO v1.0 CloudState + ASAM OSI PrecipitationIntensity",
    rows: [
      { value: "clear",       desc: "No precipitation, clear sky" },
      { value: "overcast",    desc: "Overcast sky, no precipitation" },
      { value: "light_rain",  desc: "Light rainfall" },
      { value: "heavy_rain",  desc: "Heavy rainfall" },
      { value: "snow",        desc: "Moderate snowfall" },
      { value: "heavy_snow",  desc: "Heavy snowfall" },
      { value: "fog",         desc: "Fog — low horizontal visibility" },
      { value: "mist",        desc: "Mist — reduced visibility, less severe than fog" },
      { value: "sleet",       desc: "Sleet — mixed rain, snow, and ice" },
    ],
  },
  {
    name: "lighting",
    type: "enum · required",
    description: "Ambient illumination category. Dawn and dusk are operationally distinct from day: low-angle sun and long shadows create sensor challenges absent in full daylight. overcast_day (~100 lx) differs from day (~10,000 lx) by reduced contrast and absence of shadows.",
    source: "ASAM OSI AmbientIllumination lux-level buckets",
    rows: [
      { value: "day",          desc: "Full daylight (~10,000 lx)" },
      { value: "dawn",         desc: "Sunrise transition (~1–100 lx) — low-angle sun, long shadows" },
      { value: "dusk",         desc: "Sunset transition (~1–100 lx) — low-angle sun, long shadows" },
      { value: "night",        desc: "Darkness — artificial or no illumination (<1 lx)" },
      { value: "tunnel",       desc: "Enclosed road with artificial lighting — rapid illumination transition on entry/exit" },
      { value: "overcast_day", desc: "Daylight under overcast cloud cover (~100 lx) — reduced contrast, no shadows" },
    ],
  },
  {
    name: "sensor_fidelity",
    type: "enum list",
    description: "Active degradation modes affecting the camera sensor output. Zero or more values; duplicates not allowed. An empty array means no degradation detected.",
    source: "Domain-specific (Korean urban road coverage)",
    rows: [
      { value: "clean",             desc: "No degradation detected" },
      { value: "lens_flare",        desc: "Bright light source causing lens flare artifacts" },
      { value: "droplets_on_lens",  desc: "Water droplets or condensation on the camera lens" },
      { value: "motion_blur",       desc: "Temporal blur from camera or ego-vehicle motion" },
      { value: "low_contrast",      desc: "Reduced image contrast" },
      { value: "overexposed",       desc: "Overexposure — blown-out highlight regions" },
    ],
  },
];

const TOPOLOGY_FIELDS_EN: FieldDef[] = [
  {
    name: "road_type",
    type: "enum · required",
    description: "Road classification by purpose and traffic characteristics. Based on OpenDRIVE e_roadType. 'highway' was renamed to 'motorway' for ASAM standard alignment; 'urban' was split into 'primary' and 'secondary' for finer granularity.",
    source: "OpenDRIVE v1.5M XSD e_roadType",
    rows: [
      { value: "motorway",    desc: "Controlled-access highway — limited entry/exit points (formerly 'highway')" },
      { value: "trunk",       desc: "Major national or inter-city road (high-speed, non-motorway)" },
      { value: "primary",     desc: "Primary urban or inter-city road — controlled traffic flow" },
      { value: "secondary",   desc: "Secondary urban road — moderate traffic density (formerly part of 'urban')" },
      { value: "residential", desc: "Residential street — low speed, mixed pedestrian and vehicle use" },
      { value: "service",     desc: "Service road — parking lots, back alleys, delivery access roads" },
      { value: "rural",       desc: "Rural or unclassified road outside urban areas" },
      { value: "parking",     desc: "Parking lot or structured parking area" },
      { value: "walkway",     desc: "Pedestrian walkway — no motor vehicles permitted" },
      { value: "cycling",     desc: "Bicycle-dedicated path" },
    ],
  },
  {
    name: "lane_event",
    type: "enum · required",
    description: "Lane configuration or temporary alteration at the clip location.",
    source: "Domain-derived from traffic management conventions",
    rows: [
      { value: "normal",              desc: "Standard lane markings, no temporary changes" },
      { value: "construction_divert", desc: "Construction zone with active lane diversion" },
      { value: "lane_closed",         desc: "One or more lanes closed to traffic" },
      { value: "merge",               desc: "Lanes merging — two lanes converge into one" },
      { value: "split",               desc: "Lanes splitting — one lane diverges into two" },
      { value: "unmarked",            desc: "Lanes physically present but with no visible markings" },
    ],
  },
  {
    name: "intersection_type",
    type: "enum · required",
    description: "'direct_connection' specifically refers to highway on/off-ramp merge zones (OpenDRIVE direct junction) — not a generic road connection. 'crosswalk' marks a clip centred on a designated pedestrian crossing.",
    source: "OpenDRIVE v1.5M junction types",
    rows: [
      { value: "none",              desc: "No intersection — straight road section" },
      { value: "signalized",        desc: "Traffic-signal controlled intersection" },
      { value: "unsignalized",      desc: "Uncontrolled intersection (yield or stop signs only)" },
      { value: "roundabout",        desc: "Circular intersection with yield-on-entry rule" },
      { value: "t_junction",        desc: "T-shaped three-way intersection" },
      { value: "crosswalk",         desc: "Designated pedestrian crossing (zebra crossing)" },
      { value: "direct_connection", desc: "Highway on/off-ramp — direct merge zone (OpenDRIVE direct junction)" },
    ],
  },
];

const ACTOR_FIELDS_EN: FieldDef[] = [
  {
    name: "actor_class",
    type: "enum · per actor",
    description: "Object classification by type and role. Korean urban-specific classes (e_bike_rider, delivery_motorcycle, standup_scooter_rider) are retained for local road coverage. 'vehicle_sedan' / 'vehicle_suv' were consolidated into 'vehicle_car' per ASAM OSI TYPE_CAR.",
    source: "ASAM OSI VehicleClassification + PedestrianCategory + MiscObjectCategory",
    rows: [
      { value: "pedestrian",           desc: "Foot-walking person" },
      { value: "cyclist",              desc: "Bicycle rider" },
      { value: "motorcyclist",         desc: "Motorcycle or motorbike rider" },
      { value: "standup_scooter_rider",desc: "E-scooter rider (standing position)" },
      { value: "e_bike_rider",         desc: "Electric bicycle rider" },
      { value: "delivery_motorcycle",  desc: "Delivery motorcycle or scooter (food/parcel courier)" },
      { value: "wheelchair_user",      desc: "Wheelchair occupant" },
      { value: "vehicle_car",          desc: "Passenger car — sedan, SUV, hatchback (ASAM OSI TYPE_CAR)" },
      { value: "vehicle_van",          desc: "Van or minivan (passenger or light cargo)" },
      { value: "vehicle_truck",        desc: "Truck — pickup truck or heavy goods vehicle" },
      { value: "vehicle_bus",          desc: "Bus — public transit or coach" },
      { value: "vehicle_emergency",    desc: "Emergency vehicle — ambulance, police car, fire truck" },
      { value: "vehicle_construction", desc: "Construction machine — excavator, loader, road roller" },
      { value: "animal",               desc: "Animal on or near the roadway (dog, cat, livestock)" },
      { value: "debris",               desc: "Fallen debris or foreign objects on the road" },
      { value: "construction_object",  desc: "Construction equipment, barriers, or site signage" },
      { value: "obstacle",             desc: "Generic obstacle — rock, fallen branch, pothole" },
    ],
  },
  {
    name: "state",
    type: "enum · per actor",
    description: "Behavioral state or motion intent of the detected actor at clip time.",
    source: "Domain-derived from driving scenario analysis",
    rows: [
      { value: "crossing",   desc: "Actively crossing the ego's path or travel lane" },
      { value: "hesitating", desc: "Slowed or paused — uncertain or interrupted motion" },
      { value: "jaywalking", desc: "Crossing the road without a designated crosswalk or signal" },
      { value: "cutin",      desc: "Lateral insertion into ego's lane from the side" },
      { value: "cutout",     desc: "Lateral departure from ego's lane to the side" },
      { value: "stopped",    desc: "Stopped — may resume motion (e.g. waiting at a signal)" },
      { value: "emerging",   desc: "Appearing into the scene from occlusion or off-screen" },
      { value: "tailing",    desc: "Following ego vehicle at close distance" },
      { value: "oncoming",   desc: "Approaching ego head-on in the opposite travel direction" },
      { value: "parked",     desc: "Parked — not moving, occupying a stationary position" },
      { value: "static",     desc: "Immobile object or permanently stationary actor" },
    ],
  },
  {
    name: "distance_bucket",
    type: "enum · per actor",
    description: "Spatial proximity to the ego vehicle, binned into three tiers for threat prioritisation.",
    source: "Domain-derived (threat-zone bucketing)",
    rows: [
      { value: "near", desc: "0 – 10 m — immediate threat zone" },
      { value: "mid",  desc: "10 – 50 m — reaction window" },
      { value: "far",  desc: "> 50 m — background context" },
    ],
  },
];

const PLANNER_FIELDS_EN: FieldDef[] = [
  {
    name: "ego_maneuver",
    type: "enum · required",
    description: "Primary driving maneuver executed or intended by the ego vehicle. 'emergency_brake' = AEB/panic stop (maximum deceleration, automated trigger) — distinct from 'brake_hard' (intentional high-g deceleration). 'swerve' = lateral avoidance that does not complete a full lane change.",
    source: "WOD-E2E arXiv:2510.26125 + PEGASUS HAD-F maneuver taxonomy",
    rows: [
      { value: "cruise",            desc: "Constant speed, no active maneuver" },
      { value: "accelerate",        desc: "Intentional speed increase" },
      { value: "brake_soft",        desc: "Gentle, gradual deceleration" },
      { value: "brake_hard",        desc: "Intentional hard deceleration (high g)" },
      { value: "emergency_brake",   desc: "AEB / panic stop — maximum deceleration, automated trigger" },
      { value: "nudge_left",        desc: "Small lateral adjustment to the left without changing lanes" },
      { value: "nudge_right",       desc: "Small lateral adjustment to the right without changing lanes" },
      { value: "lane_change_left",  desc: "Complete lane change to the left" },
      { value: "lane_change_right", desc: "Complete lane change to the right" },
      { value: "yield",             desc: "Slowing or pausing to give priority to another actor" },
      { value: "stop",              desc: "Full stop — vehicle at rest with no motion" },
      { value: "reverse",           desc: "Backward motion" },
      { value: "swerve",            desc: "Lateral avoidance maneuver — does not complete a full lane change" },
    ],
  },
  {
    name: "risk_level",
    type: "enum · required",
    description: "Scenario risk classification per ISO 21448 SOTIF. Drives the Review Queue priority — 'critical' clips surface first. The DNA pass rate metric counts approved / (approved + rejected), excluding pending and schema_invalid. In v0.2 it is paired with risk_level_rationale — a one-sentence free-text justification anchored to the ISO 21448 C×S (controllability × severity) decision, capped at 300 chars (the UI marks a truncated value with an ellipsis).",
    source: "ISO 21448:2022 SOTIF",
    rows: [
      { value: "nominal",  desc: "No safety concern — absence of unreasonable risk (normal operation)" },
      { value: "elevated", desc: "Tolerable risk — hazard present but mitigation is in place" },
      { value: "critical", desc: "Unreasonable risk (SOTIF trigger) — intervention or override required" },
    ],
  },
  {
    name: "safety_event",
    type: "object · required (v0.2)",
    description: "Generic safety-event channel added in v0.2. has_event (bool) gates the record; when true the clip-detail view renders a Safety Event card. collision_type is one of head_on / rear_end / t_bone / sideswipe / single_vehicle / vru_struck / none (null unless event_type is 'collision'). severity_estimate is no_harm / minor / major / fatal (null when there is no event). event_type enumerates the channel:",
    source: "ISO 21448:2022 SOTIF + Abbreviated Injury Scale (AIS) severity proxy",
    rows: [
      { value: "none",           desc: "No safety-relevant event in the clip (the nominal majority)" },
      { value: "near_miss",      desc: "Near-collision — conflict resolved without physical contact" },
      { value: "hard_brake",     desc: "Hard braking event indicating a safety-relevant situation" },
      { value: "evasive_swerve", desc: "Evasive steering maneuver to avoid a hazard" },
      { value: "collision",      desc: "Physical contact occurred — collision_type is then populated" },
    ],
  },
];

const REVIEW_STATES_EN = [
  {
    value: "pending",
    dot: "bg-amber-400 animate-pulse",
    badge: "text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/25",
    desc: "Awaiting human review. All newly ingested clips start in this state.",
  },
  {
    value: "approved",
    dot: "bg-green-400",
    badge: "text-green-600 dark:text-green-400 bg-green-500/10 border-green-500/25",
    desc: "Accepted into the curated corpus. DNA payload has been verified by a reviewer.",
  },
  {
    value: "rejected",
    dot: "bg-red-400",
    badge: "text-red-600 dark:text-red-400 bg-red-500/10 border-red-500/25",
    desc: "Manually rejected by a human reviewer. Excluded from the training corpus.",
  },
  {
    value: "rejected_schema_invalid",
    dot: "bg-faint",
    badge: "text-muted bg-faint/10 border-line/40",
    desc: "Automatically rejected at ingestion time — the DNA payload failed JSON Schema validation. Displayed as 'Schema Invalid' in the UI.",
  },
];

const GLOSSARY_EN = [
  { term: "Verify-by-Exception (VBE)", def: "A curation strategy where clips pass through automatically unless flagged. Reviewers focus effort only on uncertain or high-risk cases, dramatically reducing manual load on nominal clips." },
  { term: "Scenario DNA", def: "A 4-layer structured descriptor (ODD + Topology + Actor Dynamics + Planner Logic) attached to every clip. Stored as JSONB in PostgreSQL and indexed in Milvus. Schema version: v0.2.0 — adds a free-text scene_description plus planner_logic.risk_level_rationale and a safety_event channel over v0.1." },
  { term: "scene_description", def: "v0.2 free-text VLM narrative (a few sentences, AV-safety-expert voice) authored by the Scout from the video — not derived from the structured fields. Shown at the top of the clip-detail view and used as a high-signal input to the text embedding. Capped at 500 chars." },
  { term: "ODD (Operational Design Domain)", def: "The specific conditions under which an AV system is designed to operate safely (ISO 22736). In My-Curator, ODD covers weather, lighting, and sensor fidelity." },
  { term: "Scout", def: "The VLM that generates Scenario DNA from video frames. Current model: Cosmos-Reason2-8B FP8. Multiple Scout samples per clip are aggregated by BestOfN Aggregator using a symbolic reward signal." },
  { term: "Hybrid Search", def: "Retrieval combining Milvus ANN vector search (Cosmos-Embed1-336p, 768-dim, cosine / inner product on L2-normalised vectors) with PostgreSQL JSONB GIN filter on DNA fields. ANN top-1000 candidates are re-ranked by exact filter matching." },
  { term: "DNA Pass Rate", def: "Approved / (Approved + Rejected). Excludes pending and schema_invalid states. Shown on the Dashboard as a percentage." },
  { term: "SOTIF (ISO 21448)", def: "Safety of the Intended Functionality — ISO standard defining risk categories for AV systems. My-Curator's risk_level enum maps directly: nominal → no unreasonable risk, elevated → tolerable risk, critical → unreasonable risk trigger." },
  { term: "dna_version", def: "Schema version lock ('0.2.0'). Any schema change bumps this value and triggers a full prompt_regression + schema test run." },
  { term: "causal_trigger_actor_index", def: "Index into actor_dynamics[] identifying which actor caused the ego maneuver. Null in current single-Scout deployments — the P4-6 Judge (Qwen3-8B-AWQ) is report-only and does not populate it." },
  { term: "grounded_by_yolo26", def: "Boolean per actor. True if YOLO26 object detection independently confirmed the actor's presence, reducing hallucination risk for that actor entry." },
  { term: "hallucination_flags", def: "Array of field-name strings in the confidence block flagging fields where the Scout may have fabricated values. Used to surface low-confidence DNA regions for reviewer attention." },
];

// ── KO data ───────────────────────────────────────────────────────────────────

const TOC_ITEMS_KO = [
  { id: "overview",       label: "플랫폼 개요" },
  { id: "workflow",       label: "사용 방법" },
  { id: "odd",            label: "Layer 1 · ODD" },
  { id: "topology",       label: "Layer 2 · 토폴로지" },
  { id: "actors",         label: "Layer 3 · 액터" },
  { id: "planner",        label: "Layer 4 · 플래너" },
  { id: "review-states",  label: "리뷰 상태" },
  { id: "glossary",       label: "용어 설명" },
];

const ODD_FIELDS_KO: FieldDef[] = [
  {
    name: "weather",
    type: "enum · 필수",
    description: "클립 촬영 시점의 강수량 및 대기 상태.",
    source: "ASAM OpenSCENARIO v1.0 CloudState + ASAM OSI PrecipitationIntensity",
    rows: [
      { value: "clear",       desc: "강수 없음, 맑은 하늘" },
      { value: "overcast",    desc: "흐린 하늘, 강수 없음" },
      { value: "light_rain",  desc: "가벼운 비" },
      { value: "heavy_rain",  desc: "강한 비" },
      { value: "snow",        desc: "보통 눈" },
      { value: "heavy_snow",  desc: "강한 눈" },
      { value: "fog",         desc: "안개 — 낮은 수평 가시거리" },
      { value: "mist",        desc: "박무 — 안개보다 약한 가시거리 저하" },
      { value: "sleet",       desc: "진눈깨비 — 비, 눈, 얼음 혼합" },
    ],
  },
  {
    name: "lighting",
    type: "enum · 필수",
    description: "주변 조도 범주. 새벽과 황혼은 낮은 태양 각도와 긴 그림자로 완전한 주간과 구분되는 센서 도전을 유발함. overcast_day(~100 lx)는 day(~10,000 lx)와 대비 저하 및 그림자 소실 면에서 구분됨.",
    source: "ASAM OSI AmbientIllumination lux 구간",
    rows: [
      { value: "day",          desc: "완전한 주간 조명 (~10,000 lx)" },
      { value: "dawn",         desc: "일출 전환기 (~1–100 lx) — 낮은 태양 각도, 긴 그림자" },
      { value: "dusk",         desc: "일몰 전환기 (~1–100 lx) — 낮은 태양 각도, 긴 그림자" },
      { value: "night",        desc: "야간 — 인공 조명 또는 무조명 (<1 lx)" },
      { value: "tunnel",       desc: "인공 조명 터널 — 진입/출구 시 급격한 조도 전환" },
      { value: "overcast_day", desc: "흐린 낮 (~100 lx) — 낮은 대비, 그림자 없음" },
    ],
  },
  {
    name: "sensor_fidelity",
    type: "enum 목록",
    description: "카메라 센서 출력에 영향을 미치는 활성 열화 모드. 0개 이상의 값; 중복 불가. 빈 배열은 열화 없음을 의미.",
    source: "도메인 특화 (한국 도시 도로 커버리지)",
    rows: [
      { value: "clean",             desc: "열화 없음" },
      { value: "lens_flare",        desc: "강한 광원으로 인한 렌즈 플레어 현상" },
      { value: "droplets_on_lens",  desc: "카메라 렌즈의 수분 또는 응결" },
      { value: "motion_blur",       desc: "카메라 또는 자차 움직임으로 인한 시간적 블러" },
      { value: "low_contrast",      desc: "이미지 대비 저하" },
      { value: "overexposed",       desc: "과노출 — 하이라이트 영역 날아감" },
    ],
  },
];

const TOPOLOGY_FIELDS_KO: FieldDef[] = [
  {
    name: "road_type",
    type: "enum · 필수",
    description: "목적 및 교통 특성에 따른 도로 분류. OpenDRIVE e_roadType 기반. 'highway'는 ASAM 표준 정렬을 위해 'motorway'로 변경; 'urban'은 세밀한 구분을 위해 'primary'와 'secondary'로 분리.",
    source: "OpenDRIVE v1.5M XSD e_roadType",
    rows: [
      { value: "motorway",    desc: "제한 접근 고속도로 — 제한된 진출입 지점 (구 'highway')" },
      { value: "trunk",       desc: "주요 국도 또는 도시 간 도로 (고속, 일반 국도)" },
      { value: "primary",     desc: "주요 도시 또는 도시 간 도로 — 통제된 교통 흐름" },
      { value: "secondary",   desc: "2차 도시 도로 — 중간 교통 밀도 (구 'urban' 일부)" },
      { value: "residential", desc: "주거 도로 — 저속, 보행자와 차량 혼용" },
      { value: "service",     desc: "서비스 도로 — 주차장, 골목길, 배달 접근로" },
      { value: "rural",       desc: "도시 외부 농촌 또는 미분류 도로" },
      { value: "parking",     desc: "주차장 또는 구조화된 주차 구역" },
      { value: "walkway",     desc: "보행자 전용 통로 — 차량 진입 불가" },
      { value: "cycling",     desc: "자전거 전용 도로" },
    ],
  },
  {
    name: "lane_event",
    type: "enum · 필수",
    description: "클립 위치의 차선 구성 또는 임시 변경 상태.",
    source: "교통 관리 관례에서 도출된 도메인 분류",
    rows: [
      { value: "normal",              desc: "표준 차선 표시, 임시 변경 없음" },
      { value: "construction_divert", desc: "공사 구간으로 차선 우회 중" },
      { value: "lane_closed",         desc: "하나 이상의 차선 폐쇄" },
      { value: "merge",               desc: "차선 합류 — 두 차선이 하나로 합쳐짐" },
      { value: "split",               desc: "차선 분기 — 하나의 차선이 둘로 나뉨" },
      { value: "unmarked",            desc: "물리적으로 차선이 있으나 표시 없음" },
    ],
  },
  {
    name: "intersection_type",
    type: "enum · 필수",
    description: "'direct_connection'은 고속도로 진출입 합류 구간(OpenDRIVE direct junction)을 의미 — 일반적인 도로 연결이 아님. 'crosswalk'는 지정 횡단보도가 중심인 클립.",
    source: "OpenDRIVE v1.5M 교차로 유형",
    rows: [
      { value: "none",              desc: "교차로 없음 — 직선 도로 구간" },
      { value: "signalized",        desc: "신호 제어 교차로" },
      { value: "unsignalized",      desc: "비신호 교차로 (양보 또는 정지 표지만)" },
      { value: "roundabout",        desc: "로터리 — 진입 시 양보 규칙 적용" },
      { value: "t_junction",        desc: "T자형 삼거리 교차로" },
      { value: "crosswalk",         desc: "지정 횡단보도 (zebra crossing)" },
      { value: "direct_connection", desc: "고속도로 진출입로 — 직접 합류 구간 (OpenDRIVE direct junction)" },
    ],
  },
];

const ACTOR_FIELDS_KO: FieldDef[] = [
  {
    name: "actor_class",
    type: "enum · 액터별",
    description: "유형 및 역할에 따른 객체 분류. 한국 도시 특화 클래스(e_bike_rider, delivery_motorcycle, standup_scooter_rider)는 국내 도로 커버리지를 위해 유지. 'vehicle_sedan'/'vehicle_suv'는 ASAM OSI TYPE_CAR에 따라 'vehicle_car'로 통합.",
    source: "ASAM OSI VehicleClassification + PedestrianCategory + MiscObjectCategory",
    rows: [
      { value: "pedestrian",           desc: "도보 보행자" },
      { value: "cyclist",              desc: "자전거 탑승자" },
      { value: "motorcyclist",         desc: "오토바이 탑승자" },
      { value: "standup_scooter_rider",desc: "전동 킥보드 탑승자 (기립 자세)" },
      { value: "e_bike_rider",         desc: "전동 자전거 탑승자" },
      { value: "delivery_motorcycle",  desc: "배달 오토바이 또는 스쿠터 (음식/소포 배달)" },
      { value: "wheelchair_user",      desc: "휠체어 이용자" },
      { value: "vehicle_car",          desc: "승용차 — 세단, SUV, 해치백 (ASAM OSI TYPE_CAR)" },
      { value: "vehicle_van",          desc: "승합차 또는 미니밴 (승객 또는 소화물)" },
      { value: "vehicle_truck",        desc: "트럭 — 픽업트럭 또는 화물차" },
      { value: "vehicle_bus",          desc: "버스 — 대중교통 또는 코치" },
      { value: "vehicle_emergency",    desc: "긴급 차량 — 구급차, 경찰차, 소방차" },
      { value: "vehicle_construction", desc: "건설 장비 — 굴착기, 로더, 도로 롤러" },
      { value: "animal",               desc: "도로 위 또는 근처의 동물 (개, 고양이, 가축)" },
      { value: "debris",               desc: "도로 위 낙하물 또는 이물질" },
      { value: "construction_object",  desc: "건설 장비, 바리케이드, 또는 공사 표지판" },
      { value: "obstacle",             desc: "일반 장애물 — 돌, 낙하 가지, 포트홀" },
    ],
  },
  {
    name: "state",
    type: "enum · 액터별",
    description: "클립 시점에서 감지된 액터의 행동 상태 또는 움직임 의도.",
    source: "주행 시나리오 분석에서 도출된 도메인 분류",
    rows: [
      { value: "crossing",   desc: "자차의 경로 또는 주행 차선을 적극적으로 횡단 중" },
      { value: "hesitating", desc: "감속 또는 정지 — 불확실하거나 중단된 움직임" },
      { value: "jaywalking", desc: "지정 횡단보도나 신호 없이 도로 횡단 중" },
      { value: "cutin",      desc: "측면에서 자차 차선으로 끼어들기" },
      { value: "cutout",     desc: "자차 차선에서 측면으로 빠져나가기" },
      { value: "stopped",    desc: "정지 — 움직임 재개 가능 (예: 신호 대기)" },
      { value: "emerging",   desc: "사각지대 또는 화면 밖에서 장면으로 나타남" },
      { value: "tailing",    desc: "근접 거리에서 자차를 따라오는 중" },
      { value: "oncoming",   desc: "반대 주행 방향에서 정면으로 접근 중" },
      { value: "parked",     desc: "주차 중 — 움직임 없이 고정 위치 점유" },
      { value: "static",     desc: "움직이지 않는 객체 또는 영구 정지 액터" },
    ],
  },
  {
    name: "distance_bucket",
    type: "enum · 액터별",
    description: "위협 우선순위화를 위해 세 단계로 구분한 자차와의 공간적 근접도.",
    source: "도메인 도출 (위협 구역 구분)",
    rows: [
      { value: "near", desc: "0 – 10 m — 즉각적 위협 구역" },
      { value: "mid",  desc: "10 – 50 m — 반응 가능 거리" },
      { value: "far",  desc: "> 50 m — 배경 맥락" },
    ],
  },
];

const PLANNER_FIELDS_KO: FieldDef[] = [
  {
    name: "ego_maneuver",
    type: "enum · 필수",
    description: "자차가 실행하거나 의도한 주요 주행 조작. 'emergency_brake' = AEB/패닉 제동 (최대 감속, 자동 트리거) — 'brake_hard'(의도적 고g 감속)와 구분. 'swerve' = 차선 변경을 완료하지 않는 측면 회피.",
    source: "WOD-E2E arXiv:2510.26125 + PEGASUS HAD-F 조작 분류 체계",
    rows: [
      { value: "cruise",            desc: "정속 주행, 활성 조작 없음" },
      { value: "accelerate",        desc: "의도적 속도 증가" },
      { value: "brake_soft",        desc: "부드럽고 점진적인 감속" },
      { value: "brake_hard",        desc: "의도적 강한 감속 (고g)" },
      { value: "emergency_brake",   desc: "AEB / 패닉 제동 — 최대 감속, 자동 트리거" },
      { value: "nudge_left",        desc: "차선 변경 없이 좌측으로 소폭 조정" },
      { value: "nudge_right",       desc: "차선 변경 없이 우측으로 소폭 조정" },
      { value: "lane_change_left",  desc: "좌측으로 완전한 차선 변경" },
      { value: "lane_change_right", desc: "우측으로 완전한 차선 변경" },
      { value: "yield",             desc: "다른 액터에게 양보하기 위해 감속 또는 정지" },
      { value: "stop",              desc: "완전 정지 — 차량 정지 상태" },
      { value: "reverse",           desc: "후진 주행" },
      { value: "swerve",            desc: "측면 회피 조작 — 완전한 차선 변경 미완료" },
    ],
  },
  {
    name: "risk_level",
    type: "enum · 필수",
    description: "ISO 21448 SOTIF 기반 시나리오 위험 분류. Review Queue 우선순위를 결정 — 'critical' 클립이 먼저 표시됨. DNA 합격률은 approved / (approved + rejected)로 계산하며 pending 및 schema_invalid 제외. v0.2에서는 risk_level_rationale과 짝을 이룸 — ISO 21448 C×S(제어가능성 × 심각도) 판단에 근거한 한 문장 자유서술 근거로, 최대 300자(초과 시 UI가 말줄임표로 표시).",
    source: "ISO 21448:2022 SOTIF",
    rows: [
      { value: "nominal",  desc: "안전 우려 없음 — 불합리한 위험 없음 (정상 운행)" },
      { value: "elevated", desc: "허용 가능한 위험 — 위험 요소 존재하나 완화 조치 있음" },
      { value: "critical", desc: "불합리한 위험 (SOTIF 트리거) — 개입 또는 제어권 회수 필요" },
    ],
  },
  {
    name: "safety_event",
    type: "object · 필수 (v0.2)",
    description: "v0.2에서 추가된 범용 안전 이벤트 채널. has_event(bool)가 기록 여부를 결정하며, true일 때 클립 상세에 Safety Event 카드가 표시됨. collision_type은 head_on / rear_end / t_bone / sideswipe / single_vehicle / vru_struck / none 중 하나(event_type이 'collision'이 아니면 null). severity_estimate은 no_harm / minor / major / fatal(이벤트 없으면 null). event_type은 다음 채널을 열거함:",
    source: "ISO 21448:2022 SOTIF + Abbreviated Injury Scale (AIS) 심각도 프록시",
    rows: [
      { value: "none",           desc: "클립에 안전 관련 이벤트 없음 (nominal 다수)" },
      { value: "near_miss",      desc: "니어미스 — 물리적 접촉 없이 충돌 상황 해소" },
      { value: "hard_brake",     desc: "안전 관련 상황을 나타내는 급제동 이벤트" },
      { value: "evasive_swerve", desc: "위험 회피를 위한 급조향 조작" },
      { value: "collision",      desc: "물리적 접촉 발생 — 이때 collision_type이 채워짐" },
    ],
  },
];

const REVIEW_STATES_KO = [
  {
    value: "pending",
    dot: "bg-amber-400 animate-pulse",
    badge: "text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/25",
    desc: "사람의 검토 대기 중. 새로 수집된 모든 클립의 초기 상태.",
  },
  {
    value: "approved",
    dot: "bg-green-400",
    badge: "text-green-600 dark:text-green-400 bg-green-500/10 border-green-500/25",
    desc: "큐레이션 코퍼스에 승인됨. 검토자가 DNA 페이로드를 확인함.",
  },
  {
    value: "rejected",
    dot: "bg-red-400",
    badge: "text-red-600 dark:text-red-400 bg-red-500/10 border-red-500/25",
    desc: "검토자가 수동으로 거부함. 학습 코퍼스에서 제외됨.",
  },
  {
    value: "rejected_schema_invalid",
    dot: "bg-faint",
    badge: "text-muted bg-faint/10 border-line/40",
    desc: "수집 시 자동 거부 — DNA 페이로드가 JSON Schema 검증에 실패함. UI에서 'Schema Invalid'로 표시됨.",
  },
];

const GLOSSARY_KO = [
  { term: "Verify-by-Exception (VBE)", def: "예외 기반 검증 전략. 클립이 플래그되지 않으면 자동으로 통과하며, 검토자는 불확실하거나 고위험 케이스에만 집중하여 일반 클립의 수동 작업량을 크게 줄임." },
  { term: "Scenario DNA", def: "모든 클립에 부착된 4계층 구조 설명자 (ODD + 토폴로지 + 액터 다이내믹스 + 플래너 로직). PostgreSQL에 JSONB로 저장되고 Milvus에 인덱싱됨. 스키마 버전: v0.2.0 — v0.1 대비 자유서술 scene_description과 planner_logic.risk_level_rationale, safety_event 채널이 추가됨." },
  { term: "scene_description", def: "v0.2 자유서술 VLM 내러티브(AV 안전 전문가 어조의 몇 문장)로, 구조화 필드에서 파생되지 않고 Scout이 영상에서 직접 작성함. 클립 상세 상단에 표시되며 텍스트 임베딩의 고신호 입력으로 사용됨. 최대 500자." },
  { term: "ODD (Operational Design Domain)", def: "AV 시스템이 안전하게 작동하도록 설계된 특정 조건 (ISO 22736). My-Curator에서 ODD는 날씨, 조도, 센서 피델리티를 다룸." },
  { term: "Scout", def: "비디오 프레임에서 Scenario DNA를 생성하는 VLM. 현재 모델: Cosmos-Reason2-8B FP8. 클립당 여러 Scout 샘플이 심볼릭 보상 신호를 사용하는 BestOfN Aggregator에 의해 집계됨." },
  { term: "Hybrid Search", def: "Milvus ANN 벡터 검색 (Cosmos-Embed1-336p, 768차원, L2 정규화 벡터의 코사인/내적)과 PostgreSQL JSONB GIN 필터를 결합한 검색. ANN 상위 1,000개 후보가 정확한 필터 매칭으로 재순위 결정됨." },
  { term: "DNA Pass Rate", def: "Approved / (Approved + Rejected). pending 및 schema_invalid 상태 제외. 대시보드에 백분율로 표시됨." },
  { term: "SOTIF (ISO 21448)", def: "의도 기능 안전성 — AV 시스템의 위험 범주를 정의하는 ISO 표준. My-Curator의 risk_level 열거형이 직접 매핑: nominal → 불합리한 위험 없음, elevated → 허용 가능한 위험, critical → 불합리한 위험 트리거." },
  { term: "dna_version", def: "스키마 버전 잠금 ('0.2.0'). 스키마 변경 시 이 값을 올리고 전체 prompt_regression + schema 테스트를 실행함." },
  { term: "causal_trigger_actor_index", def: "자차 조작을 유발한 액터를 actor_dynamics[]에서 식별하는 인덱스. 현재 단일 Scout 배포에서는 null — P4-6 Judge (Qwen3-8B-AWQ)는 report-only라 이 값을 채우지 않음." },
  { term: "grounded_by_yolo26", def: "액터별 불리언. YOLO26 객체 감지가 해당 액터의 존재를 독립적으로 확인하면 true — 해당 액터 항목의 환각 위험 감소." },
  { term: "hallucination_flags", def: "Scout이 값을 임의로 생성했을 수 있는 필드 이름 문자열 배열 (confidence 블록). 검토자 주의가 필요한 낮은 신뢰도 DNA 영역을 표시하는 데 사용됨." },
];

// ── Sub-components ────────────────────────────────────────────────────────────

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[11px] bg-surface-2 text-accent px-2 py-0.5 rounded shrink-0">
      {children}
    </span>
  );
}

function LayerHeader({
  layerNum,
  layerLabel,
  icon: Icon,
  title,
  description,
  source,
}: {
  layerNum?: string;
  layerLabel?: string;
  icon: React.ElementType;
  title: string;
  description: string;
  source?: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="shrink-0 w-9 h-9 rounded-lg bg-accent/15 border border-accent/25 flex items-center justify-center">
        {layerNum ? (
          <span className="text-xs font-bold text-accent">{layerNum}</span>
        ) : (
          <Icon className="w-4 h-4 text-accent" />
        )}
      </div>
      <div>
        {layerLabel && (
          <div className="text-[10px] text-accent uppercase tracking-widest mb-0.5">{layerLabel}</div>
        )}
        <h2 className="text-base font-bold text-ink">{title}</h2>
        <p className="text-sm text-muted mt-0.5">{description}</p>
        {source && (
          <p className="text-[10px] text-faint mt-1 font-mono">Source: {source}</p>
        )}
      </div>
    </div>
  );
}

function FieldCard({ field }: { field: FieldDef }) {
  return (
    <div className="card p-4 space-y-3">
      <div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm font-semibold text-ink">{field.name}</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-line/50 text-muted border border-line/40">
            {field.type}
          </span>
        </div>
        <p className="text-xs text-muted mt-1.5 leading-relaxed">{field.description}</p>
        {field.source && (
          <p className="text-[10px] text-faint mt-1 font-mono">↳ {field.source}</p>
        )}
      </div>
      <div className="border-t border-line" />
      <div className="space-y-2">
        {field.rows.map(({ value, desc }) => (
          <div key={value} className="flex gap-3 items-baseline">
            <Tag>{value}</Tag>
            <span className="text-xs text-muted leading-relaxed">{desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HelpPage() {
  const [activeId, setActiveId] = useState("overview");
  const [lang, setLang] = useState<"en" | "ko">("en");

  const isKo = lang === "ko";
  const TOC_ITEMS      = isKo ? TOC_ITEMS_KO      : TOC_ITEMS_EN;
  const ODD_FIELDS     = isKo ? ODD_FIELDS_KO     : ODD_FIELDS_EN;
  const TOPOLOGY_FIELDS= isKo ? TOPOLOGY_FIELDS_KO: TOPOLOGY_FIELDS_EN;
  const ACTOR_FIELDS   = isKo ? ACTOR_FIELDS_KO   : ACTOR_FIELDS_EN;
  const PLANNER_FIELDS = isKo ? PLANNER_FIELDS_KO : PLANNER_FIELDS_EN;
  const REVIEW_STATES  = isKo ? REVIEW_STATES_KO  : REVIEW_STATES_EN;
  const GLOSSARY       = isKo ? GLOSSARY_KO       : GLOSSARY_EN;

  useEffect(() => {
    const mainEl = document.querySelector("main");
    if (!mainEl) return;
    const handleScroll = () => {
      let current = TOC_ITEMS[0].id;
      for (const { id } of TOC_ITEMS) {
        const el = document.getElementById(id);
        if (!el) continue;
        const rect = el.getBoundingClientRect();
        if (rect.top <= mainEl.getBoundingClientRect().top + mainEl.clientHeight * 0.4) {
          current = id;
        }
      }
      setActiveId(current);
    };
    mainEl.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => mainEl.removeEventListener("scroll", handleScroll);
  }, [TOC_ITEMS]);

  return (
    <div className="p-8 space-y-8">
      {/* page header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="t-title text-ink flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-accent" />
            {isKo ? "도움말 & 레퍼런스" : "Help & Reference"}
          </h1>
          <p className="text-sm text-muted mt-0.5">
            {isKo
              ? "My-Curator · Scenario DNA v0.2 레퍼런스 가이드"
              : "My-Curator · Scenario DNA v0.2 Reference Guide"}
          </p>
        </div>
        {/* lang toggle */}
        <div className="flex items-center gap-1 bg-surface-2 border border-line rounded-lg p-1 shrink-0">
          <button
            onClick={() => setLang("en")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              !isKo
                ? "bg-accent/20 text-accent border border-accent/30"
                : "text-muted hover:text-ink"
            }`}
          >
            EN
          </button>
          <button
            onClick={() => setLang("ko")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              isKo
                ? "bg-accent/20 text-accent border border-accent/30"
                : "text-muted hover:text-ink"
            }`}
          >
            KO
          </button>
        </div>
      </div>

      {/* body */}
      <div className="flex gap-8">

        {/* sticky TOC */}
        <aside className="hidden md:block w-48 shrink-0">
          <div className="sticky top-6 card p-3 space-y-0.5">
            <p className="text-[10px] text-faint uppercase tracking-widest px-2 pb-2">
              {isKo ? "목차" : "Contents"}
            </p>
            {TOC_ITEMS.map(({ id, label }) => (
              <a
                key={id}
                href={`#${id}`}
                onClick={() => setActiveId(id)}
                className={`block px-2 py-1.5 rounded text-xs transition-colors ${
                  activeId === id
                    ? "text-accent bg-accent/10"
                    : "text-muted hover:text-ink hover:bg-surface-hover"
                }`}
              >
                {label}
              </a>
            ))}
          </div>
        </aside>

        {/* main content */}
        <div className="flex-1 min-w-0 space-y-14">

          {/* ── 1. Overview ─────────────────────────────────────── */}
          <section id="overview" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={BookOpen}
              title={isKo ? "플랫폼 개요" : "Platform Overview"}
              description={isKo
                ? "My-Curator가 하는 일과 AV 데이터 파이프라인에서의 역할."
                : "What My-Curator does and how it fits into the AV data pipeline."}
            />
            <div className="card p-5 space-y-3">
              {isKo ? (
                <>
                  <p className="text-sm text-ink leading-relaxed">
                    <strong className="text-ink">My-Curator</strong>는 자율주행 차량 주행 클립을 위한{" "}
                    <strong className="text-ink">Verify-by-Exception (VBE)</strong> 큐레이션 플랫폼입니다.
                    NVIDIA DeepStream 9.0 파이프라인이 원본 영상을 수집하고, Scout VLM(Cosmos-Reason2-8B FP8)이
                    각 클립에 대한 구조화된{" "}
                    <strong className="text-ink">Scenario DNA</strong>를 생성하며,
                    검토자는 예외 클립 — elevated 또는 critical 위험으로 플래그된 클립 — 만 큐레이션합니다.
                  </p>
                  <p className="text-sm text-muted leading-relaxed">
                    각 클립은 <strong className="text-ink">4계층 DNA 설명자</strong>를 가집니다:
                    ODD(환경), 토폴로지(도로 인프라), 액터 다이내믹스(도로 사용자),
                    플래너 로직(자차 의도 + 위험). v0.2는 4계층 위에 자유서술{" "}
                    <strong className="text-ink">scene_description</strong> 내러티브와
                    planner_logic 내 <strong className="text-ink">risk_level_rationale</strong> ·{" "}
                    <strong className="text-ink">safety_event</strong> 채널을 추가합니다. DNA는
                    PostgreSQL(JSONB + GIN 인덱스)과 Milvus(768차원 Cosmos-Embed1-336p 임베딩)에
                    저장되어 하이브리드 검색을 지원합니다.
                  </p>
                </>
              ) : (
                <>
                  <p className="text-sm text-ink leading-relaxed">
                    <strong className="text-ink">My-Curator</strong> is a{" "}
                    <strong className="text-ink">Verify-by-Exception (VBE)</strong> curation
                    platform for autonomous-vehicle driving clips. A NVIDIA DeepStream 9.0 pipeline ingests raw
                    video, a Scout VLM (Cosmos-Reason2-8B FP8) generates structured{" "}
                    <strong className="text-ink">Scenario DNA</strong> for each clip, and
                    reviewers curate only the exceptions — clips flagged as elevated or critical risk.
                  </p>
                  <p className="text-sm text-muted leading-relaxed">
                    Each clip carries a <strong className="text-ink">4-layer DNA descriptor</strong>:
                    ODD (environment), Topology (road infrastructure), Actor Dynamics (road users), and
                    Planner Logic (ego intent + risk). v0.2 adds a free-text{" "}
                    <strong className="text-ink">scene_description</strong> narrative plus{" "}
                    <strong className="text-ink">risk_level_rationale</strong> and a{" "}
                    <strong className="text-ink">safety_event</strong> channel in planner_logic on top
                    of the four layers. DNA is stored in PostgreSQL (JSONB + GIN index)
                    and in Milvus (768-dim Cosmos-Embed1-336p embeddings) for hybrid search.
                  </p>
                </>
              )}
              <div className="flex flex-wrap gap-2 pt-1">
                {["Verify-by-Exception", "4-Layer DNA", "Hybrid Vector + Filter Search", "ISO 21448 SOTIF", "ASAM OSI / OpenDRIVE Taxonomy"].map((t) => (
                  <span key={t} className="text-[11px] px-2 py-0.5 rounded-full bg-accent/10 text-accent border border-accent/20">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </section>

          {/* ── 2. Workflow ─────────────────────────────────────── */}
          <section id="workflow" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={Zap}
              title={isKo ? "사용 방법" : "How to Use"}
              description={isKo
                ? "원본 영상에서 큐레이션 코퍼스까지의 4단계 큐레이션 워크플로우."
                : "The four-step curation workflow from raw video to curated corpus."}
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              {(isKo ? [
                { step: "1", label: "수집",           icon: Layers,       desc: "DS 파이프라인이 프레임을 캡처하고 Scout이 DNA를 생성하며, 클립이 review_status = pending으로 PG + Milvus에 저장됨." },
                { step: "2", label: "검색 & 큐레이션", icon: Navigation,   desc: "하이브리드 검색 페이지에서 자연어 또는 DNA 필터로 클립 조회. 결과는 코사인 유사도로 순위 결정됨." },
                { step: "3", label: "Review Queue",   icon: Eye,          desc: "대기 중인 클립을 순서대로 처리. 카드 클릭 시 비디오 재생, DNA 아코디언, 유사 클립이 포함된 전체 상세 보기 열림." },
                { step: "4", label: "Approve / Reject",icon: CheckCircle2, desc: "클립을 승인하여 큐레이션 코퍼스에 추가. 중복, 저품질, 잘못 레이블된 클립은 거부." },
              ] : [
                { step: "1", label: "Ingest",          icon: Layers,       desc: "DS pipeline captures frames, Scout generates DNA, clip is stored in PG + Milvus with review_status = pending." },
                { step: "2", label: "Search & Curate", icon: Navigation,   desc: "Use the hybrid search page to query clips by natural language or DNA filters. Results are ranked by cosine similarity." },
                { step: "3", label: "Review Queue",    icon: Eye,          desc: "Work through pending clips. Click any card to open the full detail view with video playback, DNA accordion, and similar clips." },
                { step: "4", label: "Approve / Reject",icon: CheckCircle2, desc: "Approve clips to add them to the curated corpus. Reject duplicates, low-quality, or mislabeled clips." },
              ]).map(({ step, label, icon: Icon, desc }) => (
                <div key={step} className="card p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-accent/20 text-accent text-xs font-bold flex items-center justify-center shrink-0">
                      {step}
                    </span>
                    <Icon className="w-4 h-4 text-accent shrink-0" />
                    <span className="text-sm font-semibold text-ink">{label}</span>
                  </div>
                  <p className="text-xs text-muted leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ── 3. Layer 1 ODD ──────────────────────────────────── */}
          <section id="odd" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L1"
              layerLabel="Layer 1"
              icon={Cloud}
              title={isKo ? "ODD — 운영 설계 도메인" : "ODD — Operational Design Domain"}
              description={isKo
                ? "클립이 촬영된 환경 조건."
                : "Environmental conditions under which the clip was captured."}
              source="ASAM OpenSCENARIO v1.0 · ASAM OSI v3.x · ISO 22736"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {ODD_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 4. Layer 2 Topology ─────────────────────────────── */}
          <section id="topology" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L2"
              layerLabel="Layer 2"
              icon={Map}
              title={isKo ? "토폴로지 — 도로 인프라" : "Topology — Road Infrastructure"}
              description={isKo
                ? "도로 분류, 차선 구성, 교차로 형상."
                : "Road classification, lane configuration, and intersection geometry."}
              source="OpenDRIVE v1.5M XSD"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {TOPOLOGY_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 5. Layer 3 Actors ───────────────────────────────── */}
          <section id="actors" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L3"
              layerLabel="Layer 3"
              icon={Users}
              title={isKo ? "액터 다이내믹스 — 동적 액터" : "Actor Dynamics — Dynamic Actors"}
              description={isKo
                ? "감지된 각 도로 사용자의 액터별 분류, 행동 상태, 근접도. 클립당 0개 이상의 액터 항목이 포함될 수 있음."
                : "Per-actor classification, behavioral state, and proximity for each detected road user. Each clip may contain zero or more actor entries."}
              source="ASAM OSI VehicleClassification · PedestrianCategory · MiscObjectCategory"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {ACTOR_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 6. Layer 4 Planner ──────────────────────────────── */}
          <section id="planner" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L4"
              layerLabel="Layer 4"
              icon={Navigation}
              title={isKo ? "플래너 로직 — 자차 의도 & 위험" : "Planner Logic — Ego Intent & Risk"}
              description={isKo
                ? "자차 조작 분류 및 ISO 21448 SOTIF 위험 수준."
                : "Ego vehicle maneuver classification and ISO 21448 SOTIF risk level."}
              source="WOD-E2E arXiv:2510.26125 · PEGASUS HAD-F · ISO 21448:2022 SOTIF"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {PLANNER_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 7. Review States ────────────────────────────────── */}
          <section id="review-states" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={CheckCircle2}
              title={isKo ? "리뷰 상태" : "Review States"}
              description={isKo
                ? "큐레이션 워크플로우에서 클립이 거치는 네 가지 생명주기 상태."
                : "The four lifecycle states a clip occupies in the curation workflow."}
            />
            <div className="card p-4 divide-y divide-[#1e3a5f]">
              {REVIEW_STATES.map(({ value, dot, badge, desc }) => (
                <div key={value} className="py-3 first:pt-0 last:pb-0 flex items-start gap-3">
                  <div className={`mt-1.5 w-2 h-2 rounded-full shrink-0 ${dot}`} />
                  <div>
                    <span className={`font-mono text-xs font-semibold px-2 py-0.5 rounded border ${badge}`}>
                      {value}
                    </span>
                    <p className="text-xs text-muted mt-1.5 leading-relaxed">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ── 8. Glossary ─────────────────────────────────────── */}
          <section id="glossary" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={BookOpen}
              title={isKo ? "용어 설명" : "Glossary"}
              description={isKo
                ? "플랫폼 전반에서 사용되는 핵심 용어 및 개념."
                : "Key terms and concepts used throughout the platform."}
            />
            <div className="card p-4 divide-y divide-[#1e3a5f]">
              {GLOSSARY.map(({ term, def }) => (
                <div key={term} className="py-3 first:pt-0 last:pb-0">
                  <div className="text-xs font-semibold text-ink mb-1">{term}</div>
                  <p className="text-xs text-muted leading-relaxed">{def}</p>
                </div>
              ))}
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}
