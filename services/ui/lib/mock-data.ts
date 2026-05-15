// Type definitions for Scenario DNA v0.1 + the review-queue lifecycle.
//
// P3-4 update: Dashboard / Search / Clip Detail pages now consume the live
// curation-api (see `lib/api.ts`).  MOCK_CLIPS + MOCK_REVIEW_QUEUE remain
// because the Review Queue page is mock-only until the P3-5 review HTTP
// endpoints land — the page joins each queue row with the matching clip via
// MOCK_CLIPS.find().  MOCK_SYSTEM_STATS is removed because /v1/stats and
// /v1/collections cover those numbers now.

export type RiskLevel = "nominal" | "elevated" | "critical";
export type ReviewState = "pending" | "approved" | "rejected" | "rejected_schema_invalid";

export interface ActorDynamic {
  actor_class: string;
  state: string;
  distance_bucket: "near" | "mid" | "far";
  confidence: number;
  grounded_by_yolo26: boolean;
}

export interface ScenarioDNA {
  dna_version: string;
  clip_id: string;
  timestamp_range: { start_s: number; end_s: number };
  odd: {
    weather: string;
    lighting: string;
    sensor_fidelity: string[];
  };
  topology: {
    road_type: string;
    lane_event: string;
    intersection_type: string;
  };
  actor_dynamics: ActorDynamic[];
  planner_logic: {
    ego_maneuver: string;
    risk_level: RiskLevel;
    causal_trigger_actor_index: number | null;
  };
  confidence: {
    overall: number;
    scout_agreement: number;
    hallucination_flags: string[];
  };
  provenance: {
    scout_models: string[];
    scout_prompt_hash: string;
    judge_model: null;
    judge_prompt_hash: null;
    pipeline_version: string;
    is_synthetic: boolean;
    reference_standards: string[];
  };
}

export interface Clip {
  clip_id: string;
  session_id: string;
  video: string;
  start_s: number;
  end_s: number;
  blob_uri: string;
  frames_blob_uri: string;
  is_gold: boolean;
  dna: ScenarioDNA;
}

export interface ReviewQueueItem {
  queue_id: number;
  clip_id: string;
  state: ReviewState;
  reviewer: string | null;
  reviewed_at: string | null;
  reason: string | null;
  created_at: string;
}

const makeProvenance = (hash: string) => ({
  scout_models: ["Cosmos-Reason2-8B-FP8"],
  scout_prompt_hash: hash,
  judge_model: null,
  judge_prompt_hash: null,
  pipeline_version: "0.1.0",
  is_synthetic: false,
  reference_standards: [
    "PEGASUS 6-Layer Model (arXiv:2012.06319)",
    "ASAM OpenSCENARIO v1.0 XSD",
    "ASAM OSI v3.x",
    "OpenDRIVE v1.5M XSD",
    "ISO 34502:2022",
    "Waymo Open Dataset E2E (arXiv:2510.26125)",
    "ISO 21448 SOTIF",
  ],
});

export const MOCK_CLIPS: Clip[] = [
  {
    clip_id: "f4fbd547-3fcb-4933-b239-c9ec96d48454",
    session_id: "gold-set",
    video: "clear_day_1.mp4",
    start_s: 0.07,
    end_s: 5.07,
    blob_uri: "clips/gold-set/f4fbd547.mp4",
    frames_blob_uri: "frames/gold-set/f4fbd547",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "f4fbd547-3fcb-4933-b239-c9ec96d48454",
      timestamp_range: { start_s: 0.07, end_s: 5.07 },
      odd: { weather: "clear", lighting: "day", sensor_fidelity: ["clean"] },
      topology: { road_type: "urban", lane_event: "normal", intersection_type: "none" },
      actor_dynamics: [
        { actor_class: "vehicle_car", state: "cruise", distance_bucket: "mid", confidence: 0.92, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "cruise", risk_level: "nominal", causal_trigger_actor_index: null },
      confidence: { overall: 0.95, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef01"),
    },
  },
  {
    clip_id: "7a2ae0f4-f0d1-4f63-89b5-f3b877d57650",
    session_id: "gold-set",
    video: "clear_day_1.mp4",
    start_s: 5.07,
    end_s: 10.07,
    blob_uri: "clips/gold-set/7a2ae0f4.mp4",
    frames_blob_uri: "frames/gold-set/7a2ae0f4",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "7a2ae0f4-f0d1-4f63-89b5-f3b877d57650",
      timestamp_range: { start_s: 5.07, end_s: 10.07 },
      odd: { weather: "clear", lighting: "day", sensor_fidelity: ["clean"] },
      topology: { road_type: "urban", lane_event: "normal", intersection_type: "signalized" },
      actor_dynamics: [
        { actor_class: "pedestrian", state: "crossing", distance_bucket: "near", confidence: 0.88, grounded_by_yolo26: true },
        { actor_class: "vehicle_car", state: "stopped", distance_bucket: "near", confidence: 0.91, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "yield", risk_level: "elevated", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.90, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef02"),
    },
  },
  {
    clip_id: "d0722965-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
    session_id: "gold-set",
    video: "rain_night_1.mp4",
    start_s: 0.0,
    end_s: 5.0,
    blob_uri: "clips/gold-set/d0722965.mp4",
    frames_blob_uri: "frames/gold-set/d0722965",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "d0722965-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
      timestamp_range: { start_s: 0.0, end_s: 5.0 },
      odd: { weather: "heavy_rain", lighting: "night", sensor_fidelity: ["droplets_on_lens", "low_contrast"] },
      topology: { road_type: "primary", lane_event: "normal", intersection_type: "none" },
      actor_dynamics: [
        { actor_class: "vehicle_car", state: "cutin", distance_bucket: "near", confidence: 0.82, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "brake_hard", risk_level: "critical", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.84, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef03"),
    },
  },
  {
    clip_id: "bff4b946-2c3d-4e5f-9a0b-1c2d3e4f5a6b",
    session_id: "gold-set",
    video: "rain_night_1.mp4",
    start_s: 5.0,
    end_s: 10.0,
    blob_uri: "clips/gold-set/bff4b946.mp4",
    frames_blob_uri: "frames/gold-set/bff4b946",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "bff4b946-2c3d-4e5f-9a0b-1c2d3e4f5a6b",
      timestamp_range: { start_s: 5.0, end_s: 10.0 },
      odd: { weather: "heavy_rain", lighting: "night", sensor_fidelity: ["droplets_on_lens"] },
      topology: { road_type: "primary", lane_event: "normal", intersection_type: "t_junction" },
      actor_dynamics: [
        { actor_class: "vehicle_car", state: "emerging", distance_bucket: "near", confidence: 0.79, grounded_by_yolo26: false },
      ],
      planner_logic: { ego_maneuver: "brake_soft", risk_level: "elevated", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.81, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef04"),
    },
  },
  {
    clip_id: "e6f7a8b9-3d4e-5f6a-0b1c-2d3e4f5a6b7c",
    session_id: "gold-set",
    video: "snow_day_1.mp4",
    start_s: 0.0,
    end_s: 5.0,
    blob_uri: "clips/gold-set/e6f7a8b9.mp4",
    frames_blob_uri: "frames/gold-set/e6f7a8b9",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "e6f7a8b9-3d4e-5f6a-0b1c-2d3e4f5a6b7c",
      timestamp_range: { start_s: 0.0, end_s: 5.0 },
      odd: { weather: "heavy_snow", lighting: "day", sensor_fidelity: ["low_contrast"] },
      topology: { road_type: "motorway", lane_event: "normal", intersection_type: "none" },
      actor_dynamics: [],
      planner_logic: { ego_maneuver: "cruise", risk_level: "nominal", causal_trigger_actor_index: null },
      confidence: { overall: 0.93, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef05"),
    },
  },
  {
    clip_id: "c1d2e3f4-4e5f-6a7b-1c2d-3e4f5a6b7c8d",
    session_id: "gold-set",
    video: "snow_day_1.mp4",
    start_s: 5.0,
    end_s: 10.0,
    blob_uri: "clips/gold-set/c1d2e3f4.mp4",
    frames_blob_uri: "frames/gold-set/c1d2e3f4",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "c1d2e3f4-4e5f-6a7b-1c2d-3e4f5a6b7c8d",
      timestamp_range: { start_s: 5.0, end_s: 10.0 },
      odd: { weather: "heavy_snow", lighting: "day", sensor_fidelity: ["low_contrast", "motion_blur"] },
      topology: { road_type: "motorway", lane_event: "lane_closed", intersection_type: "none" },
      actor_dynamics: [
        { actor_class: "vehicle_truck", state: "stopped", distance_bucket: "far", confidence: 0.85, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "brake_soft", risk_level: "elevated", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.87, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef06"),
    },
  },
  {
    clip_id: "a9b8c7d6-5f6a-7b8c-2d3e-4f5a6b7c8d9e",
    session_id: "gold-set",
    video: "tunnel_night_1.mp4",
    start_s: 0.0,
    end_s: 5.0,
    blob_uri: "clips/gold-set/a9b8c7d6.mp4",
    frames_blob_uri: "frames/gold-set/a9b8c7d6",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "a9b8c7d6-5f6a-7b8c-2d3e-4f5a6b7c8d9e",
      timestamp_range: { start_s: 0.0, end_s: 5.0 },
      odd: { weather: "clear", lighting: "tunnel", sensor_fidelity: ["overexposed"] },
      topology: { road_type: "motorway", lane_event: "normal", intersection_type: "none" },
      actor_dynamics: [
        { actor_class: "vehicle_car", state: "tailing", distance_bucket: "near", confidence: 0.90, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "cruise", risk_level: "nominal", causal_trigger_actor_index: null },
      confidence: { overall: 0.91, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef07"),
    },
  },
  {
    clip_id: "b2c3d4e5-6a7b-8c9d-3e4f-5a6b7c8d9e0f",
    session_id: "gold-set",
    video: "tunnel_night_1.mp4",
    start_s: 5.0,
    end_s: 10.0,
    blob_uri: "clips/gold-set/b2c3d4e5.mp4",
    frames_blob_uri: "frames/gold-set/b2c3d4e5",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "b2c3d4e5-6a7b-8c9d-3e4f-5a6b7c8d9e0f",
      timestamp_range: { start_s: 5.0, end_s: 10.0 },
      odd: { weather: "clear", lighting: "tunnel", sensor_fidelity: ["lens_flare"] },
      topology: { road_type: "motorway", lane_event: "merge", intersection_type: "direct_connection" },
      actor_dynamics: [
        { actor_class: "vehicle_van", state: "cutin", distance_bucket: "near", confidence: 0.87, grounded_by_yolo26: true },
        { actor_class: "vehicle_car", state: "cruise", distance_bucket: "mid", confidence: 0.91, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "nudge_right", risk_level: "elevated", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.88, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef08"),
    },
  },
  {
    clip_id: "d3e4f5a6-7b8c-9d0e-4f5a-6b7c8d9e0f1a",
    session_id: "gold-set",
    video: "dusk_residential_1.mp4",
    start_s: 0.0,
    end_s: 5.0,
    blob_uri: "clips/gold-set/d3e4f5a6.mp4",
    frames_blob_uri: "frames/gold-set/d3e4f5a6",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "d3e4f5a6-7b8c-9d0e-4f5a-6b7c8d9e0f1a",
      timestamp_range: { start_s: 0.0, end_s: 5.0 },
      odd: { weather: "clear", lighting: "dusk", sensor_fidelity: ["lens_flare"] },
      topology: { road_type: "residential", lane_event: "normal", intersection_type: "crosswalk" },
      actor_dynamics: [
        { actor_class: "cyclist", state: "crossing", distance_bucket: "near", confidence: 0.84, grounded_by_yolo26: true },
        { actor_class: "pedestrian", state: "jaywalking", distance_bucket: "near", confidence: 0.78, grounded_by_yolo26: false },
      ],
      planner_logic: { ego_maneuver: "emergency_brake", risk_level: "critical", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.82, scout_agreement: 1.0, hallucination_flags: ["actor_count_mismatch"] },
      provenance: makeProvenance("deadbeef09"),
    },
  },
  {
    clip_id: "e5f6a7b8-8c9d-0e1f-5a6b-7c8d9e0f1a2b",
    session_id: "gold-set",
    video: "dusk_residential_1.mp4",
    start_s: 5.0,
    end_s: 10.0,
    blob_uri: "clips/gold-set/e5f6a7b8.mp4",
    frames_blob_uri: "frames/gold-set/e5f6a7b8",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "e5f6a7b8-8c9d-0e1f-5a6b-7c8d9e0f1a2b",
      timestamp_range: { start_s: 5.0, end_s: 10.0 },
      odd: { weather: "clear", lighting: "dusk", sensor_fidelity: ["low_contrast"] },
      topology: { road_type: "residential", lane_event: "normal", intersection_type: "t_junction" },
      actor_dynamics: [
        { actor_class: "delivery_motorcycle", state: "emerging", distance_bucket: "near", confidence: 0.83, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "yield", risk_level: "elevated", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.86, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef0a"),
    },
  },
  {
    clip_id: "f7a8b9c0-9d0e-1f2a-6b7c-8d9e0f1a2b3c",
    session_id: "gold-set",
    video: "fog_dawn_1.mp4",
    start_s: 0.0,
    end_s: 5.0,
    blob_uri: "clips/gold-set/f7a8b9c0.mp4",
    frames_blob_uri: "frames/gold-set/f7a8b9c0",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "f7a8b9c0-9d0e-1f2a-6b7c-8d9e0f1a2b3c",
      timestamp_range: { start_s: 0.0, end_s: 5.0 },
      odd: { weather: "fog", lighting: "dawn", sensor_fidelity: ["low_contrast"] },
      topology: { road_type: "rural", lane_event: "unmarked", intersection_type: "none" },
      actor_dynamics: [],
      planner_logic: { ego_maneuver: "cruise", risk_level: "nominal", causal_trigger_actor_index: null },
      confidence: { overall: 0.89, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef0b"),
    },
  },
  {
    clip_id: "a1b2c3d4-0e1f-2a3b-7c8d-9e0f1a2b3c4d",
    session_id: "gold-set",
    video: "fog_dawn_1.mp4",
    start_s: 5.0,
    end_s: 10.0,
    blob_uri: "clips/gold-set/a1b2c3d4.mp4",
    frames_blob_uri: "frames/gold-set/a1b2c3d4",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "a1b2c3d4-0e1f-2a3b-7c8d-9e0f1a2b3c4d",
      timestamp_range: { start_s: 5.0, end_s: 10.0 },
      odd: { weather: "fog", lighting: "dawn", sensor_fidelity: ["low_contrast", "motion_blur"] },
      topology: { road_type: "rural", lane_event: "normal", intersection_type: "unsignalized" },
      actor_dynamics: [
        { actor_class: "animal", state: "crossing", distance_bucket: "near", confidence: 0.71, grounded_by_yolo26: false },
      ],
      planner_logic: { ego_maneuver: "emergency_brake", risk_level: "critical", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.75, scout_agreement: 1.0, hallucination_flags: ["low_confidence_actor"] },
      provenance: makeProvenance("deadbeef0c"),
    },
  },
  {
    clip_id: "b3c4d5e6-1f2a-3b4c-8d9e-0f1a2b3c4d5e",
    session_id: "gold-set",
    video: "urban_night_2.mp4",
    start_s: 0.0,
    end_s: 5.0,
    blob_uri: "clips/gold-set/b3c4d5e6.mp4",
    frames_blob_uri: "frames/gold-set/b3c4d5e6",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "b3c4d5e6-1f2a-3b4c-8d9e-0f1a2b3c4d5e",
      timestamp_range: { start_s: 0.0, end_s: 5.0 },
      odd: { weather: "clear", lighting: "night", sensor_fidelity: ["clean"] },
      topology: { road_type: "secondary", lane_event: "normal", intersection_type: "roundabout" },
      actor_dynamics: [
        { actor_class: "standup_scooter_rider", state: "crossing", distance_bucket: "near", confidence: 0.80, grounded_by_yolo26: true },
        { actor_class: "vehicle_car", state: "cruise", distance_bucket: "mid", confidence: 0.93, grounded_by_yolo26: true },
      ],
      planner_logic: { ego_maneuver: "yield", risk_level: "elevated", causal_trigger_actor_index: 0 },
      confidence: { overall: 0.85, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef0d"),
    },
  },
  {
    clip_id: "c5d6e7f8-2a3b-4c5d-9e0f-1a2b3c4d5e6f",
    session_id: "gold-set",
    video: "urban_night_2.mp4",
    start_s: 5.0,
    end_s: 10.0,
    blob_uri: "clips/gold-set/c5d6e7f8.mp4",
    frames_blob_uri: "frames/gold-set/c5d6e7f8",
    is_gold: true,
    dna: {
      dna_version: "0.1.0",
      clip_id: "c5d6e7f8-2a3b-4c5d-9e0f-1a2b3c4d5e6f",
      timestamp_range: { start_s: 5.0, end_s: 10.0 },
      odd: { weather: "clear", lighting: "night", sensor_fidelity: ["lens_flare"] },
      topology: { road_type: "secondary", lane_event: "normal", intersection_type: "signalized" },
      actor_dynamics: [
        { actor_class: "vehicle_bus", state: "stopped", distance_bucket: "mid", confidence: 0.95, grounded_by_yolo26: true },
        { actor_class: "e_bike_rider", state: "cutin", distance_bucket: "near", confidence: 0.77, grounded_by_yolo26: false },
      ],
      planner_logic: { ego_maneuver: "brake_hard", risk_level: "critical", causal_trigger_actor_index: 1 },
      confidence: { overall: 0.83, scout_agreement: 1.0, hallucination_flags: [] },
      provenance: makeProvenance("deadbeef0e"),
    },
  },
];

export const MOCK_REVIEW_QUEUE: ReviewQueueItem[] = [
  { queue_id: 1, clip_id: "d0722965-1a2b-4c3d-8e9f-0a1b2c3d4e5f", state: "pending", reviewer: null, reviewed_at: null, reason: null, created_at: "2026-05-08T14:22:00Z" },
  { queue_id: 2, clip_id: "a1b2c3d4-0e1f-2a3b-7c8d-9e0f1a2b3c4d", state: "pending", reviewer: null, reviewed_at: null, reason: null, created_at: "2026-05-08T14:25:00Z" },
  { queue_id: 3, clip_id: "d3e4f5a6-7b8c-9d0e-4f5a-6b7c8d9e0f1a", state: "pending", reviewer: null, reviewed_at: null, reason: null, created_at: "2026-05-08T15:10:00Z" },
  { queue_id: 4, clip_id: "c5d6e7f8-2a3b-4c5d-9e0f-1a2b3c4d5e6f", state: "approved", reviewer: "IJLee0812", reviewed_at: "2026-05-08T16:00:00Z", reason: null, created_at: "2026-05-08T14:30:00Z" },
  { queue_id: 5, clip_id: "7a2ae0f4-f0d1-4f63-89b5-f3b877d57650", state: "approved", reviewer: "IJLee0812", reviewed_at: "2026-05-08T16:05:00Z", reason: null, created_at: "2026-05-08T14:28:00Z" },
  { queue_id: 6, clip_id: "b2c3d4e5-6a7b-8c9d-3e4f-5a6b7c8d9e0f", state: "rejected", reviewer: "IJLee0812", reviewed_at: "2026-05-08T16:10:00Z", reason: "Actor class mismatch with YOLO26 inventory", created_at: "2026-05-08T14:35:00Z" },
  { queue_id: 7, clip_id: "bff4b946-2c3d-4e5f-9a0b-1c2d3e4f5a6b", state: "rejected_schema_invalid", reviewer: null, reviewed_at: null, reason: "dna_json missing required topology.lane_event", created_at: "2026-05-08T13:00:00Z" },
];

// Recall@5 is exposed as a static constant — it is computed by the offline
// gold-set benchmark (P3-3) and not by any live endpoint.  When the
// post-Phase-3 ops sprint lands operational metrics, this constant can be
// retired in favour of `GET /v1/metrics/recall`.
export const RECALL_AT_5 = 0.929;
