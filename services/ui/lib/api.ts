/**
 * curation-api client (P3-4).
 *
 * Two base URLs are honoured:
 *   - NEXT_PUBLIC_API_BASE — bundled into the client; used by browser fetches.
 *   - INTERNAL_API_BASE    — server-only; used by Server Components when the
 *                            UI runs inside the compose network.
 *
 * The branch is decided at call time via `typeof window`, so the same module
 * is safe to import from both Server and Client Components.
 */

// ── Scenario DNA types ───────────────────────────────────────────────────────

export type RiskLevel = "nominal" | "elevated" | "critical";
export type ReviewState = "pending" | "approved" | "rejected" | "rejected_schema_invalid";

export interface ActorDynamic {
  actor_class: string;
  state: string;
  distance_bucket: "near" | "mid" | "far";
  confidence: number;
  grounded_by_yolo26: boolean;
}

export interface SafetyEvent {
  has_event: boolean;
  event_type: string;
  collision_type: string | null;
  severity_estimate: string | null;
}

export interface ScenarioDNA {
  dna_version: string;
  clip_id: string;
  timestamp_range: { start_s: number; end_s: number };
  // v0.2: free-text scene narrative (absent on v0.1 clips).
  scene_description?: string;
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
    // v0.2 additions (absent on v0.1 clips).
    risk_level_rationale?: string;
    safety_event?: SafetyEvent;
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

const BROWSER_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8001";
const INTERNAL_BASE = process.env.INTERNAL_API_BASE ?? BROWSER_BASE;

function apiBase(): string {
  return typeof window === "undefined" ? INTERNAL_BASE : BROWSER_BASE;
}

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const err = new Error(`GET ${path} ${res.status} ${body}`) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as T;
}

async function postJSON<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const err = new Error(`POST ${path} ${res.status} ${body}`) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as T;
}

async function patchJSON<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const err = new Error(`PATCH ${path} ${res.status} ${body}`) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as T;
}

// ── Response shapes ─────────────────────────────────────────────────────────

export interface ClipResult {
  clip_id: string;
  score: number;
  dna_json: ScenarioDNA | null;
  start_s: number | null;
  end_s: number | null;
  blob_uri: string | null;
  source_clip_id: string | null;
}

export interface SearchResponse {
  results: ClipResult[];
  total: number;
}

export interface ClipDetail {
  clip_id: string;
  session_id: string;
  blob_uri: string;
  frames_blob_uri: string | null;
  start_s: number;
  end_s: number;
  precise_start_s: number;
  precise_end_s: number;
  source_clip_id: string | null;
  dna_version: string | null;
  dna_json: ScenarioDNA | null;
  presigned_url: string | null;
  review_status: string;
}

export interface ClipSummary {
  clip_id: string;
  session_id: string;
  blob_uri: string;
  frames_blob_uri: string | null;
  start_s: number;
  end_s: number;
  source_clip_id: string | null;
  dna_version: string | null;
  dna_json: ScenarioDNA | null;
}

export interface ClipListResponse {
  clips: ClipSummary[];
  total: number;
}

export interface CollectionInfo {
  collection_name: string;
  vector_count: number;
  dim: number;
  index_type: string;
  metric_type: string;
}

export interface CollectionsResponse {
  collections: CollectionInfo[];
}

export interface ReviewCounts {
  pending: number;
  approved: number;
  rejected: number;
  rejected_schema_invalid: number;
}

export interface StatsResponse {
  total_clips: number;
  scenario_dna_count: number;
  vector_count: number;
  review: ReviewCounts;
  dna_pass_rate: number;
}

export interface HealthResponse {
  status: "ok" | "loading";
}

export interface ReviewQueueItem {
  queue_id: number;
  clip_id: string;
  state: string;
  reviewed_at: string | null;
  reason: string | null;
  created_at: string;
  blob_uri: string;
  frames_blob_uri: string | null;
  start_s: number;
  end_s: number;
  dna_json: ScenarioDNA | null;
}

export interface ReviewQueueResponse {
  items: ReviewQueueItem[];
  total: number;
  page: number;
  size: number;
}

// ── Endpoints ───────────────────────────────────────────────────────────────

export async function searchClips(
  query: string,
  filters: Record<string, string[]>,
  limit = 20,
  dedupBySource = true,
): Promise<SearchResponse> {
  return postJSON<SearchResponse>("/v1/search", {
    query,
    filters,
    limit,
    top_k: 1000,
    dedup_by_source: dedupBySource,
  });
}

export async function getClip(clipId: string): Promise<ClipDetail> {
  return getJSON<ClipDetail>(`/v1/clips/${clipId}`);
}

export async function listClips(limit = 20): Promise<ClipListResponse> {
  return getJSON<ClipListResponse>(`/v1/clips?limit=${limit}`);
}

export async function getCollections(): Promise<CollectionsResponse> {
  return getJSON<CollectionsResponse>("/v1/collections");
}

export async function getStats(): Promise<StatsResponse> {
  return getJSON<StatsResponse>("/v1/stats");
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    return await getJSON<HealthResponse>("/health");
  } catch {
    return { status: "loading" };
  }
}

/**
 * Video-similarity search.  Returns null when the backend reports the source
 * clip has no `frames_blob_uri` (HTTP 422) — the caller is expected to fall
 * back to a DNA text query in that case.
 */
export async function searchByVideo(
  clipId: string,
  limit = 4,
): Promise<SearchResponse | null> {
  try {
    return await postJSON<SearchResponse>("/v1/search/video", {
      clip_id: clipId,
      limit,
      top_k: 1000,
    });
  } catch (err) {
    const status = (err as Error & { status?: number }).status;
    if (status === 422 || status === 404) return null;
    throw err;
  }
}

export async function reviewClip(
  clipId: string,
  action: "approve" | "reject" | "pending",
): Promise<{ clip_id: string; state: string }> {
  return patchJSON<{ clip_id: string; state: string }>(`/v1/clips/${clipId}/review`, { action });
}

export async function getReviewQueue(
  status?: string,
  page = 1,
  size = 30,
): Promise<ReviewQueueResponse> {
  const params = new URLSearchParams({ page: String(page), size: String(size) });
  if (status) params.set("status", status);
  return getJSON<ReviewQueueResponse>(`/v1/review?${params}`);
}

// ── Filter helpers ──────────────────────────────────────────────────────────

/** Keys understood by the backend `SearchFilters` schema. */
export const FILTER_KEYS = [
  "weather",
  "lighting",
  "road_type",
  "risk_level",
  "ego_maneuver",
] as const;

export type FilterKey = (typeof FILTER_KEYS)[number];

export function filtersFromSets(sets: Record<FilterKey, Set<string>>): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const k of FILTER_KEYS) {
    const v = sets[k];
    if (v && v.size > 0) out[k] = [...v];
  }
  return out;
}

