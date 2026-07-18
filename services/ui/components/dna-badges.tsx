import type { ScenarioDNA, RiskLevel, SafetyEvent } from "@/lib/api";

// Schema maxLength caps (schemas/scenario_dna_v0_2.schema.json). The Scout DNA
// normalizer hard-truncates these free-text fields to satisfy the schema, so a
// stored value at its cap was cut mid-thought. The dropped text is not
// recoverable (it is truncated before persistence), so we mark a capped value
// with an ellipsis rather than present a cut string as if it were whole.
export const SCENE_DESCRIPTION_MAX = 500;
export const RISK_RATIONALE_MAX = 300;

export function markIfTruncated(text: string, cap: number): string {
  return text.length >= cap ? `${text}…` : text;
}

const WEATHER_ICONS: Record<string, string> = {
  clear: "☀️", overcast: "☁️", light_rain: "🌦️", heavy_rain: "🌧️",
  snow: "🌨️", heavy_snow: "❄️", fog: "🌫️", mist: "🌁", sleet: "🌨️",
};
const LIGHTING_ICONS: Record<string, string> = {
  day: "🌤️", dawn: "🌅", dusk: "🌇", night: "🌙",
  tunnel: "🚇", overcast_day: "🌥️",
};

export function RiskBadge({ level, rationale }: { level: RiskLevel; rationale?: string }) {
  const cls =
    level === "critical" ? "badge-risk-critical" :
    level === "elevated" ? "badge-risk-elevated" :
    "badge-risk-nominal";
  const dot =
    level === "critical" ? "bg-red-400" :
    level === "elevated" ? "bg-amber-400" : "bg-green-400";
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full font-medium ${cls} ${rationale ? "cursor-help" : ""}`}
      title={rationale ? markIfTruncated(rationale, RISK_RATIONALE_MAX) : undefined}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {level}
    </span>
  );
}

// P3-4: components are defensive against partial DNA payloads — a clip whose
// scenario_dna row is still being processed (or whose JSON shape predates the
// current schema) may arrive missing optional fields.

export function OddbBadges({ odd }: { odd: Partial<ScenarioDNA["odd"]> | undefined | null }) {
  const weather = odd?.weather;
  const lighting = odd?.lighting;
  if (!weather && !lighting) {
    return <span className="text-xs text-faint">no ODD</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {weather && (
        <span className="badge-odd inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full">
          {WEATHER_ICONS[weather] ?? "🌡️"} {weather.replace(/_/g, " ")}
        </span>
      )}
      {lighting && (
        <span className="badge-odd inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full">
          {LIGHTING_ICONS[lighting] ?? "💡"} {lighting}
        </span>
      )}
    </div>
  );
}

export function TopologyBadges({
  topology,
}: { topology: Partial<ScenarioDNA["topology"]> | undefined | null }) {
  const road = topology?.road_type;
  const lane = topology?.lane_event;
  const intersection = topology?.intersection_type;
  if (!road && !lane && !intersection) {
    return <span className="text-xs text-faint">no topology</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {road && (
        <span className="badge-topology inline-flex items-center text-xs px-2 py-0.5 rounded-full">
          🛣️ {road.replace(/_/g, " ")}
        </span>
      )}
      {lane && lane !== "normal" && (
        <span className="badge-topology inline-flex items-center text-xs px-2 py-0.5 rounded-full">
          ⚠️ {lane.replace(/_/g, " ")}
        </span>
      )}
      {intersection && intersection !== "none" && (
        <span className="badge-topology inline-flex items-center text-xs px-2 py-0.5 rounded-full">
          ✕ {intersection.replace(/_/g, " ")}
        </span>
      )}
    </div>
  );
}

export function ActorBadges({
  actors,
}: { actors: ScenarioDNA["actor_dynamics"] | undefined | null }) {
  const list = actors ?? [];
  if (list.length === 0) return <span className="text-xs text-faint">no actors</span>;
  const topActors = list.slice(0, 3);
  return (
    <div className="flex flex-wrap gap-1">
      {topActors.map((a, i) => (
        <span key={i} className="badge-actor inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full">
          {a.actor_class.replace(/vehicle_|_/g, (m) => m === "vehicle_" ? "" : " ").trim()}
          <span className="opacity-60">·</span>
          <span className="opacity-80">{a.state}</span>
          {a.grounded_by_yolo26 && <span title="YOLO26 grounded" className="text-green-600 dark:text-green-400">✓</span>}
        </span>
      ))}
      {list.length > 3 && (
        <span className="text-xs text-muted">+{list.length - 3}</span>
      )}
    </div>
  );
}

export function PlannerBadge({
  planner,
}: { planner: Partial<ScenarioDNA["planner_logic"]> | undefined | null }) {
  if (!planner?.ego_maneuver) return null;
  return (
    <span className="badge-planner inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full">
      🎯 {planner.ego_maneuver.replace(/_/g, " ")}
    </span>
  );
}

// v0.2 safety_event card — rendered only when has_event is true (the nominal
// majority has no event).  Colour-coded by severity_estimate; collision_type is
// null across the current corpus (no collisions), so it renders as "—".
const SEVERITY_STYLE: Record<string, string> = {
  minor:    "border-amber-500/30 bg-amber-500/10 text-amber-300",
  moderate: "border-orange-500/30 bg-orange-500/10 text-orange-700 dark:text-orange-300",
  severe:   "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300",
  fatal:    "border-red-600/40 bg-red-600/15 text-red-700 dark:text-red-300",
};

function collisionLabel(collision: string | null): string {
  if (collision === null || collision === undefined) return "—";
  if (collision === "none") return "no collision";
  return collision.replace(/_/g, " ");
}

export function SafetyEventCard({ event }: { event: SafetyEvent | null | undefined }) {
  if (!event?.has_event) return null;
  const sev = event.severity_estimate ?? "";
  const style = SEVERITY_STYLE[sev] ?? "border-line/30 bg-faint/10 text-ink";
  return (
    <div className={`rounded-xl border p-3 ${style}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm">⚠️</span>
        <span className="text-xs font-semibold uppercase tracking-wide">Safety Event</span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div>
          <div className="text-[10px] uppercase opacity-60 mb-0.5">Type</div>
          <div className="font-mono">{(event.event_type || "—").replace(/_/g, " ")}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase opacity-60 mb-0.5">Severity</div>
          <div className="font-mono">{event.severity_estimate ?? "—"}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase opacity-60 mb-0.5">Collision</div>
          <div className="font-mono">{collisionLabel(event.collision_type)}</div>
        </div>
      </div>
    </div>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.9 ? "bg-green-500" : value >= 0.75 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-muted font-mono w-8 text-right">{pct}%</span>
    </div>
  );
}
