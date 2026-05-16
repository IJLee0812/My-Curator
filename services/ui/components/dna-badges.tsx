import type { ScenarioDNA, RiskLevel } from "@/lib/api";

const WEATHER_ICONS: Record<string, string> = {
  clear: "☀️", overcast: "☁️", light_rain: "🌦️", heavy_rain: "🌧️",
  snow: "🌨️", heavy_snow: "❄️", fog: "🌫️", mist: "🌁", sleet: "🌨️",
};
const LIGHTING_ICONS: Record<string, string> = {
  day: "🌤️", dawn: "🌅", dusk: "🌇", night: "🌙",
  tunnel: "🚇", overcast_day: "🌥️",
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  const cls =
    level === "critical" ? "badge-risk-critical" :
    level === "elevated" ? "badge-risk-elevated" :
    "badge-risk-nominal";
  const dot =
    level === "critical" ? "bg-red-400" :
    level === "elevated" ? "bg-amber-400" : "bg-green-400";
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full font-medium ${cls}`}>
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
    return <span className="text-xs text-slate-600">no ODD</span>;
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
    return <span className="text-xs text-slate-600">no topology</span>;
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
  if (list.length === 0) return <span className="text-xs text-slate-600">no actors</span>;
  const topActors = list.slice(0, 3);
  return (
    <div className="flex flex-wrap gap-1">
      {topActors.map((a, i) => (
        <span key={i} className="badge-actor inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full">
          {a.actor_class.replace(/vehicle_|_/g, (m) => m === "vehicle_" ? "" : " ").trim()}
          <span className="opacity-60">·</span>
          <span className="opacity-80">{a.state}</span>
          {a.grounded_by_yolo26 && <span title="YOLO26 grounded" className="text-green-400">✓</span>}
        </span>
      ))}
      {list.length > 3 && (
        <span className="text-xs text-slate-500">+{list.length - 3}</span>
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

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.9 ? "bg-green-500" : value >= 0.75 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-[#1e3a5f] rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-400 font-mono w-8 text-right">{pct}%</span>
    </div>
  );
}
