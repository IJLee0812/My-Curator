"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ChevronDown,
  Loader2,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";

import {
  ActorBadges,
  ConfidenceBar,
  OddbBadges,
  PlannerBadge,
  RiskBadge,
  TopologyBadges,
} from "@/components/dna-badges";
import { ClipThumbnail } from "@/components/clip-thumbnail";
import {
  ClipResult,
  filtersFromSets,
  searchClips,
} from "@/lib/api";

const WEATHER_OPTIONS = [
  "clear", "overcast", "light_rain", "heavy_rain",
  "snow", "heavy_snow", "fog", "mist",
];
const LIGHTING_OPTIONS = ["day", "dawn", "dusk", "night", "tunnel", "overcast_day"];
const ROAD_OPTIONS = [
  "motorway", "primary", "secondary", "residential", "rural", "parking", "service",
];
const RISK_OPTIONS = ["nominal", "elevated", "critical"] as const;
const MANEUVER_OPTIONS = [
  "cruise", "brake_soft", "brake_hard", "emergency_brake",
  "yield", "lane_change_left", "lane_change_right", "swerve", "stop",
];

function FilterGroup({
  label,
  options,
  selected,
  onToggle,
  color = "blue",
}: {
  label: string;
  options: string[];
  selected: Set<string>;
  onToggle: (v: string) => void;
  color?: string;
}) {
  const [open, setOpen] = useState(true);
  const colorMap: Record<string, string> = {
    blue: "bg-blue-500/20 text-blue-300 border-blue-500/30",
    purple: "bg-purple-500/20 text-purple-300 border-purple-500/30",
    red: "bg-red-500/20 text-red-300 border-red-500/30",
    pink: "bg-pink-500/20 text-pink-300 border-pink-500/30",
  };
  const active = colorMap[color] ?? colorMap.blue;
  return (
    <div className="mb-4">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 hover:text-slate-200"
      >
        {label}
        {selected.size > 0 && (
          <span className="text-cyan-400 normal-case font-normal">({selected.size})</span>
        )}
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? "" : "-rotate-90"}`} />
      </button>
      {open && (
        <div className="flex flex-wrap gap-1.5">
          {options.map((opt) => (
            <button
              key={opt}
              onClick={() => onToggle(opt)}
              className={`text-xs px-2 py-1 rounded-full border transition-colors ${
                selected.has(opt)
                  ? active
                  : "border-[#1e3a5f] text-slate-500 hover:text-slate-300 hover:border-slate-500"
              }`}
            >
              {opt.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [weather, setWeather] = useState<Set<string>>(new Set());
  const [lighting, setLighting] = useState<Set<string>>(new Set());
  const [road, setRoad] = useState<Set<string>>(new Set());
  const [risk, setRisk] = useState<Set<string>>(new Set());
  const [maneuver, setManeuver] = useState<Set<string>>(new Set());
  const [showFilters, setShowFilters] = useState(true);

  const [results, setResults] = useState<ClipResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const toggle =
    (setter: React.Dispatch<React.SetStateAction<Set<string>>>) => (val: string) =>
      setter((prev) => {
        const next = new Set(prev);
        if (next.has(val)) {
          next.delete(val);
        } else {
          next.add(val);
        }
        return next;
      });

  const clearAll = () => {
    setWeather(new Set());
    setLighting(new Set());
    setRoad(new Set());
    setRisk(new Set());
    setManeuver(new Set());
    setQuery("");
    setResults([]);
    setSubmitted(false);
    setError(null);
  };

  const activeFilterCount =
    weather.size + lighting.size + road.size + risk.size + maneuver.size;

  const runSearch = async () => {
    setLoading(true);
    setError(null);
    setSubmitted(true);
    try {
      const filters = filtersFromSets({
        weather,
        lighting,
        road_type: road,
        risk_level: risk,
        ego_maneuver: maneuver,
      });
      const data = await searchClips(query, filters, 20);
      setResults(data.results);
    } catch (e) {
      console.error(e);
      setError(
        e instanceof Error ? e.message : "Search request failed — see console for details.",
      );
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Debounced auto-search when filters change (not when query is being typed).
  useEffect(() => {
    if (!submitted) return;
    const id = setTimeout(() => {
      runSearch();
    }, 300);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weather, lighting, road, risk, maneuver]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runSearch();
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* filter sidebar */}
      {showFilters && (
        <aside className="w-56 shrink-0 border-r border-[#1e3a5f] overflow-y-auto p-4 bg-[#0a1120]">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" /> DNA Filters
            </span>
            {activeFilterCount > 0 && (
              <button
                onClick={clearAll}
                className="text-xs text-slate-500 hover:text-red-400 flex items-center gap-1"
              >
                <X className="w-3 h-3" /> clear
              </button>
            )}
          </div>
          <FilterGroup label="Weather" options={WEATHER_OPTIONS} selected={weather} onToggle={toggle(setWeather)} color="blue" />
          <FilterGroup label="Lighting" options={LIGHTING_OPTIONS} selected={lighting} onToggle={toggle(setLighting)} color="blue" />
          <FilterGroup label="Road Type" options={ROAD_OPTIONS} selected={road} onToggle={toggle(setRoad)} color="purple" />
          <FilterGroup label="Risk Level" options={[...RISK_OPTIONS]} selected={risk} onToggle={toggle(setRisk)} color="red" />
          <FilterGroup label="Ego Maneuver" options={MANEUVER_OPTIONS} selected={maneuver} onToggle={toggle(setManeuver)} color="pink" />
        </aside>
      )}

      {/* main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* search bar */}
        <form onSubmit={onSubmit} className="p-4 border-b border-[#1e3a5f] bg-[#070d1a]">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={`btn-ghost px-3 flex items-center gap-1.5 text-sm shrink-0 ${
                showFilters ? "border-cyan-500/40 text-cyan-400" : ""
              }`}
            >
              <SlidersHorizontal className="w-4 h-4" />
              {activeFilterCount > 0 && (
                <span className="w-4 h-4 rounded-full bg-cyan-500 text-gray-950 text-[10px] font-bold flex items-center justify-center">
                  {activeFilterCount}
                </span>
              )}
            </button>
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="text"
                placeholder="rainy night cut-in, pedestrian crossing, emergency brake…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="input-dark w-full pl-9 pr-9 text-sm"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2"
                >
                  <X className="w-4 h-4 text-slate-500 hover:text-slate-300" />
                </button>
              )}
            </div>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary text-sm flex items-center gap-1.5 shrink-0 disabled:opacity-60"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              Search
            </button>
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
            <span>
              {submitted
                ? `${results.length} result${results.length !== 1 ? "s" : ""}`
                : "Enter a query and press Search"}
            </span>
            {activeFilterCount > 0 && (
              <>
                <span>·</span>
                <span className="text-cyan-400">
                  {activeFilterCount} filter{activeFilterCount !== 1 ? "s" : ""} active
                </span>
              </>
            )}
            <span className="ml-auto text-slate-600">Hybrid: Milvus ANN → PG JSONB filter</span>
          </div>
        </form>

        {/* results */}
        <div className="flex-1 overflow-y-auto p-4">
          {error && (
            <div className="card p-4 mb-3 text-sm text-red-300 border-red-500/40">
              <strong className="font-semibold">Search failed:</strong> {error}
            </div>
          )}
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-500">
              <Loader2 className="w-8 h-8 animate-spin mb-2" />
              <p className="text-sm">Querying curation-api…</p>
            </div>
          ) : results.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-600">
              <Search className="w-12 h-12 mb-3 opacity-30" />
              <p className="text-sm">
                {submitted
                  ? "No clips match the current filters"
                  : "Type a query above to start searching"}
              </p>
              {submitted && activeFilterCount > 0 && (
                <button
                  onClick={clearAll}
                  className="mt-2 text-xs text-cyan-400 hover:text-cyan-300"
                >
                  Clear all filters
                </button>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              {results.map((clip, idx) => {
                const dna = clip.dna_json;
                const risk = dna?.planner_logic?.risk_level ?? "nominal";
                return (
                  <Link
                    key={clip.clip_id}
                    href={`/clips/${clip.clip_id}`}
                    className="card card-hover p-4 flex gap-4 items-start"
                  >
                    {/* rank + thumbnail */}
                    <div className="shrink-0 flex flex-col items-center gap-2">
                      <div className="w-6 h-6 rounded bg-[#0a1120] flex items-center justify-center text-xs text-slate-500 font-mono border border-[#1e3a5f]">
                        {idx + 1}
                      </div>
                      <div className="w-20 h-14 bg-[#0a1120] rounded border border-[#1e3a5f] flex items-center justify-center relative overflow-hidden">
                        <ClipThumbnail clipId={clip.clip_id} iconSize="sm" />
                        {clip.is_gold && (
                          <div className="absolute -top-1.5 -right-1.5 w-3.5 h-3.5 rounded-full bg-yellow-500/80 flex items-center justify-center">
                            <span className="text-[8px] text-gray-950 font-bold">G</span>
                          </div>
                        )}
                      </div>
                      <div className="text-[10px] font-mono text-slate-600 text-center leading-tight">
                        {clip.start_s !== null && clip.end_s !== null
                          ? `${clip.start_s.toFixed(1)}–${clip.end_s.toFixed(1)}s`
                          : "—"}
                      </div>
                      <div className="text-[10px] font-mono text-cyan-400 text-center leading-tight">
                        {clip.score.toFixed(3)}
                      </div>
                    </div>

                    {/* DNA summary */}
                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="text-xs font-mono text-slate-400 truncate">
                            {clip.clip_id}
                          </div>
                          <div className="text-xs text-slate-600 mt-0.5">
                            {clip.source_clip_id
                              ? `source: ${clip.source_clip_id}`
                              : (clip.blob_uri ?? "")}
                          </div>
                        </div>
                        <RiskBadge level={risk} />
                      </div>
                      <OddbBadges odd={dna?.odd} />
                      <TopologyBadges topology={dna?.topology} />
                      <ActorBadges actors={dna?.actor_dynamics} />
                      <div className="flex items-center gap-2">
                        <PlannerBadge planner={dna?.planner_logic} />
                        {dna?.confidence?.overall !== undefined && (
                          <ConfidenceBar value={dna.confidence.overall} />
                        )}
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
