"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
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
  "snow", "heavy_snow", "fog", "mist", "sleet",
];
const LIGHTING_OPTIONS = ["day", "dawn", "dusk", "night", "tunnel", "overcast_day"];
const ROAD_OPTIONS = [
  "motorway", "trunk", "primary", "secondary", "residential",
  "rural", "parking", "service", "walkway", "cycling",
];
const RISK_OPTIONS = ["nominal", "elevated", "critical"] as const;
const MANEUVER_OPTIONS = [
  "cruise", "accelerate", "brake_soft", "brake_hard", "emergency_brake",
  "nudge_left", "nudge_right", "lane_change_left", "lane_change_right",
  "yield", "stop", "reverse", "swerve",
];

const PAGE_SIZE_OPTIONS = [10, 20, 50] as const;
type PageSize = (typeof PAGE_SIZE_OPTIONS)[number];

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
    blue: "bg-blue-500/20 text-blue-700 dark:text-blue-300 border-blue-500/30",
    purple: "bg-purple-500/20 text-purple-700 dark:text-purple-300 border-purple-500/30",
    red: "bg-red-500/20 text-red-700 dark:text-red-300 border-red-500/30",
    pink: "bg-pink-500/20 text-pink-700 dark:text-pink-300 border-pink-500/30",
  };
  const active = colorMap[color] ?? colorMap.blue;
  return (
    <div className="mb-4">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full text-xs font-semibold text-muted uppercase tracking-wider mb-2 hover:text-ink"
      >
        {label}
        {selected.size > 0 && (
          <span className="text-accent normal-case font-normal">({selected.size})</span>
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
                  : "border-line text-muted hover:text-ink hover:border-line"
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

function SearchPageInner() {
  const [query, setQuery] = useState("");
  const [weather, setWeather] = useState<Set<string>>(new Set());
  const [lighting, setLighting] = useState<Set<string>>(new Set());
  const [road, setRoad] = useState<Set<string>>(new Set());
  const [risk, setRisk] = useState<Set<string>>(new Set());
  const [maneuver, setManeuver] = useState<Set<string>>(new Set());
  const [groupBySource, setGroupBySource] = useState(true);
  const [showFilters, setShowFilters] = useState(true);

  const [allResults, setAllResults] = useState<ClipResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const [perPage, setPerPage] = useState<PageSize>(20);
  const [page, setPage] = useState(1);
  const [initialized, setInitialized] = useState(false);

  const router = useRouter();
  const searchParams = useSearchParams();

  const totalPages = Math.max(1, Math.ceil(allResults.length / perPage));
  const pageResults = allResults.slice((page - 1) * perPage, page * perPage);

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
    setAllResults([]);
    setSubmitted(false);
    setError(null);
    setPage(1);
    router.replace("/search");
  };

  const activeFilterCount =
    weather.size + lighting.size + road.size + risk.size + maneuver.size;

  const runSearch = async () => {
    setLoading(true);
    setError(null);
    setSubmitted(true);
    setPage(1);
    try {
      const filters = filtersFromSets({
        weather,
        lighting,
        road_type: road,
        risk_level: risk,
        ego_maneuver: maneuver,
      });
      const data = await searchClips(query, filters, query.trim() ? 20 : 500, groupBySource);
      setAllResults(data.results);
      // Sync filter state to URL so router.back() can restore it
      const params = new URLSearchParams();
      if (query) params.set("q", query);
      [...weather].forEach((v) => params.append("weather", v));
      [...lighting].forEach((v) => params.append("lighting", v));
      [...road].forEach((v) => params.append("road_type", v));
      [...risk].forEach((v) => params.append("risk_level", v));
      [...maneuver].forEach((v) => params.append("ego_maneuver", v));
      // Grouping defaults on; only record the opt-out to keep the URL clean.
      if (!groupBySource) params.set("group", "0");
      const qs = params.toString();
      router.replace(qs ? `/search?${qs}` : "/search");
    } catch (e) {
      console.error(e);
      setError(
        e instanceof Error ? e.message : "Search request failed — see console for details.",
      );
      setAllResults([]);
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
  }, [weather, lighting, road, risk, maneuver, groupBySource]);

  // Restore filter state from URL params on mount (e.g., after router.back()).
  useEffect(() => {
    const q = searchParams.get("q") ?? "";
    const w = new Set(searchParams.getAll("weather"));
    const l = new Set(searchParams.getAll("lighting"));
    const ro = new Set(searchParams.getAll("road_type"));
    const ri = new Set(searchParams.getAll("risk_level"));
    const m = new Set(searchParams.getAll("ego_maneuver"));
    const grouped = searchParams.get("group") !== "0"; // default on
    if (q || w.size || l.size || ro.size || ri.size || m.size || searchParams.has("group")) {
      setQuery(q);
      setWeather(w);
      setLighting(l);
      setRoad(ro);
      setRisk(ri);
      setManeuver(m);
      setGroupBySource(grouped);
      setInitialized(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-run search once after URL-based state restoration.
  useEffect(() => {
    if (initialized) runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialized]);

  // Reset to first page when perPage changes.
  useEffect(() => {
    setPage(1);
  }, [perPage]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runSearch();
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* filter sidebar */}
      {showFilters && (
        <aside className="w-44 sm:w-48 lg:w-56 shrink-0 border-r border-line overflow-y-auto p-4 bg-surface-2">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold text-ink flex items-center gap-1.5">
              <SlidersHorizontal className="w-3.5 h-3.5 text-accent" /> DNA Filters
            </span>
            {activeFilterCount > 0 && (
              <button
                onClick={clearAll}
                className="text-xs text-muted hover:text-red-600 dark:text-red-400 flex items-center gap-1"
              >
                <X className="w-3 h-3" /> clear
              </button>
            )}
          </div>
          <label className="flex items-start gap-2 mb-4 cursor-pointer group">
            <input
              type="checkbox"
              checked={groupBySource}
              onChange={(e) => setGroupBySource(e.target.checked)}
              className="mt-0.5 accent-accent shrink-0"
            />
            <span className="text-xs text-muted group-hover:text-ink leading-snug">
              Group adjacent windows
              <span className="block text-[10px] text-faint">
                collapse near-duplicate windows of the same source clip
              </span>
            </span>
          </label>
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
        <form onSubmit={onSubmit} className="p-4 border-b border-line bg-canvas">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={`btn-ghost px-3 flex items-center gap-1.5 text-sm shrink-0 ${
                showFilters ? "border-accent/40 text-accent" : ""
              }`}
            >
              <SlidersHorizontal className="w-4 h-4" />
              {activeFilterCount > 0 && (
                <span className="w-4 h-4 rounded-full bg-accent text-on-accent text-[10px] font-bold flex items-center justify-center">
                  {activeFilterCount}
                </span>
              )}
            </button>
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
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
                  <X className="w-4 h-4 text-muted hover:text-ink" />
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
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 mt-3 text-sm text-muted">
            <span className="shrink-0">
              {submitted
                ? allResults.length > 0
                  ? `${allResults.length} result${allResults.length !== 1 ? "s" : ""} · page ${page} / ${totalPages}`
                  : "0 results"
                : "Enter a query and press Search"}
            </span>
            {activeFilterCount > 0 && (
              <>
                <span>·</span>
                <span className="text-accent shrink-0">
                  {activeFilterCount} filter{activeFilterCount !== 1 ? "s" : ""} active
                </span>
              </>
            )}
            <span className="hidden lg:inline ml-auto text-faint">Hybrid: Milvus ANN → PG JSONB filter</span>
            {/* per-page selector */}
            <div className="flex items-center gap-2 ml-auto lg:ml-4 shrink-0">
              <span className="text-faint hidden sm:inline">per page</span>
              <div className="flex gap-1.5">
                {PAGE_SIZE_OPTIONS.map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setPerPage(n)}
                    className={`px-3 py-1 rounded text-sm border transition-colors ${
                      perPage === n
                        ? "border-accent/60 text-accent bg-accent/10"
                        : "border-line text-muted hover:text-ink"
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </form>

        {/* results */}
        <div className="flex-1 overflow-y-auto p-4">
          {error && (
            <div className="card p-4 mb-3 text-sm text-red-700 dark:text-red-300 border-red-500/40">
              <strong className="font-semibold">Search failed:</strong> {error}
            </div>
          )}
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full text-muted">
              <Loader2 className="w-8 h-8 animate-spin mb-2" />
              <p className="text-sm">Querying curation-api…</p>
            </div>
          ) : pageResults.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-faint">
              <Search className="w-12 h-12 mb-3 opacity-30" />
              <p className="text-sm">
                {submitted
                  ? "No clips match the current filters"
                  : "Type a query above to start searching"}
              </p>
              {submitted && activeFilterCount > 0 && (
                <button
                  onClick={clearAll}
                  className="mt-2 text-xs text-accent hover:text-accent"
                >
                  Clear all filters
                </button>
              )}
            </div>
          ) : (
            <>
              <div className="space-y-2">
                {pageResults.map((clip, idx) => {
                  const globalIdx = (page - 1) * perPage + idx;
                  const dna = clip.dna_json;
                  const risk = dna?.planner_logic?.risk_level ?? "nominal";
                  return (
                    <Link
                      key={clip.clip_id}
                      href={`/clips/${clip.clip_id}`}
                      className="card card-hover p-4 sm:p-5 flex gap-4 sm:gap-6 items-center"
                    >
                      {/* rank + thumbnail */}
                      <div className="shrink-0 flex flex-col items-center gap-2">
                        <div className="w-7 h-7 rounded bg-surface-2 flex items-center justify-center text-xs text-muted font-mono border border-line">
                          {globalIdx + 1}
                        </div>
                        <div className="w-24 h-[4.5rem] sm:w-48 sm:h-36 md:w-64 md:h-48 bg-surface-2 rounded border border-line flex items-center justify-center relative overflow-hidden">
                          <ClipThumbnail clipId={clip.clip_id} iconSize="sm" />
                        </div>
                        <div className="text-xs font-mono text-faint text-center leading-tight">
                          {clip.start_s !== null && clip.end_s !== null
                            ? `${clip.start_s.toFixed(1)}–${clip.end_s.toFixed(1)}s`
                            : "—"}
                        </div>
                        <div className="text-xs font-mono text-accent text-center leading-tight">
                          {clip.score.toFixed(3)}
                        </div>
                      </div>

                      {/* DNA summary */}
                      <div className="flex-1 min-w-0 space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="text-xs font-mono text-muted truncate">
                              {clip.clip_id}
                            </div>
                            <div className="text-xs text-faint mt-0.5">
                              {clip.source_clip_id
                                ? `source: ${clip.source_clip_id}`
                                : (clip.blob_uri ?? "")}
                            </div>
                          </div>
                          <RiskBadge level={risk} rationale={dna?.planner_logic?.risk_level_rationale} />
                        </div>
                        {dna?.scene_description && (
                          <p className="text-xs text-muted leading-snug line-clamp-2">
                            {dna.scene_description}
                          </p>
                        )}
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

              {/* pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-2 mt-4 pt-4 border-t border-line">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="p-1.5 rounded border border-line text-muted hover:text-ink hover:border-line disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>

                  {Array.from({ length: totalPages }, (_, i) => i + 1)
                    .filter((p) => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
                    .reduce<(number | "…")[]>((acc, p, i, arr) => {
                      if (i > 0 && (p as number) - (arr[i - 1] as number) > 1) acc.push("…");
                      acc.push(p);
                      return acc;
                    }, [])
                    .map((p, i) =>
                      p === "…" ? (
                        <span key={`ellipsis-${i}`} className="text-faint text-xs px-1">…</span>
                      ) : (
                        <button
                          key={p}
                          onClick={() => setPage(p as number)}
                          className={`min-w-[2rem] h-8 rounded border text-xs font-mono transition-colors ${
                            page === p
                              ? "border-accent/60 text-accent bg-accent/10"
                              : "border-line text-muted hover:text-ink hover:border-line"
                          }`}
                        >
                          {p}
                        </button>
                      )
                    )}

                  <button
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="p-1.5 rounded border border-line text-muted hover:text-ink hover:border-line disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense>
      <SearchPageInner />
    </Suspense>
  );
}
