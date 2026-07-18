"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AlertTriangle, ArrowRight, Film, Loader2 } from "lucide-react";

import { listClips } from "@/lib/api";
import type { ClipSummary } from "@/lib/api";
import { OddbBadges, RiskBadge } from "@/components/dna-badges";
import { ClipThumbnail } from "@/components/clip-thumbnail";

const COUNT_OPTIONS = [6, 8, 12] as const;
type ClipCount = (typeof COUNT_OPTIONS)[number];

export default function RecentClipsSection({
  initialClips,
}: {
  initialClips: ClipSummary[];
}) {
  const [count, setCount] = useState<ClipCount>(6);
  const [clips, setClips] = useState<ClipSummary[]>(initialClips);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (count === 6 && clips === initialClips) return;
    let cancelled = false;
    setLoading(true);
    listClips(count)
      .then((res) => { if (!cancelled) setClips(res.clips); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count]);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <Film className="w-4 h-4 text-cyan-400" /> Recent Clips
        </h2>
        <div className="flex items-center gap-3">
          {/* count selector */}
          <div className="flex items-center gap-1">
            {COUNT_OPTIONS.map((n) => (
              <button
                key={n}
                onClick={() => setCount(n)}
                className={`text-xs px-2 py-0.5 rounded transition-colors ${
                  count === n
                    ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                    : "text-slate-500 hover:text-slate-300 border border-transparent hover:border-slate-600"
                }`}
              >
                {n}
              </button>
            ))}
          </div>
          <Link
            href="/search"
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            Browse all <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </div>

      {/* risk distribution — live with count selector */}
      {clips.length > 0 && (
        <div className="card p-5 mb-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" /> Risk Distribution (recent {count})
          </h2>
          {(["nominal", "elevated", "critical"] as const).map((level) => {
            const n = clips.filter(
              (c) => c.dna_json?.planner_logic?.risk_level === level,
            ).length;
            const pct = clips.length > 0 ? Math.round((n / clips.length) * 100) : 0;
            const hex =
              level === "critical" ? "#ef4444" : level === "elevated" ? "#f59e0b" : "#22c55e";
            return (
              <div key={level} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="capitalize text-slate-400">{level}</span>
                  <span className="text-slate-500">{n} ({pct}%)</span>
                </div>
                <div className="h-2 bg-[#1e3a5f] rounded-full">
                  <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: hex, opacity: 0.8 }} />
                </div>
              </div>
            );
          })}
          <div className="mt-3 pt-3 border-t border-[#1e3a5f] text-xs text-slate-500">
            <span className="text-red-400 font-semibold">
              {clips.filter((c) => c.dna_json?.planner_logic?.risk_level === "critical").length} critical
            </span>{" "}in the last {clips.length} clip{clips.length === 1 ? "" : "s"}
          </div>
        </div>
      )}

      {clips.length === 0 && !loading ? (
        <div className="card p-6 text-center text-xs text-slate-500">
          No clips yet — once the DS pipeline ingests segments they will appear here.
        </div>
      ) : (
        <div className="relative">
          {loading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#060c18]/60 rounded-lg">
              <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
            </div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {clips.map((clip) => {
              const risk = clip.dna_json?.planner_logic?.risk_level ?? "nominal";
              const odd = clip.dna_json?.odd;
              const sourceTag = clip.source_clip_id
                ? `source: ${clip.source_clip_id}`
                : clip.session_id;
              return (
                <Link
                  key={clip.clip_id}
                  href={`/clips/${clip.clip_id}`}
                  className="card card-hover p-5 block"
                >
                  <div className="w-full h-56 bg-[#0a1120] rounded-lg mb-4 flex items-center justify-center border border-[#1e3a5f] relative overflow-hidden">
                    <ClipThumbnail clipId={clip.clip_id} />
                    <div className="absolute top-2 right-2">
                      <RiskBadge level={risk} />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-mono text-slate-400 truncate">
                        {clip.clip_id.slice(0, 22)}…
                      </div>
                      <span className="text-xs font-mono text-slate-500 shrink-0 ml-2">
                        {clip.start_s.toFixed(1)}–{clip.end_s.toFixed(1)}s
                      </span>
                    </div>
                    <OddbBadges odd={odd} />
                    <div className="text-sm text-slate-500 truncate">{sourceTag}</div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
