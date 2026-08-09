"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Loader2 } from "lucide-react";
import { ClipThumbnail } from "@/components/clip-thumbnail";

import { RiskBadge } from "@/components/dna-badges";
import {
  ClipResult,
  searchByVideo,
  searchClips,
} from "@/lib/api";
import type { ScenarioDNA } from "@/lib/api";

const LIMIT = 4;

type Mode = "video" | "fallback" | null;

/**
 * Similar-clip panel for Clip Detail (P3-4).
 *
 * Strategy:
 *   1. Try POST /v1/search/video first — works when frames_blob_uri exists.
 *   2. On 422 (no frames captured) the API returns null → fall back to a
 *      DNA-text query against POST /v1/search using "<weather> <lighting>"
 *      so the panel still shows something useful for ingest-format clips.
 */
export default function SimilarClipsPanel({
  clipId,
  dna,
}: {
  clipId: string;
  dna: ScenarioDNA | null;
}) {
  const [results, setResults] = useState<ClipResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<Mode>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const video = await searchByVideo(clipId, LIMIT + 1);
        if (cancelled) return;
        if (video) {
          const filtered = video.results.filter((r) => r.clip_id !== clipId).slice(0, LIMIT);
          setResults(filtered);
          setMode("video");
        } else {
          // DNA-text fallback
          const weather = dna?.odd?.weather ?? "";
          const lighting = dna?.odd?.lighting ?? "";
          const text = `${weather} ${lighting}`.trim() || "driving scene";
          const txt = await searchClips(text, {}, LIMIT + 1);
          if (cancelled) return;
          const filtered = txt.results.filter((r) => r.clip_id !== clipId).slice(0, LIMIT);
          setResults(filtered);
          setMode("fallback");
        }
      } catch (e) {
        if (cancelled) return;
        console.error(e);
        setError(
          e instanceof Error ? e.message : "Similar-clip lookup failed",
        );
        setResults([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [clipId, dna]);

  if (loading) {
    return (
      <div>
        <h2 className="text-sm font-semibold text-ink mb-3">Similar Clips</h2>
        <div className="card p-6 flex items-center justify-center text-muted text-xs">
          <Loader2 className="w-4 h-4 animate-spin mr-2" /> Computing similar clips…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h2 className="text-sm font-semibold text-ink mb-3">Similar Clips</h2>
        <div className="card p-4 text-xs text-red-700 dark:text-red-300 border-red-500/40">{error}</div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div>
        <h2 className="text-sm font-semibold text-ink mb-3">Similar Clips</h2>
        <div className="card p-4 text-xs text-muted">
          No similar clips found in the current corpus.
        </div>
      </div>
    );
  }

  const title =
    mode === "video"
      ? "Similar Clips — video-tower nearest neighbours"
      : `Similar Clips — DNA fallback (${dna?.odd?.weather ?? "?"} · ${dna?.odd?.lighting ?? "?"})`;

  return (
    <div>
      <h2 className="text-sm font-semibold text-ink mb-3">{title}</h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {results.map((c) => {
          const risk = c.dna_json?.planner_logic?.risk_level ?? "nominal";
          return (
            <Link
              key={c.clip_id}
              href={`/clips/${c.clip_id}`}
              className="card card-hover p-3 block"
            >
              <div className="w-full h-32 bg-surface-2 rounded-lg mb-2 flex items-center justify-center border border-line relative overflow-hidden">
                <ClipThumbnail clipId={c.clip_id} iconSize="sm" />
              </div>
              <div className="text-[10px] font-mono text-muted truncate">
                {c.clip_id.slice(0, 16)}…
              </div>
              <div className="mt-1 flex items-center justify-between gap-1">
                <RiskBadge level={risk} />
                <span className="text-[10px] font-mono text-accent">
                  {c.score.toFixed(3)}
                </span>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
