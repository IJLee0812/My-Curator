import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardCheck,
  Database,
  Film,
  Search,
  TrendingUp,
} from "lucide-react";

import {
  getCollections,
  getHealth,
  getStats,
  listClips,
} from "@/lib/api";
import type { ClipSummary, CollectionInfo, StatsResponse } from "@/lib/api";
import { OddbBadges, RiskBadge } from "@/components/dna-badges";
import { ClipThumbnail } from "@/components/clip-thumbnail";
import { RECALL_AT_5 } from "@/lib/mock-data";

function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  accent,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ElementType;
  accent?: string;
}) {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center ${accent ?? "bg-cyan-500/15"}`}
        >
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="text-2xl font-bold text-slate-100">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  );
}

function SystemStatusRow({
  label,
  status,
  detail,
}: {
  label: string;
  status: "ok" | "warn" | "off";
  detail: string;
}) {
  const dot =
    status === "ok"
      ? "bg-green-400 animate-pulse"
      : status === "warn"
        ? "bg-amber-400"
        : "bg-slate-600";
  const text =
    status === "ok"
      ? "text-green-400"
      : status === "warn"
        ? "text-amber-400"
        : "text-slate-500";
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-[#1e3a5f] last:border-0">
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${dot}`} />
        <span className="text-sm text-slate-300">{label}</span>
      </div>
      <span className={`text-xs font-mono ${text}`}>{detail}</span>
    </div>
  );
}

export const dynamic = "force-dynamic";

type DashboardData = {
  stats: StatsResponse | null;
  collection: CollectionInfo | null;
  health: "ok" | "loading" | "down";
  clips: ClipSummary[];
};

async function loadDashboard(): Promise<DashboardData> {
  const [statsR, collectionsR, healthR, clipsR] = await Promise.allSettled([
    getStats(),
    getCollections(),
    getHealth(),
    listClips(6),
  ]);

  const stats = statsR.status === "fulfilled" ? statsR.value : null;
  const collection =
    collectionsR.status === "fulfilled"
      ? (collectionsR.value.collections[0] ?? null)
      : null;
  const health: "ok" | "loading" | "down" =
    healthR.status === "fulfilled"
      ? healthR.value.status === "ok"
        ? "ok"
        : "loading"
      : "down";
  const clips = clipsR.status === "fulfilled" ? clipsR.value.clips : [];
  return { stats, collection, health, clips };
}

export default async function DashboardPage() {
  const { stats, collection, health, clips } = await loadDashboard();

  const totalClips = stats?.total_clips ?? 0;
  const review = stats?.review ?? {
    pending: 0,
    approved: 0,
    rejected: 0,
    rejected_schema_invalid: 0,
  };
  const dnaPassRate = stats?.dna_pass_rate ?? 0;
  const vectorCount = collection?.vector_count ?? 0;

  const criticalCount = clips.filter(
    (c) => c.dna_json?.planner_logic?.risk_level === "critical",
  ).length;

  const apiStatus: "ok" | "warn" | "off" =
    health === "ok" ? "ok" : health === "loading" ? "warn" : "off";
  const apiDetail =
    health === "ok"
      ? ":8001 · healthy"
      : health === "loading"
        ? ":8001 · warming up"
        : ":8001 · unreachable";

  return (
    <div className="p-6 space-y-6">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Scenario DNA v0.1 · corpus: {totalClips} clip{totalClips === 1 ? "" : "s"} ·
            DNA rows: {stats?.scenario_dna_count ?? 0}
          </p>
        </div>
        <Link href="/search" className="btn-primary flex items-center gap-2 text-sm">
          <Search className="w-4 h-4" />
          Search Clips
        </Link>
      </div>

      {/* stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Clips"
          value={totalClips}
          sub={`${vectorCount} vectors in Milvus`}
          icon={Film}
        />
        <StatCard
          label="Pending Review"
          value={review.pending}
          sub="awaiting curation"
          icon={ClipboardCheck}
          accent="bg-amber-500/15 text-amber-400"
        />
        <StatCard
          label="DNA Pass Rate"
          value={`${(dnaPassRate * 100).toFixed(1)}%`}
          sub="approved / decided"
          icon={CheckCircle2}
          accent="bg-green-500/15 text-green-400"
        />
        <StatCard
          label="Recall@5"
          value={`${(RECALL_AT_5 * 100).toFixed(1)}%`}
          sub="hybrid search · gold set"
          icon={TrendingUp}
          accent="bg-cyan-500/15 text-cyan-400"
        />
      </div>

      {/* middle row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* review breakdown */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <ClipboardCheck className="w-4 h-4 text-cyan-400" /> Review Queue
          </h2>
          {([
            { label: "Approved", count: review.approved, bar: "bg-green-500", color: "text-green-400" },
            { label: "Rejected", count: review.rejected, bar: "bg-red-500", color: "text-red-400" },
            { label: "Pending", count: review.pending, bar: "bg-amber-500", color: "text-amber-400" },
            { label: "Schema Invalid", count: review.rejected_schema_invalid, bar: "bg-slate-500", color: "text-slate-400" },
          ] as const).map(({ label, count, bar, color }) => {
            const total =
              review.approved + review.rejected + review.pending + review.rejected_schema_invalid;
            const pct = total > 0 ? Math.round((count / total) * 100) : 0;
            return (
              <div key={label} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className={color}>{label}</span>
                  <span className="text-slate-500">{count} ({pct}%)</span>
                </div>
                <div className="h-1.5 bg-[#1e3a5f] rounded-full">
                  <div className={`h-full ${bar}/60 rounded-full`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
          <Link href="/review" className="mt-2 flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300">
            View queue <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        {/* risk distribution (recent clips) */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" /> Risk Distribution (recent)
          </h2>
          {(["nominal", "elevated", "critical"] as const).map((level) => {
            const count = clips.filter(
              (c) => c.dna_json?.planner_logic?.risk_level === level,
            ).length;
            const pct = clips.length > 0 ? Math.round((count / clips.length) * 100) : 0;
            const bar =
              level === "critical"
                ? "bg-red-500"
                : level === "elevated"
                  ? "bg-amber-500"
                  : "bg-green-500";
            return (
              <div key={level} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="capitalize text-slate-400">{level}</span>
                  <span className="text-slate-500">{count} ({pct}%)</span>
                </div>
                <div className="h-1.5 bg-[#1e3a5f] rounded-full">
                  <div className={`h-full ${bar}/70 rounded-full`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
          <div className="mt-3 pt-3 border-t border-[#1e3a5f] text-xs text-slate-500">
            <span className="text-red-400 font-semibold">{criticalCount} critical</span> in the last
            {" "}{clips.length} clip{clips.length === 1 ? "" : "s"}
          </div>
        </div>

        {/* system status */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" /> System Status
          </h2>
          <SystemStatusRow label="curation-api" status={apiStatus} detail={apiDetail} />
          <SystemStatusRow
            label={collection ? `Milvus ${collection.index_type}` : "Milvus"}
            status={collection ? "ok" : "off"}
            detail={collection ? `${collection.vector_count} vectors · ${collection.dim}-dim` : "—"}
          />
          <SystemStatusRow
            label="PostgreSQL 17"
            status={stats ? "ok" : "off"}
            detail={`${stats?.scenario_dna_count ?? 0} DNA rows`}
          />
          <SystemStatusRow
            label="Cosmos-Embed1-336p"
            status={health === "ok" ? "ok" : "warn"}
            detail={
              health === "ok"
                ? "GPU 0 · text tower warm"
                : "warming up / loading"
            }
          />
        </div>
      </div>

      {/* Milvus collection */}
      {collection && (
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <Database className="w-4 h-4 text-cyan-400" />
            Milvus Collection — <span className="font-mono text-cyan-400">{collection.collection_name}</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Vectors", value: String(collection.vector_count) },
              { label: "Dimension", value: String(collection.dim) },
              { label: "Index", value: collection.index_type },
              { label: "Metric", value: collection.metric_type },
            ].map(({ label, value }) => (
              <div key={label} className="bg-[#0a1120] rounded-lg p-3 border border-[#1e3a5f]">
                <div className="text-xs text-slate-500 mb-1">{label}</div>
                <div className="text-sm font-mono font-semibold text-cyan-300">{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* recent clips */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <Film className="w-4 h-4 text-cyan-400" /> Recent Clips
          </h2>
          <Link
            href="/search"
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            Browse all <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
        {clips.length === 0 ? (
          <div className="card p-6 text-center text-xs text-slate-500">
            No clips yet — once the DS pipeline ingests segments they will appear here.
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
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
                  className="card card-hover p-4 block"
                >
                  <div className="w-full h-28 bg-[#0a1120] rounded-lg mb-3 flex items-center justify-center border border-[#1e3a5f] relative overflow-hidden">
                    <ClipThumbnail clipId={clip.clip_id} />
                    <div className="absolute top-2 right-2">
                      <RiskBadge level={risk} />
                    </div>
                    {clip.is_gold && (
                      <div className="absolute top-2 left-2">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                          gold
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-[11px] font-mono text-slate-500 truncate">
                        {clip.clip_id.slice(0, 18)}…
                      </div>
                      <span className="text-[10px] font-mono text-slate-500 shrink-0 ml-1">
                        {clip.start_s.toFixed(1)}–{clip.end_s.toFixed(1)}s
                      </span>
                    </div>
                    <OddbBadges odd={odd} />
                    <div className="text-xs text-slate-500 truncate">{sourceTag}</div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
