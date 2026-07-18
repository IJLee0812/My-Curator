import Link from "next/link";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  ClipboardCheck,
  Database,
  Film,
  Search,
} from "lucide-react";

import {
  getCollections,
  getHealth,
  getStats,
  listClips,
} from "@/lib/api";
import type { ClipSummary, CollectionInfo, StatsResponse } from "@/lib/api";
import RecentClipsSection from "./RecentClipsSection";

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
        <span className="text-xs text-muted uppercase tracking-wider">{label}</span>
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center ${accent ?? "bg-accent/15 text-accent"}`}
        >
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="text-2xl font-bold text-ink">{value}</div>
      {sub && <div className="text-xs text-muted mt-1">{sub}</div>}
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
        : "bg-line";
  const text =
    status === "ok"
      ? "text-green-600 dark:text-green-400"
      : status === "warn"
        ? "text-amber-600 dark:text-amber-400"
        : "text-muted";
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-line last:border-0">
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${dot}`} />
        <span className="text-sm text-ink">{label}</span>
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
  const dnaPassRate = stats?.dna_pass_rate ?? null;
  const vectorCount = collection?.vector_count ?? 0;
  const decided = review.approved + review.rejected + review.rejected_schema_invalid;

  const apiStatus: "ok" | "warn" | "off" =
    health === "ok" ? "ok" : health === "loading" ? "warn" : "off";
  const apiDetail =
    health === "ok"
      ? ":8001 · healthy"
      : health === "loading"
        ? ":8001 · warming up"
        : ":8001 · unreachable";

  return (
    <div className="p-8 space-y-8">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="t-title text-ink">Dashboard</h1>
          <p className="text-sm text-muted mt-0.5">
            Scenario DNA v0.2 · corpus: {totalClips} clip{totalClips === 1 ? "" : "s"} ·
            DNA rows: {stats?.scenario_dna_count ?? 0}
          </p>
        </div>
        <Link href="/search" className="btn-primary flex items-center gap-2 text-sm">
          <Search className="w-4 h-4" />
          Search Clips
        </Link>
      </div>

      {/* stats */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
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
          accent="bg-amber-500/15 text-amber-600 dark:text-amber-400"
        />
        <StatCard
          label="DNA Pass Rate"
          value={decided === 0 || dnaPassRate === null ? "N/A" : `${(dnaPassRate * 100).toFixed(1)}%`}
          sub={decided === 0 ? "no decisions yet" : "approved / decided"}
          icon={CheckCircle2}
          accent="bg-green-500/15 text-green-600 dark:text-green-400"
        />
      </div>

      {/* middle row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* review breakdown */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-ink mb-4 flex items-center gap-2">
            <ClipboardCheck className="w-4 h-4 text-accent" /> Review Queue
          </h2>
          {([
            { label: "Approved", count: review.approved, hex: "#22c55e", color: "text-green-600 dark:text-green-400" },
            { label: "Rejected", count: review.rejected, hex: "#ef4444", color: "text-red-600 dark:text-red-400" },
            { label: "Pending", count: review.pending, hex: "#f59e0b", color: "text-amber-600 dark:text-amber-400" },
            { label: "Schema Invalid", count: review.rejected_schema_invalid, hex: "#64748b", color: "text-muted" },
          ] as const).map(({ label, count, hex, color }) => {
            const total =
              review.approved + review.rejected + review.pending + review.rejected_schema_invalid;
            const pct = total > 0 ? Math.round((count / total) * 100) : 0;
            return (
              <div key={label} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className={color}>{label}</span>
                  <span className="text-muted">{count} ({pct}%)</span>
                </div>
                <div className="h-2 bg-surface-2 rounded-full">
                  <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: hex, opacity: 0.75 }} />
                </div>
              </div>
            );
          })}
          <Link href="/review" className="mt-2 flex items-center gap-1 text-xs text-accent hover:text-accent">
            View queue <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        {/* system status */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-ink mb-2 flex items-center gap-2">
            <Activity className="w-4 h-4 text-accent" /> System Status
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
          <h2 className="text-sm font-semibold text-ink mb-4 flex items-center gap-2">
            <Database className="w-4 h-4 text-accent" />
            Milvus Collection — <span className="font-mono text-accent">{collection.collection_name}</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Vectors", value: String(collection.vector_count) },
              { label: "Dimension", value: String(collection.dim) },
              { label: "Index", value: collection.index_type },
              { label: "Metric", value: collection.metric_type === "IP" ? "Inner Product (Cosine Similarity)" : collection.metric_type },
            ].map(({ label, value }) => (
              <div key={label} className="bg-surface-2 rounded-lg p-3 border border-line">
                <div className="text-xs text-muted mb-1">{label}</div>
                <div className="text-sm font-mono font-semibold text-accent">{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* recent clips */}
      <RecentClipsSection initialClips={clips} />
    </div>
  );
}
