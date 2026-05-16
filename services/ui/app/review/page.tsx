"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  CheckCircle2,
  ClipboardCheck,
  Clock,
  ExternalLink,
  Filter,
  Loader2,
  XCircle,
} from "lucide-react";
import { getReviewQueue, reviewClip } from "@/lib/api";
import type { ReviewQueueItem } from "@/lib/api";
import { OddbBadges, RiskBadge } from "@/components/dna-badges";

type Tab = "pending" | "approved" | "rejected";

const TAB_CONFIG: Record<Tab, { label: string; color: string; dot: string }> = {
  pending:  { label: "Pending",  color: "text-amber-400", dot: "bg-amber-400" },
  approved: { label: "Approved", color: "text-green-400", dot: "bg-green-400" },
  rejected: { label: "Rejected", color: "text-red-400",   dot: "bg-red-400" },
};

function isRejected(state: string) {
  return state === "rejected" || state === "rejected_schema_invalid";
}

function tabMatches(state: string, tab: Tab) {
  return tab === "rejected" ? isRejected(state) : state === tab;
}

export default function ReviewQueuePage() {
  const [tab, setTab] = useState<Tab>("pending");
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [acting, setActing] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getReviewQueue(undefined, 200);
      setItems(res.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load review queue");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const act = async (clipId: string, action: "approve" | "reject") => {
    setActing((s) => new Set(s).add(clipId));
    try {
      const res = await reviewClip(clipId, action);
      setItems((prev) =>
        prev.map((i) => (i.clip_id === clipId ? { ...i, state: res.state } : i))
      );
    } catch (e) {
      console.error("review action failed", e);
    } finally {
      setActing((s) => {
        const n = new Set(s);
        n.delete(clipId);
        return n;
      });
    }
  };

  const counts: Record<Tab, number> = {
    pending:  items.filter((i) => i.state === "pending").length,
    approved: items.filter((i) => i.state === "approved").length,
    rejected: items.filter((i) => isRejected(i.state)).length,
  };

  const visible = items.filter((i) => tabMatches(i.state, tab));

  return (
    <div className="p-6 space-y-5">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <ClipboardCheck className="w-5 h-5 text-cyan-400" />
            Review Queue
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Verify-by-Exception curation workflow · {items.length} items
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/25 px-3 py-1.5 rounded-lg">
            <Clock className="w-3.5 h-3.5" />
            {counts.pending} pending
          </div>
        </div>
      </div>

      {/* tabs */}
      <div className="flex gap-1 border-b border-[#1e3a5f]">
        {(["pending", "approved", "rejected"] as Tab[]).map((key) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              tab === key
                ? "border-cyan-500 text-cyan-400"
                : "border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            {TAB_CONFIG[key].label}
            <span className={`ml-2 text-xs px-1.5 py-0.5 rounded-full ${
              tab === key ? "bg-cyan-500/20 text-cyan-400" : "bg-[#1e3a5f] text-slate-500"
            }`}>
              {counts[key]}
            </span>
          </button>
        ))}
      </div>

      {/* body */}
      {loading ? (
        <div className="card p-12 text-center text-slate-500">
          <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
          <p className="text-sm">Loading review queue…</p>
        </div>
      ) : error ? (
        <div className="card p-6 text-center text-red-400 border-red-500/40">{error}</div>
      ) : (
        <div className="space-y-2">
          {visible.length === 0 ? (
            <div className="card p-12 text-center text-slate-600">
              <Filter className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">No items in this category</p>
            </div>
          ) : (
            visible.map((item) => {
              const cfg = TAB_CONFIG[isRejected(item.state) ? "rejected" : (item.state as Tab)] ?? TAB_CONFIG.pending;
              const isPending = item.state === "pending";
              const isActing = acting.has(item.clip_id);
              const risk = item.dna_json?.planner_logic?.risk_level ?? "nominal";

              return (
                <div key={item.queue_id} className="card p-4 flex gap-4 items-start">
                  {/* queue id + state */}
                  <div className="shrink-0 flex flex-col items-center gap-2 w-20">
                    <div className="text-xs font-mono text-slate-600">#{item.queue_id}</div>
                    <div className={`flex flex-col items-center gap-1 text-xs ${cfg.color}`}>
                      <div className={`w-2 h-2 rounded-full ${cfg.dot} ${isPending ? "animate-pulse" : ""}`} />
                      <span className="text-center leading-tight">{cfg.label}</span>
                      {item.state === "rejected_schema_invalid" && (
                        <span className="text-[10px] text-slate-500 text-center leading-tight">schema</span>
                      )}
                    </div>
                  </div>

                  {/* clip info */}
                  <div className="flex-1 min-w-0 space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="text-xs font-mono text-slate-300 truncate">{item.clip_id}</div>
                        <div className="text-xs text-slate-500 mt-0.5">
                          {item.start_s.toFixed(1)}–{item.end_s.toFixed(1)}s
                          {item.is_gold && (
                            <span className="ml-2 px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 text-[10px]">
                              gold
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <RiskBadge level={risk} />
                        <Link
                          href={`/clips/${item.clip_id}`}
                          className="text-slate-500 hover:text-cyan-400 transition-colors"
                          title="View detail"
                        >
                          <ExternalLink className="w-3.5 h-3.5" />
                        </Link>
                      </div>
                    </div>

                    <OddbBadges odd={item.dna_json?.odd} />

                    {item.reason && (
                      <div className="text-xs bg-[#0a1120] border border-[#1e3a5f] rounded px-2 py-1.5 text-slate-400">
                        <span className="text-slate-600">reason: </span>{item.reason}
                      </div>
                    )}

                    <div className="flex items-center gap-3 text-xs text-slate-600">
                      <span>
                        Created:{" "}
                        {new Date(item.created_at).toLocaleString("ko-KR", {
                          timeZone: "Asia/Seoul",
                          dateStyle: "short",
                          timeStyle: "short",
                        })}
                      </span>
                      {item.reviewed_at && (
                        <span>
                          · Reviewed:{" "}
                          {new Date(item.reviewed_at).toLocaleString("ko-KR", {
                            timeZone: "Asia/Seoul",
                            dateStyle: "short",
                            timeStyle: "short",
                          })}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* actions */}
                  {isPending ? (
                    <div className="shrink-0 flex flex-col gap-2">
                      <button
                        disabled={isActing}
                        onClick={() => act(item.clip_id, "approve")}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-green-500/30 text-green-400 hover:bg-green-500/10 text-xs font-medium transition-colors disabled:opacity-40"
                      >
                        {isActing ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <CheckCircle2 className="w-3.5 h-3.5" />
                        )}
                        Approve
                      </button>
                      <button
                        disabled={isActing}
                        onClick={() => act(item.clip_id, "reject")}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-red-500/30 text-red-400 hover:bg-red-500/10 text-xs font-medium transition-colors disabled:opacity-40"
                      >
                        {isActing ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5" />
                        )}
                        Reject
                      </button>
                    </div>
                  ) : (
                    <div className={`shrink-0 flex items-center gap-1.5 text-xs ${cfg.color} px-3 py-1.5`}>
                      {item.state === "approved" ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : (
                        <XCircle className="w-4 h-4" />
                      )}
                      <span className="font-medium">{cfg.label}</span>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
