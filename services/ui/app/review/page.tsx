"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Clock,
  Filter,
  Loader2,
  XCircle,
} from "lucide-react";
import { getReviewQueue, getStats, reviewClip } from "@/lib/api";
import type { ReviewQueueItem } from "@/lib/api";
import { OddbBadges, RiskBadge } from "@/components/dna-badges";

type Tab = "pending" | "approved" | "rejected" | "schema_invalid";

const TAB_CONFIG: Record<Tab, { label: string; color: string; dot: string }> = {
  pending:        { label: "Pending",        color: "text-amber-600 dark:text-amber-400", dot: "bg-amber-400" },
  approved:       { label: "Approved",       color: "text-green-600 dark:text-green-400", dot: "bg-green-400" },
  rejected:       { label: "Rejected",       color: "text-red-600 dark:text-red-400",   dot: "bg-red-400"   },
  schema_invalid: { label: "Schema Invalid", color: "text-muted", dot: "bg-faint" },
};

const PAGE_SIZES = [30, 50, 100] as const;

const ZERO_COUNTS: Record<Tab, number> = { pending: 0, approved: 0, rejected: 0, schema_invalid: 0 };

const TABS: Tab[] = ["pending", "approved", "rejected", "schema_invalid"];

/** Windowed page list: first, last, current ±1, ellipses ("…") in the gaps. */
function pageWindow(current: number, totalPages: number): (number | "…")[] {
  if (totalPages <= 7) return Array.from({ length: totalPages }, (_, i) => i + 1);
  const out: (number | "…")[] = [1];
  const lo = Math.max(2, current - 1);
  const hi = Math.min(totalPages - 1, current + 1);
  if (lo > 2) out.push("…");
  for (let p = lo; p <= hi; p++) out.push(p);
  if (hi < totalPages - 1) out.push("…");
  out.push(totalPages);
  return out;
}

function ReviewQueuePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // Seed pagination state from the URL so a card → detail → Back round-trip
  // restores the tab/page/size the user was on (state alone resets on remount).
  const [tab, setTab] = useState<Tab>(() => {
    const t = searchParams.get("tab") as Tab | null;
    return t && TABS.includes(t) ? t : "pending";
  });
  const [page, setPage] = useState(() => {
    const p = parseInt(searchParams.get("page") ?? "1", 10);
    return Number.isFinite(p) && p >= 1 ? p : 1;
  });
  const [size, setSize] = useState<number>(() => {
    const s = Number(searchParams.get("size"));
    return (PAGE_SIZES as readonly number[]).includes(s) ? s : 30;
  });

  // Mirror pagination state into the URL (replace: no extra history entries for
  // in-page navigation; the card click still pushes a fresh entry to return to).
  useEffect(() => {
    const params = new URLSearchParams({ tab, page: String(page), size: String(size) });
    router.replace(`/review?${params}`, { scroll: false });
  }, [tab, page, size, router]);
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [total, setTotal] = useState(0);
  const [counts, setCounts] = useState<Record<Tab, number>>(ZERO_COUNTS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [acting, setActing] = useState<Set<string>>(new Set());
  const [dismissing, setDismissing] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [res, stats] = await Promise.all([
        getReviewQueue(tab, page, size),
        getStats(),
      ]);
      setItems(res.items);
      setTotal(res.total);
      setCounts({
        pending: stats.review.pending,
        approved: stats.review.approved,
        rejected: stats.review.rejected,
        schema_invalid: stats.review.rejected_schema_invalid,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load review queue");
    } finally {
      setLoading(false);
    }
  }, [tab, page, size]);

  useEffect(() => { load(); }, [load]);

  const totalPages = Math.max(1, Math.ceil(total / size));

  const changeTab = (next: Tab) => { setTab(next); setPage(1); };
  const changeSize = (next: number) => { setSize(next); setPage(1); };

  const act = async (clipId: string, action: "approve" | "reject") => {
    setActing((s) => new Set(s).add(clipId));
    try {
      await reviewClip(clipId, action);
      // Play the exit animation, then re-fetch so the reviewed clip leaves the
      // tab and the page backfills from the server (totals stay accurate).
      setDismissing((s) => new Set(s).add(clipId));
      setTimeout(() => {
        setDismissing((s) => { const n = new Set(s); n.delete(clipId); return n; });
        if (items.length === 1 && page > 1) setPage((p) => p - 1);
        else load();
      }, 320);
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

  return (
    <div className="p-8 space-y-8">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="t-title text-ink flex items-center gap-2">
            <ClipboardCheck className="w-5 h-5 text-accent" />
            Review Queue
          </h1>
          <p className="text-sm text-muted mt-0.5">
            Verify-by-Exception curation workflow · {counts[tab]} in {TAB_CONFIG[tab].label}
          </p>
          <p className="text-xs text-faint mt-0.5">
            Click any clip card to open the detail review page
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400 bg-amber-500/10 border border-amber-500/25 px-3 py-1.5 rounded-lg">
            <Clock className="w-3.5 h-3.5" />
            {counts.pending} pending
          </div>
        </div>
      </div>

      {/* tabs */}
      <div className="flex gap-1 border-b border-line">
        {(["pending", "approved", "rejected", "schema_invalid"] as Tab[]).map((key) => (
          <button
            key={key}
            onClick={() => changeTab(key)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              tab === key
                ? "border-accent text-accent"
                : "border-transparent text-muted hover:text-ink"
            }`}
          >
            {TAB_CONFIG[key].label}
            <span className={`ml-2 text-xs px-1.5 py-0.5 rounded-full ${
              tab === key ? "bg-accent/20 text-accent" : "bg-surface-2 text-muted"
            }`}>
              {counts[key]}
            </span>
          </button>
        ))}
      </div>

      {/* page-size selector */}
      <div className="flex items-center justify-between text-xs text-muted">
        <span>
          {total === 0
            ? "No items"
            : `Showing ${(page - 1) * size + 1}–${Math.min(page * size, total)} of ${total}`}
        </span>
        <div className="flex items-center gap-1.5">
          <span className="text-faint">Per page</span>
          {PAGE_SIZES.map((s) => (
            <button
              key={s}
              onClick={() => changeSize(s)}
              className={`px-2.5 py-1 rounded border text-xs font-mono transition-colors ${
                size === s
                  ? "border-accent text-accent bg-accent/10"
                  : "border-line text-muted hover:text-ink"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* body */}
      {loading ? (
        <div className="card p-12 text-center text-muted">
          <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
          <p className="text-sm">Loading review queue…</p>
        </div>
      ) : error ? (
        <div className="card p-6 text-center text-red-600 dark:text-red-400 border-red-500/40">{error}</div>
      ) : (
        <div className="space-y-2">
          {items.length === 0 ? (
            <div className="card p-12 text-center text-faint">
              <Filter className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">No items in this category</p>
            </div>
          ) : (
            items.map((item) => {
              const stateKey: Tab = item.state === "rejected_schema_invalid" ? "schema_invalid" : (item.state as Tab);
              const cfg = TAB_CONFIG[stateKey] ?? TAB_CONFIG.pending;
              const isPending = item.state === "pending";
              const isActing = acting.has(item.clip_id);
              const isDismissing = dismissing.has(item.clip_id);
              const risk = item.dna_json?.planner_logic?.risk_level ?? "nominal";

              return (
                <div
                  key={item.queue_id}
                  className={`card p-4 flex gap-4 items-start cursor-pointer hover:border-line transition-all duration-300 ${
                    isDismissing ? "opacity-0 translate-x-8 pointer-events-none" : "opacity-100 translate-x-0"
                  }`}
                  onClick={() => router.push(`/clips/${item.clip_id}?from=review`)}
                >
                  {/* queue id + state */}
                  <div className="shrink-0 flex flex-col items-center gap-2 w-20">
                    <div className="text-xs font-mono text-faint">#{item.queue_id}</div>
                    <div className={`flex flex-col items-center gap-1 text-xs ${cfg.color}`}>
                      <div className={`w-2 h-2 rounded-full ${cfg.dot} ${isPending ? "animate-pulse" : ""}`} />
                      <span className="text-center leading-tight">{cfg.label}</span>
                    </div>
                  </div>

                  {/* clip info */}
                  <div className="flex-1 min-w-0 space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="text-xs font-mono text-ink truncate">{item.clip_id}</div>
                        <div className="text-xs text-muted mt-0.5">
                          {item.start_s.toFixed(1)}–{item.end_s.toFixed(1)}s
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <RiskBadge level={risk} />
                      </div>
                    </div>

                    <OddbBadges odd={item.dna_json?.odd} />

                    {item.reason && (
                      <div className="text-xs bg-surface-2 border border-line rounded px-2 py-1.5 text-muted">
                        <span className="text-faint">reason: </span>{item.reason}
                      </div>
                    )}

                    <div className="flex items-center gap-3 text-xs text-faint">
                      <span>
                        Created:{" "}
                        {new Date(item.created_at).toLocaleString("en-US", {
                          timeZone: "Asia/Seoul",
                          year: "numeric",
                          month: "numeric",
                          day: "numeric",
                          hour: "numeric",
                          minute: "2-digit",
                        })}
                      </span>
                      {item.reviewed_at && (
                        <span>
                          · Reviewed:{" "}
                          {new Date(item.reviewed_at).toLocaleString("en-US", {
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
                        onClick={(e) => { e.stopPropagation(); act(item.clip_id, "approve"); }}
                        className="flex items-center justify-center gap-1.5 min-w-[90px] px-3 py-1.5 rounded-lg border border-green-500/30 text-green-600 dark:text-green-400 hover:bg-green-500/10 text-xs font-medium transition-colors disabled:opacity-40"
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
                        onClick={(e) => { e.stopPropagation(); act(item.clip_id, "reject"); }}
                        className="flex items-center justify-center gap-1.5 min-w-[90px] px-3 py-1.5 rounded-lg border border-red-500/30 text-red-600 dark:text-red-400 hover:bg-red-500/10 text-xs font-medium transition-colors disabled:opacity-40"
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

      {/* pagination */}
      {!loading && !error && totalPages > 1 && (
        <div className="flex items-center justify-center gap-1.5 pt-2">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="p-1.5 rounded border border-line text-muted hover:text-ink disabled:opacity-30 disabled:cursor-not-allowed"
            aria-label="Previous page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          {pageWindow(page, totalPages).map((p, i) =>
            p === "…" ? (
              <span key={`gap-${i}`} className="px-2 text-faint text-xs">…</span>
            ) : (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={`min-w-[32px] px-2 py-1 rounded border text-xs font-mono transition-colors ${
                  page === p
                    ? "border-accent text-accent bg-accent/10"
                    : "border-line text-muted hover:text-ink"
                }`}
              >
                {p}
              </button>
            )
          )}
          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="p-1.5 rounded border border-line text-muted hover:text-ink disabled:opacity-30 disabled:cursor-not-allowed"
            aria-label="Next page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

// useSearchParams() must sit under a Suspense boundary (Next.js app router).
export default function ReviewQueuePageWrapper() {
  return (
    <Suspense fallback={null}>
      <ReviewQueuePage />
    </Suspense>
  );
}
