"use client";

import { useState } from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { reviewClip } from "@/lib/api";

type ActiveState = "approved" | "rejected" | null;

function mapInitial(s: string): ActiveState {
  if (s === "approved") return "approved";
  if (s === "rejected" || s === "rejected_schema_invalid") return "rejected";
  return null;
}

export default function ApproveRejectButtons({
  clipId,
  dnaVersion,
  initialStatus,
}: {
  clipId: string;
  dnaVersion: string | null;
  initialStatus: string;
}) {
  const [status, setStatus] = useState<ActiveState>(mapInitial(initialStatus));
  const [loading, setLoading] = useState<"approve" | "reject" | null>(null);
  const [copied, setCopied] = useState(false);
  const [reverted, setReverted] = useState(false);

  const act = async (action: "approve" | "reject") => {
    if (loading) return;
    // Toggle: clicking the already-active state reverts to pending.
    const effectiveAction: "approve" | "reject" | "pending" =
      (action === "approve" && status === "approved") ||
      (action === "reject"  && status === "rejected")
        ? "pending"
        : action;
    setLoading(action);
    try {
      const res = await reviewClip(clipId, effectiveAction);
      if (res.state === "approved") { setStatus("approved"); setReverted(false); }
      else if (res.state === "rejected") { setStatus("rejected"); setReverted(false); }
      else { setStatus(null); setReverted(true); }
    } catch (e) {
      console.error("review action failed", e);
    } finally {
      setLoading(null);
    }
  };

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(clipId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("clipboard write failed", e);
    }
  };

  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Curation Decision</h3>
        <button
          type="button"
          onClick={copyId}
          className="text-xs text-muted hover:text-accent"
        >
          {copied ? "copied" : "copy clip_id"}
        </button>
      </div>
      <div className="flex gap-2 flex-wrap">
        <button
          disabled={!!loading}
          onClick={() => act("approve")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors disabled:opacity-60 ${
            status === "approved"
              ? "bg-green-500 text-on-accent border-green-500"
              : "border-green-500/30 text-green-600 dark:text-green-400 hover:bg-green-500/10"
          }`}
        >
          {loading === "approve" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <CheckCircle2 className="w-4 h-4" />
          )}
          Approve
        </button>
        <button
          disabled={!!loading}
          onClick={() => act("reject")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors disabled:opacity-60 ${
            status === "rejected"
              ? "bg-red-500 text-on-accent border-red-500"
              : "border-red-500/30 text-red-600 dark:text-red-400 hover:bg-red-500/10"
          }`}
        >
          {loading === "reject" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <XCircle className="w-4 h-4" />
          )}
          Reject
        </button>
      </div>
      {(status || reverted) && (
        <div className="text-xs text-muted bg-surface-2 rounded-lg p-2 border border-line">
          {status === "approved" && "✓ Clip approved — persisted to review_queue."}
          {status === "rejected" && "✗ Clip rejected — persisted to review_queue."}
          {reverted && !status && "↩ Reverted to pending."}
        </div>
      )}
      <div className="text-xs text-faint">DNA v{dnaVersion ?? "—"}</div>
    </div>
  );
}
