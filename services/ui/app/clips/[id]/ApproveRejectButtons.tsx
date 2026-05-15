"use client";

import { useState } from "react";
import { AlertCircle, CheckCircle2, XCircle } from "lucide-react";

type ReviewState = "approve" | "reject" | "flag" | null;

function ActionButton({
  label,
  variant,
  onClick,
  active,
}: {
  label: string;
  variant: "approve" | "reject" | "flag";
  onClick: () => void;
  active: boolean;
}) {
  const styles = {
    approve: active
      ? "bg-green-500 text-gray-950 border-green-500"
      : "border-green-500/30 text-green-400 hover:bg-green-500/10",
    reject: active
      ? "bg-red-500 text-gray-950 border-red-500"
      : "border-red-500/30 text-red-400 hover:bg-red-500/10",
    flag: active
      ? "bg-amber-500 text-gray-950 border-amber-500"
      : "border-amber-500/30 text-amber-400 hover:bg-amber-500/10",
  };
  const icons = {
    approve: <CheckCircle2 className="w-4 h-4" />,
    reject: <XCircle className="w-4 h-4" />,
    flag: <AlertCircle className="w-4 h-4" />,
  };
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${styles[variant]}`}
    >
      {icons[variant]}
      {label}
    </button>
  );
}

/**
 * Local-state-only Approve / Reject / Flag controls (P3-4).
 *
 * Real persistence (POST /v1/review/{id}/approve, etc.) ships with P3-5.
 * Until then this component just toggles a UI flag so curators can preview
 * the interaction.
 */
export default function ApproveRejectButtons({
  clipId,
  dnaVersion,
}: {
  clipId: string;
  dnaVersion: string | null;
}) {
  const [state, setState] = useState<ReviewState>(null);
  const [copied, setCopied] = useState(false);

  const toggle = (next: NonNullable<ReviewState>) =>
    setState((prev) => (prev === next ? null : next));

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
        <h3 className="text-sm font-semibold text-slate-300">Curation Decision</h3>
        <button
          type="button"
          onClick={copyId}
          className="text-xs text-slate-500 hover:text-cyan-400"
        >
          {copied ? "copied" : "copy clip_id"}
        </button>
      </div>
      <div className="flex gap-2 flex-wrap">
        <ActionButton
          label="Approve"
          variant="approve"
          active={state === "approve"}
          onClick={() => toggle("approve")}
        />
        <ActionButton
          label="Reject"
          variant="reject"
          active={state === "reject"}
          onClick={() => toggle("reject")}
        />
        <ActionButton
          label="Flag"
          variant="flag"
          active={state === "flag"}
          onClick={() => toggle("flag")}
        />
      </div>
      {state && (
        <div className="text-xs text-slate-400 bg-[#0a1120] rounded-lg p-2 border border-[#1e3a5f]">
          {state === "approve" && "✓ Clip approved — local preview only (P3-5 will persist)."}
          {state === "reject" && "✗ Clip rejected — local preview only (P3-5 will persist)."}
          {state === "flag" && "⚑ Flagged for further review — local preview only (P3-5 will persist)."}
        </div>
      )}
      <div className="text-xs text-slate-600">
        DNA v{dnaVersion ?? "—"} · Endpoints land in P3-5
      </div>
    </div>
  );
}
