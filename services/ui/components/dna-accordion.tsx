"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import type { ScenarioDNA } from "@/lib/api";
import {
  ActorBadges,
  ConfidenceBar,
  OddbBadges,
  PlannerBadge,
  RiskBadge,
  TopologyBadges,
} from "@/components/dna-badges";

function AccordionSection({
  title,
  color,
  children,
  defaultOpen = false,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-[#1e3a5f] rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className={`w-full flex items-center justify-between px-4 py-3 text-sm font-medium ${color} hover:opacity-90 transition-opacity`}
      >
        <span>{title}</span>
        {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>
      {open && <div className="bg-[#0a1120] px-4 py-3">{children}</div>}
    </div>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 py-1.5 border-b border-[#1e3a5f]/50 last:border-0">
      <span className="text-xs text-slate-500 w-36 shrink-0 pt-0.5">{k}</span>
      <span className="text-xs font-mono text-slate-300 break-all">{v}</span>
    </div>
  );
}

/**
 * 4-Layer DNA accordion (P3-4).
 *
 * Renders the full Scenario DNA v0.1 structure.  Pass `dna={null}` when the
 * underlying clip has no DNA row yet (the accordion shows a single empty
 * section instead of crashing on field access).
 */
export default function DnaAccordion({ dna }: { dna: ScenarioDNA | null }) {
  if (!dna) {
    return (
      <div className="border border-[#1e3a5f] rounded-xl p-4 text-xs text-slate-500">
        No DNA available — the embedder/scout pipeline has not produced a
        scenario_dna row for this clip yet.
      </div>
    );
  }

  const odd = dna.odd ?? { weather: "", lighting: "", sensor_fidelity: [] };
  const topology =
    dna.topology ?? { road_type: "", lane_event: "", intersection_type: "" };
  const actors = dna.actor_dynamics ?? [];
  const planner =
    dna.planner_logic ?? {
      ego_maneuver: "",
      risk_level: "nominal",
      causal_trigger_actor_index: null,
    };
  const provenance = dna.provenance;

  return (
    <div className="space-y-2">
      <AccordionSection
        title="Layer 1 — ODD (Operational Design Domain)"
        color="bg-blue-500/10 text-blue-300 border-b border-blue-500/20"
        defaultOpen
      >
        <div className="space-y-1 mb-3">
          <KV k="weather" v={odd.weather || "—"} />
          <KV k="lighting" v={odd.lighting || "—"} />
          <KV
            k="sensor_fidelity"
            v={(odd.sensor_fidelity ?? []).join(", ") || "—"}
          />
        </div>
        <OddbBadges odd={odd} />
      </AccordionSection>

      <AccordionSection
        title="Layer 2 — Topology (Road Infrastructure)"
        color="bg-purple-500/10 text-purple-300 border-b border-purple-500/20"
        defaultOpen
      >
        <div className="space-y-1 mb-3">
          <KV k="road_type" v={topology.road_type || "—"} />
          <KV k="lane_event" v={topology.lane_event || "—"} />
          <KV k="intersection_type" v={topology.intersection_type || "—"} />
        </div>
        <TopologyBadges topology={topology} />
      </AccordionSection>

      <AccordionSection
        title={`Layer 3 — Actor Dynamics (${actors.length} actor${actors.length !== 1 ? "s" : ""})`}
        color="bg-orange-500/10 text-orange-300 border-b border-orange-500/20"
        defaultOpen
      >
        {actors.length === 0 ? (
          <p className="text-xs text-slate-600 italic">No dynamic actors detected</p>
        ) : (
          <div className="space-y-3">
            {actors.map((actor, i) => (
              <div key={i} className="bg-[#0f1b2e] rounded-lg p-3 border border-[#1e3a5f]">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-mono text-orange-300">[{i}]</span>
                  <span className="text-xs font-semibold text-slate-200">
                    {actor.actor_class.replace(/_/g, " ")}
                  </span>
                  {actor.grounded_by_yolo26 && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-green-500/15 text-green-400 border border-green-500/25">
                      YOLO26 ✓
                    </span>
                  )}
                </div>
                <div className="space-y-1">
                  <KV k="state" v={actor.state} />
                  <KV k="distance_bucket" v={actor.distance_bucket} />
                  <KV k="confidence" v={<ConfidenceBar value={actor.confidence} />} />
                </div>
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      <AccordionSection
        title="Layer 4 — Planner Logic (Ego Intent + Risk)"
        color="bg-pink-500/10 text-pink-300 border-b border-pink-500/20"
        defaultOpen
      >
        <div className="space-y-1 mb-3">
          <KV k="ego_maneuver" v={planner.ego_maneuver || "—"} />
          <KV k="risk_level" v={<RiskBadge level={planner.risk_level} />} />
          <KV
            k="causal_trigger"
            v={
              planner.causal_trigger_actor_index !== null &&
              planner.causal_trigger_actor_index !== undefined
                ? `actor_dynamics[${planner.causal_trigger_actor_index}]`
                : <span className="text-slate-600">null</span>
            }
          />
        </div>
        <PlannerBadge planner={planner} />
      </AccordionSection>

      {dna.confidence && (
        <AccordionSection
          title="Confidence"
          color="bg-cyan-500/10 text-cyan-300 border-b border-cyan-500/20"
        >
          <div className="space-y-3">
            <div>
              <div className="text-xs text-slate-500 mb-1">Overall</div>
              <ConfidenceBar value={dna.confidence.overall} />
            </div>
            <div>
              <div className="text-xs text-slate-500 mb-1">Scout Agreement</div>
              <ConfidenceBar value={dna.confidence.scout_agreement} />
            </div>
            {dna.confidence.hallucination_flags?.length > 0 && (
              <div>
                <div className="text-xs text-amber-400 mb-1">⚠ Hallucination flags</div>
                {dna.confidence.hallucination_flags.map((f) => (
                  <div
                    key={f}
                    className="text-xs font-mono text-amber-300 bg-amber-500/10 px-2 py-1 rounded border border-amber-500/20 mb-1"
                  >
                    {f}
                  </div>
                ))}
              </div>
            )}
          </div>
        </AccordionSection>
      )}

      {provenance && (
        <AccordionSection
          title="Provenance"
          color="bg-slate-500/10 text-slate-400 border-b border-slate-500/20"
        >
          <div className="space-y-1">
            <KV k="scout_models" v={provenance.scout_models?.join(", ") || "—"} />
            <KV k="scout_prompt_hash" v={provenance.scout_prompt_hash || "—"} />
            <KV k="pipeline_version" v={provenance.pipeline_version || "—"} />
            <KV k="is_synthetic" v={String(provenance.is_synthetic ?? false)} />
            <KV
              k="judge_model"
              v={
                provenance.judge_model ?? (
                  <span className="text-slate-600">null (post-v0.1)</span>
                )
              }
            />
          </div>
        </AccordionSection>
      )}
    </div>
  );
}
