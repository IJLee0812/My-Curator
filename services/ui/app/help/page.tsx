"use client";

import { useEffect, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  Cloud,
  Eye,
  Layers,
  Map,
  Navigation,
  Users,
  Zap,
} from "lucide-react";

// ── TOC ───────────────────────────────────────────────────────────────────────

const TOC_ITEMS = [
  { id: "overview",       label: "Platform Overview" },
  { id: "workflow",       label: "How to Use" },
  { id: "odd",            label: "Layer 1 · ODD" },
  { id: "topology",       label: "Layer 2 · Topology" },
  { id: "actors",         label: "Layer 3 · Actors" },
  { id: "planner",        label: "Layer 4 · Planner" },
  { id: "review-states",  label: "Review States" },
  { id: "glossary",       label: "Glossary" },
];

// ── Types ─────────────────────────────────────────────────────────────────────

type EnumRow = { value: string; desc: string };
type FieldDef = { name: string; type: string; description: string; source?: string; rows: EnumRow[] };

// ── Data ──────────────────────────────────────────────────────────────────────

const ODD_FIELDS: FieldDef[] = [
  {
    name: "weather",
    type: "enum · required",
    description: "Precipitation and atmospheric state at clip time.",
    source: "ASAM OpenSCENARIO v1.0 CloudState + ASAM OSI PrecipitationIntensity",
    rows: [
      { value: "clear",       desc: "No precipitation, clear sky" },
      { value: "overcast",    desc: "Overcast sky, no precipitation" },
      { value: "light_rain",  desc: "Light rainfall" },
      { value: "heavy_rain",  desc: "Heavy rainfall" },
      { value: "snow",        desc: "Moderate snowfall" },
      { value: "heavy_snow",  desc: "Heavy snowfall" },
      { value: "fog",         desc: "Fog — low horizontal visibility" },
      { value: "mist",        desc: "Mist — reduced visibility, less severe than fog" },
      { value: "sleet",       desc: "Sleet — mixed rain, snow, and ice" },
    ],
  },
  {
    name: "lighting",
    type: "enum · required",
    description: "Ambient illumination category. Dawn and dusk are operationally distinct from day: low-angle sun and long shadows create sensor challenges absent in full daylight. overcast_day (~100 lx) differs from day (~10,000 lx) by reduced contrast and absence of shadows.",
    source: "ASAM OSI AmbientIllumination lux-level buckets",
    rows: [
      { value: "day",          desc: "Full daylight (~10,000 lx)" },
      { value: "dawn",         desc: "Sunrise transition (~1–100 lx) — low-angle sun, long shadows" },
      { value: "dusk",         desc: "Sunset transition (~1–100 lx) — low-angle sun, long shadows" },
      { value: "night",        desc: "Darkness — artificial or no illumination (<1 lx)" },
      { value: "tunnel",       desc: "Enclosed road with artificial lighting — rapid illumination transition on entry/exit" },
      { value: "overcast_day", desc: "Daylight under overcast cloud cover (~100 lx) — reduced contrast, no shadows" },
    ],
  },
  {
    name: "sensor_fidelity",
    type: "enum list",
    description: "Active degradation modes affecting the camera sensor output. Zero or more values; duplicates not allowed. An empty array means no degradation detected.",
    source: "Domain-specific (Korean urban road coverage)",
    rows: [
      { value: "clean",             desc: "No degradation detected" },
      { value: "lens_flare",        desc: "Bright light source causing lens flare artifacts" },
      { value: "droplets_on_lens",  desc: "Water droplets or condensation on the camera lens" },
      { value: "motion_blur",       desc: "Temporal blur from camera or ego-vehicle motion" },
      { value: "low_contrast",      desc: "Reduced image contrast" },
      { value: "overexposed",       desc: "Overexposure — blown-out highlight regions" },
    ],
  },
];

const TOPOLOGY_FIELDS: FieldDef[] = [
  {
    name: "road_type",
    type: "enum · required",
    description: "Road classification by purpose and traffic characteristics. Based on OpenDRIVE e_roadType. 'highway' was renamed to 'motorway' for ASAM standard alignment; 'urban' was split into 'primary' and 'secondary' for finer granularity.",
    source: "OpenDRIVE v1.5M XSD e_roadType",
    rows: [
      { value: "motorway",    desc: "Controlled-access highway — limited entry/exit points (formerly 'highway')" },
      { value: "trunk",       desc: "Major national or inter-city road (high-speed, non-motorway)" },
      { value: "primary",     desc: "Primary urban or inter-city road — controlled traffic flow" },
      { value: "secondary",   desc: "Secondary urban road — moderate traffic density (formerly part of 'urban')" },
      { value: "residential", desc: "Residential street — low speed, mixed pedestrian and vehicle use" },
      { value: "service",     desc: "Service road — parking lots, back alleys, delivery access roads" },
      { value: "rural",       desc: "Rural or unclassified road outside urban areas" },
      { value: "parking",     desc: "Parking lot or structured parking area" },
      { value: "walkway",     desc: "Pedestrian walkway — no motor vehicles permitted" },
      { value: "cycling",     desc: "Bicycle-dedicated path" },
    ],
  },
  {
    name: "lane_event",
    type: "enum · required",
    description: "Lane configuration or temporary alteration at the clip location.",
    source: "Domain-derived from traffic management conventions",
    rows: [
      { value: "normal",              desc: "Standard lane markings, no temporary changes" },
      { value: "construction_divert", desc: "Construction zone with active lane diversion" },
      { value: "lane_closed",         desc: "One or more lanes closed to traffic" },
      { value: "merge",               desc: "Lanes merging — two lanes converge into one" },
      { value: "split",               desc: "Lanes splitting — one lane diverges into two" },
      { value: "unmarked",            desc: "Lanes physically present but with no visible markings" },
    ],
  },
  {
    name: "intersection_type",
    type: "enum · required",
    description: "'direct_connection' specifically refers to highway on/off-ramp merge zones (OpenDRIVE direct junction) — not a generic road connection. 'crosswalk' marks a clip centred on a designated pedestrian crossing.",
    source: "OpenDRIVE v1.5M junction types",
    rows: [
      { value: "none",              desc: "No intersection — straight road section" },
      { value: "signalized",        desc: "Traffic-signal controlled intersection" },
      { value: "unsignalized",      desc: "Uncontrolled intersection (yield or stop signs only)" },
      { value: "roundabout",        desc: "Circular intersection with yield-on-entry rule" },
      { value: "t_junction",        desc: "T-shaped three-way intersection" },
      { value: "crosswalk",         desc: "Designated pedestrian crossing (zebra crossing)" },
      { value: "direct_connection", desc: "Highway on/off-ramp — direct merge zone (OpenDRIVE direct junction)" },
    ],
  },
];

const ACTOR_FIELDS: FieldDef[] = [
  {
    name: "actor_class",
    type: "enum · per actor",
    description: "Object classification by type and role. Korean urban-specific classes (e_bike_rider, delivery_motorcycle, standup_scooter_rider) are retained for local road coverage. 'vehicle_sedan' / 'vehicle_suv' were consolidated into 'vehicle_car' per ASAM OSI TYPE_CAR.",
    source: "ASAM OSI VehicleClassification + PedestrianCategory + MiscObjectCategory",
    rows: [
      { value: "pedestrian",           desc: "Foot-walking person" },
      { value: "cyclist",              desc: "Bicycle rider" },
      { value: "motorcyclist",         desc: "Motorcycle or motorbike rider" },
      { value: "standup_scooter_rider",desc: "E-scooter rider (standing position)" },
      { value: "e_bike_rider",         desc: "Electric bicycle rider" },
      { value: "delivery_motorcycle",  desc: "Delivery motorcycle or scooter (food/parcel courier)" },
      { value: "wheelchair_user",      desc: "Wheelchair occupant" },
      { value: "vehicle_car",          desc: "Passenger car — sedan, SUV, hatchback (ASAM OSI TYPE_CAR)" },
      { value: "vehicle_van",          desc: "Van or minivan (passenger or light cargo)" },
      { value: "vehicle_truck",        desc: "Truck — pickup truck or heavy goods vehicle" },
      { value: "vehicle_bus",          desc: "Bus — public transit or coach" },
      { value: "vehicle_emergency",    desc: "Emergency vehicle — ambulance, police car, fire truck" },
      { value: "vehicle_construction", desc: "Construction machine — excavator, loader, road roller" },
      { value: "animal",               desc: "Animal on or near the roadway (dog, cat, livestock)" },
      { value: "debris",               desc: "Fallen debris or foreign objects on the road" },
      { value: "construction_object",  desc: "Construction equipment, barriers, or site signage" },
      { value: "obstacle",             desc: "Generic obstacle — rock, fallen branch, pothole" },
    ],
  },
  {
    name: "state",
    type: "enum · per actor",
    description: "Behavioral state or motion intent of the detected actor at clip time.",
    source: "Domain-derived from driving scenario analysis",
    rows: [
      { value: "crossing",   desc: "Actively crossing the ego's path or travel lane" },
      { value: "hesitating", desc: "Slowed or paused — uncertain or interrupted motion" },
      { value: "jaywalking", desc: "Crossing the road without a designated crosswalk or signal" },
      { value: "cutin",      desc: "Lateral insertion into ego's lane from the side" },
      { value: "cutout",     desc: "Lateral departure from ego's lane to the side" },
      { value: "stopped",    desc: "Stopped — may resume motion (e.g. waiting at a signal)" },
      { value: "emerging",   desc: "Appearing into the scene from occlusion or off-screen" },
      { value: "tailing",    desc: "Following ego vehicle at close distance" },
      { value: "oncoming",   desc: "Approaching ego head-on in the opposite travel direction" },
      { value: "parked",     desc: "Parked — not moving, occupying a stationary position" },
      { value: "static",     desc: "Immobile object or permanently stationary actor" },
    ],
  },
  {
    name: "distance_bucket",
    type: "enum · per actor",
    description: "Spatial proximity to the ego vehicle, binned into three tiers for threat prioritisation.",
    source: "Domain-derived (threat-zone bucketing)",
    rows: [
      { value: "near", desc: "0 – 10 m — immediate threat zone" },
      { value: "mid",  desc: "10 – 50 m — reaction window" },
      { value: "far",  desc: "> 50 m — background context" },
    ],
  },
];

const PLANNER_FIELDS: FieldDef[] = [
  {
    name: "ego_maneuver",
    type: "enum · required",
    description: "Primary driving maneuver executed or intended by the ego vehicle. 'emergency_brake' = AEB/panic stop (maximum deceleration, automated trigger) — distinct from 'brake_hard' (intentional high-g deceleration). 'swerve' = lateral avoidance that does not complete a full lane change.",
    source: "WOD-E2E arXiv:2510.26125 + PEGASUS HAD-F maneuver taxonomy",
    rows: [
      { value: "cruise",            desc: "Constant speed, no active maneuver" },
      { value: "accelerate",        desc: "Intentional speed increase" },
      { value: "brake_soft",        desc: "Gentle, gradual deceleration" },
      { value: "brake_hard",        desc: "Intentional hard deceleration (high g)" },
      { value: "emergency_brake",   desc: "AEB / panic stop — maximum deceleration, automated trigger" },
      { value: "nudge_left",        desc: "Small lateral adjustment to the left without changing lanes" },
      { value: "nudge_right",       desc: "Small lateral adjustment to the right without changing lanes" },
      { value: "lane_change_left",  desc: "Complete lane change to the left" },
      { value: "lane_change_right", desc: "Complete lane change to the right" },
      { value: "yield",             desc: "Slowing or pausing to give priority to another actor" },
      { value: "stop",              desc: "Full stop — vehicle at rest with no motion" },
      { value: "reverse",           desc: "Backward motion" },
      { value: "swerve",            desc: "Lateral avoidance maneuver — does not complete a full lane change" },
    ],
  },
  {
    name: "risk_level",
    type: "enum · required",
    description: "Scenario risk classification per ISO 21448 SOTIF. Drives the Review Queue priority — 'critical' clips surface first. The DNA pass rate metric counts approved / (approved + rejected), excluding pending and schema_invalid.",
    source: "ISO 21448:2022 SOTIF",
    rows: [
      { value: "nominal",  desc: "No safety concern — absence of unreasonable risk (normal operation)" },
      { value: "elevated", desc: "Tolerable risk — hazard present but mitigation is in place" },
      { value: "critical", desc: "Unreasonable risk (SOTIF trigger) — intervention or override required" },
    ],
  },
];

const REVIEW_STATES = [
  {
    value: "pending",
    dot: "bg-amber-400 animate-pulse",
    badge: "text-amber-400 bg-amber-500/10 border-amber-500/25",
    desc: "Awaiting human review. All newly ingested clips start in this state.",
  },
  {
    value: "approved",
    dot: "bg-green-400",
    badge: "text-green-400 bg-green-500/10 border-green-500/25",
    desc: "Accepted into the curated corpus. DNA payload has been verified by a reviewer.",
  },
  {
    value: "rejected",
    dot: "bg-red-400",
    badge: "text-red-400 bg-red-500/10 border-red-500/25",
    desc: "Manually rejected by a human reviewer. Excluded from the training corpus.",
  },
  {
    value: "rejected_schema_invalid",
    dot: "bg-slate-500",
    badge: "text-slate-400 bg-slate-500/10 border-slate-600/40",
    desc: "Automatically rejected at ingestion time — the DNA payload failed JSON Schema validation. Displayed as 'Schema Invalid' in the UI.",
  },
];

const GLOSSARY = [
  { term: "Verify-by-Exception (VBE)", def: "A curation strategy where clips pass through automatically unless flagged. Reviewers focus effort only on uncertain or high-risk cases, dramatically reducing manual load on nominal clips." },
  { term: "Scenario DNA", def: "A 4-layer structured descriptor (ODD + Topology + Actor Dynamics + Planner Logic) attached to every clip. Stored as JSONB in PostgreSQL and indexed in Milvus. Schema version: v0.1.0 (frozen)." },
  { term: "ODD (Operational Design Domain)", def: "The specific conditions under which an AV system is designed to operate safely (ISO 22736). In My-Curator, ODD covers weather, lighting, and sensor fidelity." },
  { term: "Scout", def: "The VLM that generates Scenario DNA from video frames. Current model: Cosmos-Reason2-8B FP8. Multiple Scout samples per clip are aggregated by BestOfN Aggregator using a symbolic reward signal." },
  { term: "Gold Set", def: "A manually verified subset of clips (is_gold = true) used as ground truth for Recall@5 benchmark evaluation. Current: 14 clips, Recall@5 = 0.929." },
  { term: "Hybrid Search", def: "Retrieval combining Milvus ANN vector search (Cosmos-Embed1-336p, 768-dim, cosine / inner product on L2-normalised vectors) with PostgreSQL JSONB GIN filter on DNA fields. ANN top-1000 candidates are re-ranked by exact filter matching." },
  { term: "DNA Pass Rate", def: "Approved / (Approved + Rejected). Excludes pending and schema_invalid states. Shown on the Dashboard as a percentage." },
  { term: "SOTIF (ISO 21448)", def: "Safety of the Intended Functionality — ISO standard defining risk categories for AV systems. My-Curator's risk_level enum maps directly: nominal → no unreasonable risk, elevated → tolerable risk, critical → unreasonable risk trigger." },
  { term: "dna_version", def: "Schema version lock ('0.1.0'). Any schema change bumps this value and triggers a full prompt_regression + schema test run." },
  { term: "causal_trigger_actor_index", def: "Index into actor_dynamics[] identifying which actor caused the ego maneuver. Reserved for post-v0.1 Judge model (Qwen2.5-14B-AWQ); null in current v0.1.0 single-Scout deployments." },
  { term: "grounded_by_yolo26", def: "Boolean per actor. True if YOLO26 object detection independently confirmed the actor's presence, reducing hallucination risk for that actor entry." },
  { term: "hallucination_flags", def: "Array of field-name strings in the confidence block flagging fields where the Scout may have fabricated values. Used to surface low-confidence DNA regions for reviewer attention." },
];

// ── Sub-components ────────────────────────────────────────────────────────────

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[11px] bg-[#1e3a5f] text-cyan-300 px-2 py-0.5 rounded shrink-0">
      {children}
    </span>
  );
}

function LayerHeader({
  layerNum,
  layerLabel,
  icon: Icon,
  title,
  description,
  source,
}: {
  layerNum?: string;
  layerLabel?: string;
  icon: React.ElementType;
  title: string;
  description: string;
  source?: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="shrink-0 w-9 h-9 rounded-lg bg-cyan-500/15 border border-cyan-500/25 flex items-center justify-center">
        {layerNum ? (
          <span className="text-xs font-bold text-cyan-400">{layerNum}</span>
        ) : (
          <Icon className="w-4 h-4 text-cyan-400" />
        )}
      </div>
      <div>
        {layerLabel && (
          <div className="text-[10px] text-cyan-500 uppercase tracking-widest mb-0.5">{layerLabel}</div>
        )}
        <h2 className="text-base font-bold text-slate-100">{title}</h2>
        <p className="text-sm text-slate-500 mt-0.5">{description}</p>
        {source && (
          <p className="text-[10px] text-slate-600 mt-1 font-mono">Source: {source}</p>
        )}
      </div>
    </div>
  );
}

function FieldCard({ field }: { field: FieldDef }) {
  return (
    <div className="card p-4 space-y-3">
      <div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm font-semibold text-slate-100">{field.name}</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400 border border-slate-600/40">
            {field.type}
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">{field.description}</p>
        {field.source && (
          <p className="text-[10px] text-slate-600 mt-1 font-mono">↳ {field.source}</p>
        )}
      </div>
      <div className="border-t border-[#1e3a5f]" />
      <div className="space-y-2">
        {field.rows.map(({ value, desc }) => (
          <div key={value} className="flex gap-3 items-baseline">
            <Tag>{value}</Tag>
            <span className="text-xs text-slate-400 leading-relaxed">{desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HelpPage() {
  const [activeId, setActiveId] = useState("overview");

  useEffect(() => {
    const mainEl = document.querySelector("main");
    if (!mainEl) return;
    const handleScroll = () => {
      let current = TOC_ITEMS[0].id;
      for (const { id } of TOC_ITEMS) {
        const el = document.getElementById(id);
        if (!el) continue;
        const rect = el.getBoundingClientRect();
        // section top within the upper 40% of the viewport → mark active
        if (rect.top <= mainEl.getBoundingClientRect().top + mainEl.clientHeight * 0.4) {
          current = id;
        }
      }
      setActiveId(current);
    };
    mainEl.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => mainEl.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="p-6 space-y-6">
      {/* page header */}
      <div>
        <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          Help &amp; Reference
        </h1>
        <p className="text-sm text-slate-500 mt-0.5">
          My-Curator · Scenario DNA v0.1 Reference Guide
        </p>
      </div>

      {/* body */}
      <div className="flex gap-8">

        {/* sticky TOC */}
        <aside className="hidden md:block w-48 shrink-0">
          <div className="sticky top-6 card p-3 space-y-0.5">
            <p className="text-[10px] text-slate-600 uppercase tracking-widest px-2 pb-2">
              Contents
            </p>
            {TOC_ITEMS.map(({ id, label }) => (
              <a
                key={id}
                href={`#${id}`}
                onClick={() => setActiveId(id)}
                className={`block px-2 py-1.5 rounded text-xs transition-colors ${
                  activeId === id
                    ? "text-cyan-400 bg-cyan-500/10"
                    : "text-slate-500 hover:text-slate-300 hover:bg-[#111f36]"
                }`}
              >
                {label}
              </a>
            ))}
          </div>
        </aside>

        {/* main content */}
        <div className="flex-1 min-w-0 space-y-14">

          {/* ── 1. Overview ─────────────────────────────────────── */}
          <section id="overview" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={BookOpen}
              title="Platform Overview"
              description="What My-Curator does and how it fits into the AV data pipeline."
            />
            <div className="card p-5 space-y-3">
              <p className="text-sm text-slate-300 leading-relaxed">
                <strong className="text-slate-100">My-Curator</strong> is a{" "}
                <strong className="text-slate-100">Verify-by-Exception (VBE)</strong> curation
                platform for autonomous-vehicle driving clips. A NVIDIA DeepStream 9.0 pipeline ingests raw
                video, a Scout VLM (Cosmos-Reason2-8B FP8) generates structured{" "}
                <strong className="text-slate-100">Scenario DNA</strong> for each clip, and
                reviewers curate only the exceptions — clips flagged as elevated or critical risk.
              </p>
              <p className="text-sm text-slate-400 leading-relaxed">
                Each clip carries a <strong className="text-slate-200">4-layer DNA descriptor</strong>:
                ODD (environment), Topology (road infrastructure), Actor Dynamics (road users), and
                Planner Logic (ego intent + risk). DNA is stored in PostgreSQL (JSONB + GIN index)
                and in Milvus (768-dim Cosmos-Embed1-336p embeddings) for hybrid search.
              </p>
              <div className="flex flex-wrap gap-2 pt-1">
                {["Verify-by-Exception", "4-Layer DNA", "Hybrid Vector + Filter Search", "ISO 21448 SOTIF", "ASAM OSI / OpenDRIVE Taxonomy"].map((t) => (
                  <span key={t} className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </section>

          {/* ── 2. Workflow ─────────────────────────────────────── */}
          <section id="workflow" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={Zap}
              title="How to Use"
              description="The four-step curation workflow from raw video to curated corpus."
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              {[
                { step: "1", label: "Ingest",          icon: Layers,       desc: "DS pipeline captures frames, Scout generates DNA, clip is stored in PG + Milvus with review_status = pending." },
                { step: "2", label: "Search & Curate", icon: Navigation,   desc: "Use the hybrid search page to query clips by natural language or DNA filters. Results are ranked by cosine similarity." },
                { step: "3", label: "Review Queue",    icon: Eye,          desc: "Work through pending clips. Click any card to open the full detail view with video playback, DNA accordion, and similar clips." },
                { step: "4", label: "Approve / Reject",icon: CheckCircle2, desc: "Approve clips to add them to the curated corpus. Reject duplicates, low-quality, or mislabeled clips." },
              ].map(({ step, label, icon: Icon, desc }) => (
                <div key={step} className="card p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 text-xs font-bold flex items-center justify-center shrink-0">
                      {step}
                    </span>
                    <Icon className="w-4 h-4 text-cyan-400 shrink-0" />
                    <span className="text-sm font-semibold text-slate-200">{label}</span>
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ── 3. Layer 1 ODD ──────────────────────────────────── */}
          <section id="odd" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L1"
              layerLabel="Layer 1"
              icon={Cloud}
              title="ODD — Operational Design Domain"
              description="Environmental conditions under which the clip was captured."
              source="ASAM OpenSCENARIO v1.0 · ASAM OSI v3.x · ISO 22736"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {ODD_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 4. Layer 2 Topology ─────────────────────────────── */}
          <section id="topology" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L2"
              layerLabel="Layer 2"
              icon={Map}
              title="Topology — Road Infrastructure"
              description="Road classification, lane configuration, and intersection geometry."
              source="OpenDRIVE v1.5M XSD"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {TOPOLOGY_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 5. Layer 3 Actors ───────────────────────────────── */}
          <section id="actors" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L3"
              layerLabel="Layer 3"
              icon={Users}
              title="Actor Dynamics — Dynamic Actors"
              description="Per-actor classification, behavioral state, and proximity for each detected road user. Each clip may contain zero or more actor entries."
              source="ASAM OSI VehicleClassification · PedestrianCategory · MiscObjectCategory"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {ACTOR_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 6. Layer 4 Planner ──────────────────────────────── */}
          <section id="planner" className="scroll-mt-6 space-y-4">
            <LayerHeader
              layerNum="L4"
              layerLabel="Layer 4"
              icon={Navigation}
              title="Planner Logic — Ego Intent &amp; Risk"
              description="Ego vehicle maneuver classification and ISO 21448 SOTIF risk level."
              source="WOD-E2E arXiv:2510.26125 · PEGASUS HAD-F · ISO 21448:2022 SOTIF"
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {PLANNER_FIELDS.map((f) => <FieldCard key={f.name} field={f} />)}
            </div>
          </section>

          {/* ── 7. Review States ────────────────────────────────── */}
          <section id="review-states" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={CheckCircle2}
              title="Review States"
              description="The four lifecycle states a clip occupies in the curation workflow."
            />
            <div className="card p-4 divide-y divide-[#1e3a5f]">
              {REVIEW_STATES.map(({ value, dot, badge, desc }) => (
                <div key={value} className="py-3 first:pt-0 last:pb-0 flex items-start gap-3">
                  <div className={`mt-1.5 w-2 h-2 rounded-full shrink-0 ${dot}`} />
                  <div>
                    <span className={`font-mono text-xs font-semibold px-2 py-0.5 rounded border ${badge}`}>
                      {value}
                    </span>
                    <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ── 8. Glossary ─────────────────────────────────────── */}
          <section id="glossary" className="scroll-mt-6 space-y-4">
            <LayerHeader
              icon={BookOpen}
              title="Glossary"
              description="Key terms and concepts used throughout the platform."
            />
            <div className="card p-4 divide-y divide-[#1e3a5f]">
              {GLOSSARY.map(({ term, def }) => (
                <div key={term} className="py-3 first:pt-0 last:pb-0">
                  <div className="text-xs font-semibold text-slate-200 mb-1">{term}</div>
                  <p className="text-xs text-slate-500 leading-relaxed">{def}</p>
                </div>
              ))}
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}
