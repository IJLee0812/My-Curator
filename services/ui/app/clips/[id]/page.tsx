import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";

import DnaAccordion from "@/components/dna-accordion";
import { RiskBadge } from "@/components/dna-badges";
import { getClip } from "@/lib/api";

import ApproveRejectButtons from "./ApproveRejectButtons";
import SimilarClipsPanel from "./SimilarClipsPanel";
import VideoPlayer from "./VideoPlayer";

export const dynamic = "force-dynamic";

type PageProps = {
  // Next 16: dynamic route params arrive as a Promise.
  params: Promise<{ id: string }>;
};

export default async function ClipDetailPage({ params }: PageProps) {
  const { id } = await params;

  let clip;
  try {
    clip = await getClip(id);
  } catch (err) {
    const status = (err as Error & { status?: number }).status;
    if (status === 404 || status === 422) notFound();
    throw err;
  }

  const dna = clip.dna_json;
  const risk = dna?.planner_logic?.risk_level ?? "nominal";

  return (
    <div className="p-6 space-y-5 max-w-5xl">
      {/* back + title */}
      <div className="flex items-center gap-3">
        <Link
          href="/search"
          className="p-2 rounded-lg hover:bg-[#111f36] transition-colors"
        >
          <ArrowLeft className="w-4 h-4 text-slate-400" />
        </Link>
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-base font-bold text-slate-100 font-mono truncate">
              {clip.clip_id}
            </h1>
            {clip.is_gold && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                gold set
              </span>
            )}
            <RiskBadge level={risk} />
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            {clip.start_s.toFixed(2)}s – {clip.end_s.toFixed(2)}s · session: {clip.session_id}
            {clip.source_clip_id && (
              <>
                {" · "}
                <span className="text-slate-400">
                  source: <span className="font-mono">{clip.source_clip_id}</span>
                </span>
              </>
            )}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        {/* left: video + actions */}
        <div className="lg:col-span-2 space-y-4">
          <VideoPlayer
            clipId={clip.clip_id}
            presignedUrl={clip.presigned_url}
            blobUri={clip.blob_uri}
            framesBlobUri={clip.frames_blob_uri}
            startS={clip.precise_start_s}
            endS={clip.precise_end_s}
            durationSeconds={clip.end_s - clip.start_s}
          />

          <ApproveRejectButtons
            clipId={clip.clip_id}
            dnaVersion={clip.dna_version}
            initialStatus={clip.review_status}
          />
        </div>

        {/* right: 4-Layer DNA accordion */}
        <div className="lg:col-span-3">
          <DnaAccordion dna={dna} />
        </div>
      </div>

      {/* similar clips */}
      <SimilarClipsPanel clipId={clip.clip_id} dna={dna} />
    </div>
  );
}
