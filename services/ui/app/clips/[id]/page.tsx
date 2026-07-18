import { notFound } from "next/navigation";

import DnaAccordion from "@/components/dna-accordion";
import { RiskBadge } from "@/components/dna-badges";
import { getClip } from "@/lib/api";

import ApproveRejectButtons from "./ApproveRejectButtons";
import BackButton from "./BackButton";
import SimilarClipsPanel from "./SimilarClipsPanel";
import VideoPlayer from "./VideoPlayer";

export const dynamic = "force-dynamic";

type PageProps = {
  // Next 16: dynamic route params arrive as a Promise.
  params: Promise<{ id: string }>;
  searchParams: Promise<{ from?: string }>;
};

export default async function ClipDetailPage({ params, searchParams }: PageProps) {
  const { id } = await params;
  const { from } = await searchParams;
  const backHref = from === "review" ? "/review" : "/search";

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
    <div className="p-4 sm:p-6 space-y-5 w-full">
      {/* back + title */}
      <div className="flex items-center gap-3">
        <BackButton fallback={backHref} />
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-sm sm:text-base font-bold text-slate-100 font-mono truncate">
              {clip.clip_id}
            </h1>
            <RiskBadge level={risk} />
          </div>
          <p className="text-xs text-slate-500 mt-0.5 flex flex-wrap gap-x-1.5">
            <span>{clip.start_s.toFixed(2)}s – {clip.end_s.toFixed(2)}s</span>
            <span>·</span>
            <span>session: {clip.session_id}</span>
            {clip.source_clip_id && (
              <>
                <span>·</span>
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
        <div className="lg:col-span-3 space-y-4">
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
        <div className="lg:col-span-2">
          <DnaAccordion dna={dna} />
        </div>
      </div>

      {/* similar clips */}
      <SimilarClipsPanel clipId={clip.clip_id} dna={dna} />
    </div>
  );
}
