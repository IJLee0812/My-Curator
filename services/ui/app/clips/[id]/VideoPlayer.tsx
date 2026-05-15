"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink, Film, RefreshCw } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8001";

// MinIO presigned URLs are signed for 3600 s.  Show a reload prompt a touch
// before that so the user is not surprised by a 403 mid-watch.
const TTL_MS = 3600 * 1000;
const WARN_BEFORE_MS = 60 * 1000;

export default function VideoPlayer({
  clipId,
  presignedUrl,
  blobUri,
  framesBlobUri,
  startS,
  endS,
  durationSeconds,
}: {
  clipId: string;
  presignedUrl: string | null;
  blobUri: string;
  framesBlobUri: string | null;
  startS: number;
  endS: number;
  durationSeconds: number;
}) {
  const [expired, setExpired] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Use presigned URL for MinIO blobs (already trimmed segments, no seek needed).
  // For NAS file:// blobs serve the full source; JS enforces startS/endS boundaries
  // because browser #t= fragment support for end time is unreliable.
  const isNasStream = !presignedUrl && blobUri.startsWith("file://");
  const videoSrc = presignedUrl ?? (isNasStream ? `${API_BASE}/v1/clips/${clipId}/stream` : null);

  useEffect(() => {
    setExpired(false);
    if (!presignedUrl) return;
    const id = setTimeout(() => setExpired(true), TTL_MS - WARN_BEFORE_MS);
    return () => clearTimeout(id);
  }, [presignedUrl]);

  // Enforce segment boundaries for NAS streams.
  // - loadedmetadata: seek to startS on load.
  // - timeupdate: pause when segment end is reached.
  // - play: if currentTime is at/past endS (user pressed play after segment ended),
  //   reset to startS so the segment loops from the beginning.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isNasStream) return;
    const onLoadedMetadata = () => { video.currentTime = startS; };
    const onTimeUpdate = () => { if (video.currentTime >= endS) video.pause(); };
    const onPlay = () => { if (video.currentTime >= endS) video.currentTime = startS; };
    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("play", onPlay);
    return () => {
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("play", onPlay);
    };
  }, [isNasStream, startS, endS]);

  if (!videoSrc) {
    return (
      <div className="card overflow-hidden">
        <div className="aspect-video bg-[#060c18] border-b border-[#1e3a5f] flex flex-col items-center justify-center gap-2">
          <Film className="w-10 h-10 text-slate-700" />
          <p className="text-xs text-slate-500">No video source available</p>
          <p className="text-[10px] text-slate-600 font-mono">{blobUri}</p>
        </div>
        <div className="p-3 text-xs text-slate-600 font-mono">
          {durationSeconds.toFixed(2)}s · {framesBlobUri ?? "no frames"}
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <div className="aspect-video bg-[#060c18] border-b border-[#1e3a5f] relative">
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          preload="metadata"
          className="w-full h-full"
        />
        {expired && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#060c18]/85 text-center px-6">
            <RefreshCw className="w-6 h-6 text-amber-400 mb-2" />
            <div className="text-sm text-slate-200 font-medium">
              Presigned URL is about to expire
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Reload the page to fetch a fresh URL from curation-api.
            </div>
            <button
              onClick={() => window.location.reload()}
              className="mt-3 btn-primary text-xs"
            >
              Reload
            </button>
          </div>
        )}
      </div>
      <div className="p-3 flex items-center justify-between text-xs text-slate-500">
        <span className="font-mono">
          {durationSeconds.toFixed(2)}s · {framesBlobUri ?? "no frames"}
        </span>
        <a
          href={videoSrc}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300"
          title={blobUri}
        >
          <ExternalLink className="w-3 h-3" /> Open MP4
        </a>
      </div>
    </div>
  );
}
