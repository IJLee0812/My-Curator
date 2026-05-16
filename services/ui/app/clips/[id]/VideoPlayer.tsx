"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink, Film, Pause, Play, RefreshCw, RotateCcw } from "lucide-react";

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
  durationSeconds: _durationSeconds,
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
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const isNasStream = !presignedUrl && blobUri.startsWith("file://");
  const videoSrc = presignedUrl ?? (isNasStream ? `${API_BASE}/v1/clips/${clipId}/stream` : null);

  useEffect(() => {
    setExpired(false);
    if (!presignedUrl) return;
    const id = setTimeout(() => setExpired(true), TTL_MS - WARN_BEFORE_MS);
    return () => clearTimeout(id);
  }, [presignedUrl]);

  // Sync playing state with video events for overlay rendering.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onPlay  = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    video.addEventListener("play",  onPlay);
    video.addEventListener("pause", onPause);
    return () => {
      video.removeEventListener("play",  onPlay);
      video.removeEventListener("pause", onPause);
    };
  }, []);

  // NAS streams: enforce segment boundaries + loop by seeking back to startS.
  // Presigned clips are already trimmed; native `loop` attribute handles repeat.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isNasStream) return;
    const onLoadedMetadata = () => { video.currentTime = startS; };
    const onTimeUpdate = () => { if (video.currentTime >= endS) video.currentTime = startS; };
    const onPlay = () => { if (video.currentTime >= endS) video.currentTime = startS; };
    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("timeupdate",     onTimeUpdate);
    video.addEventListener("play",           onPlay);
    return () => {
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("timeupdate",     onTimeUpdate);
      video.removeEventListener("play",           onPlay);
    };
  }, [isNasStream, startS, endS]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    video.paused ? video.play() : video.pause();
  };

  if (!videoSrc) {
    return (
      <div className="card overflow-hidden">
        <div className="aspect-video bg-[#060c18] border-b border-[#1e3a5f] flex flex-col items-center justify-center gap-3 px-6 text-center">
          <Film className="w-10 h-10 text-slate-700" />
          <div>
            <p className="text-sm font-medium text-slate-400">No streamable video source</p>
            <p className="text-xs text-slate-600 mt-1">
              This clip has no presigned URL and its blob URI is not a local file stream.
              The raw video may have been evicted from MinIO or was never uploaded.
            </p>
          </div>
          <div className="w-full max-w-sm bg-[#0a1120] border border-[#1e3a5f] rounded px-3 py-2 text-left">
            <p className="text-[10px] text-slate-600 mb-1 uppercase tracking-wider">blob_uri</p>
            <p className="text-[11px] text-slate-500 font-mono break-all">{blobUri}</p>
          </div>
        </div>
        <div className="p-3 flex items-center justify-between text-xs text-slate-500">
          <span className="font-mono">{startS.toFixed(2)}s – {endS.toFixed(2)}s</span>
          <span className="text-slate-600">{framesBlobUri ? "frames available" : "no frames"}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <div
        className="aspect-video bg-[#060c18] border-b border-[#1e3a5f] relative group cursor-pointer"
        onClick={togglePlay}
      >
        <video
          ref={videoRef}
          src={videoSrc}
          autoPlay
          muted
          loop={!isNasStream}
          preload="metadata"
          className="w-full h-full"
        />

        {/* Play/Pause overlay — always visible when paused, fades in on hover while playing */}
        <div
          className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 pointer-events-none ${
            playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
          }`}
        >
          <div className="w-14 h-14 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center border border-white/20">
            {playing ? (
              <Pause className="w-6 h-6 text-white" />
            ) : (
              <Play className="w-6 h-6 text-white ml-0.5" />
            )}
          </div>
        </div>

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
              onClick={(e) => { e.stopPropagation(); window.location.reload(); }}
              className="mt-3 btn-primary text-xs"
            >
              Reload
            </button>
          </div>
        )}
      </div>

      <div className="p-3 flex items-center justify-between text-xs text-slate-500">
        <span className="font-mono flex items-center gap-1.5">
          <RotateCcw className="w-3 h-3 text-cyan-500/60" />
          {startS.toFixed(2)}s – {endS.toFixed(2)}s
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
