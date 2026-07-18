"use client";

import { useEffect, useRef, useState } from "react";
import { Film, Pause, Play, RefreshCw, RotateCcw } from "lucide-react";

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
    // When endS equals the file's natural duration, the browser fires `ended`
    // before `timeupdate` can seek back.  `ended` also resets currentTime to 0
    // before the next `play` event, so the >= endS guard in onPlay would miss.
    // Fix: handle `ended` explicitly (restart loop) and also catch currentTime < startS.
    const onEnded = () => { video.currentTime = startS; video.play(); };
    const onPlay  = () => { if (video.currentTime < startS || video.currentTime >= endS) video.currentTime = startS; };
    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("timeupdate",     onTimeUpdate);
    video.addEventListener("ended",          onEnded);
    video.addEventListener("play",           onPlay);
    return () => {
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("timeupdate",     onTimeUpdate);
      video.removeEventListener("ended",          onEnded);
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
        <div className="aspect-video bg-canvas border-b border-line flex flex-col items-center justify-center gap-3 px-6 text-center">
          <Film className="w-10 h-10 text-faint" />
          <div>
            <p className="text-sm font-medium text-muted">No streamable video source</p>
            <p className="text-xs text-faint mt-1">
              This clip has no presigned URL and its blob URI is not a local file stream.
              The raw video may have been evicted from MinIO or was never uploaded.
            </p>
          </div>
          <div className="w-full max-w-sm bg-surface-2 border border-line rounded px-3 py-2 text-left">
            <p className="text-[10px] text-faint mb-1 uppercase tracking-wider">blob_uri</p>
            <p className="text-[11px] text-muted font-mono break-all">{blobUri}</p>
          </div>
        </div>
        <div className="p-3 flex items-center justify-between text-xs text-muted">
          <span className="font-mono">{startS.toFixed(2)}s – {endS.toFixed(2)}s</span>
          <span className="text-faint">{framesBlobUri ? "frames available" : "no frames"}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <div
        className="aspect-video bg-canvas border-b border-line relative group cursor-pointer"
        onClick={togglePlay}
        onContextMenu={(e) => e.preventDefault()}
        onDragStart={(e) => e.preventDefault()}
      >
        <video
          ref={videoRef}
          src={videoSrc}
          autoPlay
          muted
          loop={!isNasStream}
          preload="metadata"
          controlsList="nodownload noremoteplayback noplaybackrate"
          disablePictureInPicture
          disableRemotePlayback
          draggable={false}
          onContextMenu={(e) => e.preventDefault()}
          onDragStart={(e) => e.preventDefault()}
          className="w-full h-full select-none"
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
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-canvas/85 text-center px-6">
            <RefreshCw className="w-6 h-6 text-amber-600 dark:text-amber-400 mb-2" />
            <div className="text-sm text-ink font-medium">
              Presigned URL is about to expire
            </div>
            <div className="text-xs text-muted mt-1">
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

      <div className="p-3 flex items-center justify-between text-xs text-muted">
        <span className="font-mono flex items-center gap-1.5">
          <RotateCcw className="w-3 h-3 text-accent/60" />
          {startS.toFixed(2)}s – {endS.toFixed(2)}s
        </span>
        <span className="text-faint truncate max-w-[60%]" title={blobUri}>
          {blobUri.replace(/^file:\/\//, "").replace(/^[a-z]+:\/\//, "")}
        </span>
      </div>
    </div>
  );
}
