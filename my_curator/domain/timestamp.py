"""Parse timestamp sidecar files for frame-accurate segment seeking.

Format:
    FPS,{fps}
    Size,{width},{height}
    {frame_idx},{unix_ts_us}
    ...

Usage::
    precise_start, precise_end = get_precise_times(
        blob_uri="file://session/00123/video/00123.mp4",
        start_s=10.5,
        end_s=15.5,
        video_data_root="/video_data",
    )
    # Falls back to start_s/end_s when sidecar absent or unreadable.
"""

from __future__ import annotations

from pathlib import Path


def parse_timestamp_file(ts_path: Path) -> tuple[int, list[int]]:
    """Return (fps, [unix_ts_us per frame]) from a .timestamp sidecar.

    Skips header lines (FPS,... and Size,...).
    Raises ValueError on malformed files.
    """
    fps = 30
    timestamps: list[int] = []
    with ts_path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("FPS,"):
                fps = int(line.split(",", 1)[1])
            elif line.startswith("Size,"):
                continue
            else:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    timestamps.append(int(parts[1]))
    return fps, timestamps


def _frame_aligned_times(
    start_s: float,
    end_s: float,
    fps: int,
    total_frames: int,
) -> tuple[float, float]:
    start_frame = min(round(start_s * fps), total_frames - 1)
    end_frame = min(round(end_s * fps), total_frames - 1)
    return start_frame / fps, end_frame / fps


def get_precise_times(
    blob_uri: str,
    start_s: float,
    end_s: float,
    video_data_root: str,
) -> tuple[float, float]:
    """Return frame-aligned (start_s, end_s) using the timestamp sidecar.

    Falls back to the raw DB values when:
    - blob_uri is not file://
    - the resolved path would escape ``video_data_root``
      (directory-traversal defence; mirrors
      :func:`my_curator.adapters.storage.streaming.resolve_path`)
    - sidecar file does not exist
    - sidecar is unreadable / malformed

    Silent fall-through (rather than raising) is preferred here because
    precise times are best-effort metadata — a hostile blob_uri must
    not break the primary ``GET /v1/clips/{id}`` response.
    """
    if not blob_uri.startswith("file://"):
        return start_s, end_s

    rel = blob_uri[len("file://") :].lstrip("/")
    try:
        root_resolved = Path(video_data_root).resolve()
        video_path = (root_resolved / rel).resolve()
        if not video_path.is_relative_to(root_resolved):
            # Containment escape — refuse to open the sidecar; fall back
            # to the caller-supplied start_s / end_s.
            return start_s, end_s
    except (OSError, RuntimeError):
        # resolve() may raise on broken symlinks / permission errors.
        return start_s, end_s
    ts_path = video_path.with_suffix(".timestamp")

    if not ts_path.exists():
        return start_s, end_s

    try:
        fps, timestamps = parse_timestamp_file(ts_path)
        if not timestamps:
            return start_s, end_s
        return _frame_aligned_times(start_s, end_s, fps, len(timestamps))
    except Exception:
        return start_s, end_s
