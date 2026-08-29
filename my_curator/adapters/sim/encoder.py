"""Turn recorded frames into the three videos, through ``gst-launch-1.0``.

The simulator hands back raw BGRA frames; this module encodes them, labels them and builds
the side-by-side comparison against the source segment. GStreamer is driven as a
subprocess rather than through ``gi``: no Python bindings are installed anywhere this runs,
and the pipelines are simple enough that a command line expresses them fully.

``cv2`` is banned project-wide, so every pixel here moves through GStreamer.

Pipeline construction is pure and unit-testable; only :func:`run_pipeline` needs GStreamer
present.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

GST = "gst-launch-1.0"

#: One pane of the comparison view. Three of them make 1920x360.
PANE_WIDTH = 640
PANE_HEIGHT = 360

_RAW_FORMAT = "bgra"
_BYTES_PER_PIXEL = 4
_FONT = "Sans, 16"
_X264 = ("x264enc", "speed-preset=medium", "bitrate=4000")


class EncodingError(RuntimeError):
    """A GStreamer pipeline exited non-zero."""


def frame_bytes(width: int, height: int) -> int:
    return width * height * _BYTES_PER_PIXEL


def _overlay(text: str, valignment: str = "top") -> list[str]:
    return [
        "textoverlay",
        f"text={text}",
        f"valignment={valignment}",
        "halignment=left",
        f'font-desc="{_FONT}"',
        "shaded-background=true",
        "!",
        "videoconvert",
        "!",
    ]


def _raw_source(path: Path, width: int, height: int, fps: int) -> list[str]:
    return [
        "filesrc",
        f"location={path}",
        "!",
        "rawvideoparse",
        f"width={width}",
        f"height={height}",
        f"format={_RAW_FORMAT}",
        f"framerate={fps}/1",
        "!",
        "videoconvert",
        "!",
    ]


def _h264_sink(path: Path, fps: int) -> list[str]:
    return [
        *_X264,
        f"key-int-max={fps}",
        "!",
        "h264parse",
        "!",
        "qtmux",
        "!",
        "filesink",
        f"location={path}",
    ]


def view_pipeline(
    raw: Path,
    out: Path,
    *,
    width: int,
    height: int,
    fps: int,
    overlay: str,
) -> list[str]:
    """Encode one recorded view, burning in the overlay and a running clock."""
    clock = [
        "timeoverlay",
        "valignment=top",
        "halignment=right",
        f'font-desc="{_FONT}"',
        "shaded-background=true",
        "!",
        "videoconvert",
        "!",
    ]
    return [
        GST,
        "-q",
        *_raw_source(raw, width, height, fps),
        *_overlay(overlay),
        *clock,
        *_h264_sink(out, fps),
    ]


def extract_pipeline(source: Path, out: Path, fps: int) -> list[str]:
    """Decode a source clip to raw pane-sized frames, so a segment is a byte range."""
    return [
        GST,
        "-q",
        "filesrc",
        f"location={source}",
        "!",
        "decodebin",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        "videorate",
        "!",
        f"video/x-raw,format=BGRA,width={PANE_WIDTH},height={PANE_HEIGHT},framerate={fps}/1",
        "!",
        "filesink",
        f"location={out}",
    ]


def _pane(raw: Path, width: int, height: int, fps: int, label: str, sink: str) -> list[str]:
    scale: list[str] = []
    if (width, height) != (PANE_WIDTH, PANE_HEIGHT):
        scale = [
            "videoscale",
            "!",
            f"video/x-raw,width={PANE_WIDTH},height={PANE_HEIGHT}",
            "!",
            "videoconvert",
            "!",
        ]
    return [*_raw_source(raw, width, height, fps), *scale, *_overlay(label), sink]


def compare_pipeline(
    original_raw: Path,
    ego_raw: Path,
    chase_raw: Path,
    out: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> list[str]:
    """Compose original | ego | chase into one strip."""
    compositor = [
        "compositor",
        "name=mix",
        "background=black",
        "sink_0::xpos=0",
        f"sink_1::xpos={PANE_WIDTH}",
        f"sink_2::xpos={PANE_WIDTH * 2}",
        "!",
        f"video/x-raw,width={PANE_WIDTH * 3},height={PANE_HEIGHT}",
        "!",
        "videoconvert",
        "!",
        *_h264_sink(out, fps),
    ]
    return [
        GST,
        "-q",
        *compositor,
        *_pane(original_raw, PANE_WIDTH, PANE_HEIGHT, fps, "original", "mix.sink_0"),
        *_pane(ego_raw, width, height, fps, "synthetic ego", "mix.sink_1"),
        *_pane(chase_raw, width, height, fps, "synthetic chase", "mix.sink_2"),
    ]


def run_pipeline(command: list[str], *, what: str, timeout_s: int = 600) -> None:
    log.debug("%s: %s", what, " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        raise EncodingError(f"{what} failed: {(result.stderr or result.stdout).strip()[:400]}")


def slice_raw(source: Path, out: Path, *, start_s: float, duration_s: float, fps: int) -> int:
    """Copy the segment's frames out of a decoded clip. Fixed-size frames make this exact."""
    size = frame_bytes(PANE_WIDTH, PANE_HEIGHT)
    available = source.stat().st_size // size
    first = int(round(start_s * fps))
    wanted = int(round(duration_s * fps))
    count = max(0, min(wanted, available - first))
    if count <= 0:
        raise EncodingError(
            f"segment [{start_s}, {start_s + duration_s}] lies outside the decoded clip "
            f"({available} frames at {fps} fps)"
        )
    with source.open("rb") as src, out.open("wb") as dst:
        src.seek(first * size)
        remaining = count * size
        while remaining > 0:
            chunk = src.read(min(1 << 22, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    return count
