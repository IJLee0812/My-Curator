"""Serve file:// video segments via HTTP range requests (no file writes).

The curation-api mounts VIDEO_DATA_ROOT read-only at /video_data.
blob_uri "file://{relative}" is resolved against that mount point.

FastAPI's FileResponse handles Accept-Ranges / byte-range negotiation
natively via Starlette, so the browser #t= fragment seek works without
any server-side slicing.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse


def _video_data_root() -> str:
    root = os.environ.get("VIDEO_DATA_ROOT", "")
    if not root:
        raise HTTPException(
            status_code=503,
            detail="VIDEO_DATA_ROOT not configured — NAS mount unavailable",
        )
    return root


def resolve_path(blob_uri: str) -> Path:
    """Convert file://{relative} to absolute path under VIDEO_DATA_ROOT."""
    if not blob_uri.startswith("file://"):
        raise HTTPException(
            status_code=422,
            detail=f"blob_uri scheme not streamable: {blob_uri!r}",
        )
    rel = blob_uri[len("file://"):]
    return Path(_video_data_root()) / rel


async def serve_segment(blob_uri: str) -> FileResponse:
    """Return a FileResponse for the NAS video at blob_uri.

    The caller is responsible for appending #t=start,end to the URL
    returned to the client; this endpoint serves the full file with
    byte-range support so the browser can seek to the right position.
    """
    path = resolve_path(blob_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on NAS")
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )
