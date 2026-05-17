"""Video streaming helpers for curation-api (P3-4+) — unified module.

Combines the byte-range file:// streaming path (formerly src/streaming/base.py)
and the minio:// / legacy presigned-URL resolver (formerly src/streaming/minio.py).

Public surface:
  resolve_path(blob_uri)        — file:// → absolute path under VIDEO_DATA_ROOT
  serve_segment(blob_uri)       — FastAPI FileResponse with Accept-Ranges
  get_presigned_url(minio, blob_uri)
                                 — minio:// / bare bucket/key → presigned GET URL
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from my_curator.adapters.storage.minio import MinIORepository


def _video_data_root() -> str:
    root = os.environ.get("VIDEO_DATA_ROOT", "")
    if not root:
        raise HTTPException(
            status_code=503,
            detail="VIDEO_DATA_ROOT not configured — video data mount unavailable",
        )
    return root


def resolve_path(blob_uri: str) -> Path:
    """Convert file://{relative} to absolute path under VIDEO_DATA_ROOT."""
    if not blob_uri.startswith("file://"):
        raise HTTPException(
            status_code=422,
            detail=f"blob_uri scheme not streamable: {blob_uri!r}",
        )
    rel = blob_uri[len("file://") :]
    return Path(_video_data_root()) / rel


async def serve_segment(blob_uri: str) -> FileResponse:
    """Return a FileResponse for the video at blob_uri.

    The caller is responsible for appending #t=start,end to the URL returned to
    the client; this endpoint serves the full file with byte-range support so
    the browser can seek to the right position.
    """
    path = resolve_path(blob_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


async def get_presigned_url(
    minio: MinIORepository,
    blob_uri: str,
    *,
    expires_in: int = 3600,
) -> str | None:
    """Return a presigned GET URL for a MinIO-backed blob_uri.

    Supports two formats:
      minio://bucket/key          new canonical form
      bucket/key                  legacy form (clips/, raw/, frames/, artifacts/)

    Returns None for unrecognised or non-MinIO schemes (stream://, file://)
    so the caller can fall back to a null presigned_url without raising.
    """
    if blob_uri.startswith("minio://"):
        path = blob_uri[len("minio://") :]
    elif blob_uri.startswith(("stream://", "file://")):
        return None
    else:
        path = blob_uri

    bucket, _, key = path.partition("/")
    if not bucket or not key:
        return None

    return await minio.presigned_url(bucket, key, expires_in=expires_in)
