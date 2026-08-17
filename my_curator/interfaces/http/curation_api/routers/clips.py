"""GET /v1/clips/{id} + GET /v1/clips — clip detail and listing (P3-2 / P3-4).

P3-2 introduced the per-clip detail endpoint with DNA JSON and a MinIO
presigned URL.  P3-4 extends ``ClipDetail`` with ``source_clip_id`` and
``frames_blob_uri`` (UI metadata) and adds a list endpoint that drives the
Dashboard "recent clips" widget.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from my_curator.adapters.storage.minio import MinIORepository
from my_curator.adapters.storage.pg import PGRepository
from my_curator.adapters.storage.streaming import (
    get_presigned_url as _get_presigned_url,
)
from my_curator.adapters.storage.streaming import serve_segment
from my_curator.domain.timestamp import get_precise_times

from ..deps import get_minio, get_pg

router = APIRouter()


class ClipDetail(BaseModel):
    clip_id: str
    session_id: str
    blob_uri: str
    frames_blob_uri: str | None
    start_s: float
    end_s: float
    precise_start_s: float
    precise_end_s: float
    source_clip_id: str | None
    dna_version: str | None
    dna_json: dict[str, Any] | None
    presigned_url: str | None
    review_status: str


class ClipSummary(BaseModel):
    """Slim clip row used by the recent-clips list (Dashboard).

    No presigned URL: the list view does not embed videos, so we skip the
    extra MinIO sign step per row.
    """

    clip_id: str
    session_id: str
    blob_uri: str
    frames_blob_uri: str | None
    start_s: float
    end_s: float
    source_clip_id: str | None
    dna_version: str | None
    dna_json: dict[str, Any] | None


class ClipListResponse(BaseModel):
    clips: list[ClipSummary]
    total: int


@router.get("/v1/clips/{clip_id}", response_model=ClipDetail)
async def get_clip(
    clip_id: str,
    pg: PGRepository = Depends(get_pg),
    minio: MinIORepository = Depends(get_minio),
) -> ClipDetail:
    try:
        uid = UUID(clip_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="clip_id must be a valid UUID") from None

    row = await pg.get_clip_with_blob_uri(uid)
    if row is None:
        raise HTTPException(status_code=404, detail="clip not found")

    blob_uri: str = row["blob_uri"]
    start_s: float = row["start_s"]
    end_s: float = row["end_s"]
    url: str | None = await _get_presigned_url(minio, blob_uri)

    video_root = os.environ.get("VIDEO_DATA_ROOT", "")
    precise_start_s, precise_end_s = get_precise_times(blob_uri, start_s, end_s, video_root)

    return ClipDetail(
        clip_id=str(row["clip_id"]),
        session_id=row["session_id"],
        blob_uri=blob_uri,
        frames_blob_uri=row.get("frames_blob_uri"),
        start_s=start_s,
        end_s=end_s,
        precise_start_s=precise_start_s,
        precise_end_s=precise_end_s,
        source_clip_id=row.get("source_clip_id"),
        dna_version=row.get("dna_version"),
        dna_json=row.get("dna_json"),
        presigned_url=url,
        review_status=row.get("review_status") or "pending",
    )


FRAMES_PER_CLIP = 8


async def _sibling_frame_key(pg: PGRepository, row: dict) -> str | None:
    """Frame key of a sibling segment covering this clip's start, or None.

    Segments of one source overlap, so the frame returned is a real frame of
    this clip's own time range. The 8 frames are sampled evenly across the
    sibling's window, so the index closest to that instant is the one whose
    position matches it proportionally.
    """
    source_clip_id = row.get("source_clip_id")
    start_s = row.get("start_s")
    if not source_clip_id or start_s is None:
        return None

    at_s = float(start_s)
    sibling = await pg.find_frames_sibling(source_clip_id, at_s)
    if sibling is None:
        return None

    span = float(sibling["end_s"]) - float(sibling["start_s"])
    frac = (at_s - float(sibling["start_s"])) / span if span > 0 else 0.0
    idx = round(min(max(frac, 0.0), 1.0) * (FRAMES_PER_CLIP - 1))
    return f"{sibling['frames_blob_uri']}/frame_{idx}.jpg"


@router.get("/v1/clips/{clip_id}/thumbnail")
async def thumbnail_clip(
    clip_id: str,
    pg: PGRepository = Depends(get_pg),
    minio: MinIORepository = Depends(get_minio),
):
    """Serve the first extracted frame (frame_0.jpg) as the clip's thumbnail.

    frames_blob_uri stored in DB = "frames/{session_id}/{clip_id}" (key prefix).
    MinIO bucket = "frames", key = "{frames_blob_uri}/frame_0.jpg".
    Clips stored without frames of their own fall back to a sibling segment
    covering the same instant; 404 only when no such frame exists.
    """
    try:
        uid = UUID(clip_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="clip_id must be a valid UUID") from None

    row = await pg.get_clip_with_blob_uri(uid)
    if row is None:
        raise HTTPException(status_code=404, detail="clip not found")

    frames_blob_uri: str | None = row.get("frames_blob_uri")
    if frames_blob_uri:
        frame_key = f"{frames_blob_uri}/frame_0.jpg"
    else:
        frame_key = await _sibling_frame_key(pg, row)
        if frame_key is None:
            raise HTTPException(status_code=404, detail="No frames available for this clip")

    try:
        data = await minio.download_bytes("frames", frame_key)
    except Exception:
        raise HTTPException(status_code=404, detail="Frame not found in MinIO") from None

    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            # Anti-download hardening (#44): suppress the browser's
            # default "save as" prompt by serving the response inline
            # with an empty filename, drop disk-cache so the bytes do
            # not persist after the tab is closed, and block MIME
            # sniffing so the response cannot be reinterpreted as
            # another downloadable type.
            "Content-Disposition": 'inline; filename=""',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/v1/clips/{clip_id}/stream")
async def stream_clip(
    clip_id: str,
    pg: PGRepository = Depends(get_pg),
    minio: MinIORepository = Depends(get_minio),
):
    """Stream the source video segment for a clip.

    Dispatch by blob_uri scheme:
      file://  → byte-range FileResponse from VIDEO_DATA_ROOT mount
      minio:// → HTTP redirect to a short-lived presigned GET URL
      stream:// → 404 (no path recorded for legacy clips — re-ingest required)
    """
    try:
        uid = UUID(clip_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="clip_id must be a valid UUID") from None

    row = await pg.get_clip_with_blob_uri(uid)
    if row is None:
        raise HTTPException(status_code=404, detail="clip not found")

    blob_uri: str = row["blob_uri"]

    if blob_uri.startswith("file://"):
        return await serve_segment(blob_uri)

    presigned = await _get_presigned_url(minio, blob_uri)
    if presigned:
        return RedirectResponse(url=presigned, status_code=302)

    raise HTTPException(
        status_code=404,
        detail=(
            "No streamable source for this clip. "
            "stream:// URIs have no path info — re-ingest with a file:// source."
        ),
    )


@router.get("/v1/clips", response_model=ClipListResponse)
async def list_clips(
    limit: int = Query(default=20, ge=1, le=100),
    pg: PGRepository = Depends(get_pg),
) -> ClipListResponse:
    """Return the most recently inserted clips (P3-4).

    Drives the Dashboard "recent clips" widget.  Ordered by ``created_at DESC``.
    """
    rows = await pg.list_clips(limit=limit)
    clips = [
        ClipSummary(
            clip_id=str(r["clip_id"]),
            session_id=r["session_id"],
            blob_uri=r["blob_uri"],
            frames_blob_uri=r.get("frames_blob_uri"),
            start_s=r["start_s"],
            end_s=r["end_s"],
            source_clip_id=r.get("source_clip_id"),
            dna_version=r.get("dna_version"),
            dna_json=r.get("dna_json"),
        )
        for r in rows
    ]
    return ClipListResponse(clips=clips, total=len(clips))
