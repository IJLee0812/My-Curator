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

from src.storage.minio import MinIORepository
from src.storage.pg import PGRepository
from src.streaming.base import serve_segment
from src.streaming.minio import get_presigned_url as _get_presigned_url
from src.streaming.timestamp import get_precise_times

from .deps import get_minio, get_pg

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
    is_gold: bool
    source_clip_id: str | None
    dna_version: str | None
    dna_json: dict[str, Any] | None
    presigned_url: str | None


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
    is_gold: bool
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
        is_gold=bool(row.get("is_gold", False)),
        source_clip_id=row.get("source_clip_id"),
        dna_version=row.get("dna_version"),
        dna_json=row.get("dna_json"),
        presigned_url=url,
    )


@router.get("/v1/clips/{clip_id}/thumbnail")
async def thumbnail_clip(
    clip_id: str,
    pg: PGRepository = Depends(get_pg),
    minio: MinIORepository = Depends(get_minio),
):
    """Redirect to the presigned URL for the first extracted frame (frame_0.jpg).

    frames_blob_uri stored in DB = "frames/{session_id}/{clip_id}" (key prefix).
    MinIO bucket = "frames", key = "{frames_blob_uri}/frame_0.jpg".
    Returns 404 for clips without frames_blob_uri (legacy stream:// or /v1/ingest path).
    """
    try:
        uid = UUID(clip_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="clip_id must be a valid UUID") from None

    row = await pg.get_clip_with_blob_uri(uid)
    if row is None:
        raise HTTPException(status_code=404, detail="clip not found")

    frames_blob_uri: str | None = row.get("frames_blob_uri")
    if not frames_blob_uri:
        raise HTTPException(status_code=404, detail="No frames available for this clip")

    first_frame_key = f"{frames_blob_uri}/frame_0.jpg"
    try:
        data = await minio.download_bytes("frames", first_frame_key)
    except Exception:
        raise HTTPException(status_code=404, detail="Frame not found in MinIO") from None

    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/v1/clips/{clip_id}/stream")
async def stream_clip(
    clip_id: str,
    pg: PGRepository = Depends(get_pg),
    minio: MinIORepository = Depends(get_minio),
):
    """Stream the source video segment for a clip.

    Dispatch by blob_uri scheme:
      file://  → byte-range FileResponse from VIDEO_DATA_ROOT mount (NAS / KITTI / nuScenes)
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
            is_gold=bool(r.get("is_gold", False)),
            source_clip_id=r.get("source_clip_id"),
            dna_version=r.get("dna_version"),
            dna_json=r.get("dna_json"),
        )
        for r in rows
    ]
    return ClipListResponse(clips=clips, total=len(clips))
