"""GET /v1/clips/{id} — clip detail with DNA JSON and MinIO presigned URL (P3-2)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.storage.minio import MinIORepository
from src.storage.pg import PGRepository

from .deps import get_minio, get_pg

router = APIRouter()


class ClipDetail(BaseModel):
    clip_id: str
    session_id: str
    blob_uri: str
    start_s: float
    end_s: float
    dna_version: str | None
    dna_json: dict[str, Any] | None
    presigned_url: str


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
    bucket, _, key = blob_uri.partition("/")
    url = await minio.presigned_url(bucket, key, expires_in=3600)

    return ClipDetail(
        clip_id=str(row["clip_id"]),
        session_id=row["session_id"],
        blob_uri=blob_uri,
        start_s=row["start_s"],
        end_s=row["end_s"],
        dna_version=row.get("dna_version"),
        dna_json=row.get("dna_json"),
        presigned_url=url,
    )
