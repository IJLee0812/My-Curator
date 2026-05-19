"""POST /v1/search/video — video-to-video similarity search (P3-2).

Loads 8 frames from the MinIO frames bucket for the referenced clip_id,
encodes them via the Cosmos-Embed1-336p video tower, and returns the
nearest-neighbour clips from Milvus.

The frames key prefix is read from ``clips.frames_blob_uri`` in PG (set by
CurationConsumer when the DS pipeline message carries a frames blob URI).
Clips ingested via ``/v1/ingest`` (no frames upload) have NULL here and
return 422 — video-search only applies to DS-pipeline-originated clips.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from my_curator.adapters.embed.text_tower import CosmosEmbed1Encoder
from my_curator.adapters.storage.frame_loader import load_frames
from my_curator.adapters.storage.milvus import MilvusRepository
from my_curator.adapters.storage.minio import MinIORepository
from my_curator.adapters.storage.pg import PGRepository

from ..deps import get_embedder, get_milvus, get_minio, get_pg
from .search import ClipResult, SearchResponse

router = APIRouter()


class VideoSearchRequest(BaseModel):
    clip_id: str
    limit: int = Field(default=20, ge=1, le=1000)
    top_k: int = Field(default=1000, ge=1, le=10000)


@router.post("/v1/search/video", response_model=SearchResponse)
async def search_video(
    req: VideoSearchRequest,
    milvus: MilvusRepository = Depends(get_milvus),
    minio: MinIORepository = Depends(get_minio),
    pg: PGRepository = Depends(get_pg),
    embedder: CosmosEmbed1Encoder = Depends(get_embedder),
) -> SearchResponse:
    try:
        clip_uuid = UUID(req.clip_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="clip_id must be a valid UUID") from exc

    row = await pg.get_clip_with_blob_uri(clip_uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Clip {req.clip_id} not found")

    key_prefix = row.get("frames_blob_uri")
    if not key_prefix:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Clip {req.clip_id} has no frames_blob_uri — video-search "
                "is only supported for DS pipeline clips with captured frames."
            ),
        )

    try:
        frames_tensor = await load_frames(minio, "frames", key_prefix)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Frames not found at {key_prefix} for clip {req.clip_id}: {exc}",
        ) from exc

    query_vec = await asyncio.to_thread(embedder.encode_video, frames_tensor)
    milvus_results = await milvus.search(query_vec, top_k=req.top_k)

    if not milvus_results:
        return SearchResponse(results=[], total=0)

    clip_ids: list[UUID] = [r["clip_id"] for r in milvus_results]
    score_map: dict[UUID, float] = {r["clip_id"]: r["score"] for r in milvus_results}

    pg_results = await pg.filter_dna_by_ids(clip_ids, {}, limit=len(clip_ids))
    ranked = sorted(pg_results, key=lambda r: score_map.get(r["clip_id"], 0.0), reverse=True)
    ranked = ranked[: req.limit]

    results = [
        ClipResult(
            clip_id=str(r["clip_id"]),
            score=score_map.get(r["clip_id"], 0.0),
            dna_json=r.get("dna_json"),
            start_s=r.get("start_s"),
            end_s=r.get("end_s"),
            blob_uri=r.get("blob_uri"),
            is_gold=r.get("is_gold"),
            source_clip_id=r.get("source_clip_id"),
        )
        for r in ranked
    ]
    return SearchResponse(results=results, total=len(results))
