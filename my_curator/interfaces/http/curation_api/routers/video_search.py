"""POST /v1/search/video — find clips similar to a reference clip (P3-2 / P4-7).

Given a clip_id, builds the reference clip's narrative-text vector (from its DNA)
and, when frames exist, its video vector (Cosmos-Embed1 video tower), then runs
hybrid similarity over the dual-vector collection.  Clips without frames fall
back to text-only similarity (they used to 422).  The reference clip is excluded
from its own results.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from my_curator.adapters.embed.text_tower import CosmosEmbed1Encoder
from my_curator.adapters.storage.frame_loader import load_frames
from my_curator.adapters.storage.milvus import MilvusHybridRepository
from my_curator.adapters.storage.minio import MinIORepository
from my_curator.adapters.storage.pg import PGRepository
from my_curator.domain.scout.dna_text import dna_to_text

from ..deps import get_embedder, get_hybrid, get_minio, get_pg
from .search import ClipResult, SearchResponse

router = APIRouter()


class VideoSearchRequest(BaseModel):
    clip_id: str
    limit: int = Field(default=20, ge=1, le=1000)
    top_k: int = Field(default=1000, ge=1, le=10000)


@router.post("/v1/search/video", response_model=SearchResponse)
async def search_video(
    req: VideoSearchRequest,
    hybrid: MilvusHybridRepository = Depends(get_hybrid),
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

    text_vec = await asyncio.to_thread(embedder.encode_text, dna_to_text(row.get("dna_json") or {}))

    video_vec = None
    key_prefix = row.get("frames_blob_uri")
    if key_prefix:
        try:
            frames_tensor = await load_frames(minio, "frames", key_prefix)
            video_vec = await asyncio.to_thread(embedder.encode_video, frames_tensor)
        except Exception:
            video_vec = None  # fall back to text-only similarity

    if video_vec is not None:
        milvus_results = await hybrid.hybrid_search(
            text_vec=text_vec, video_vec=video_vec, top_k=req.top_k, require_video=False
        )
    else:
        milvus_results = await hybrid.search_text(text_vec, top_k=req.top_k)

    milvus_results = [r for r in milvus_results if r["clip_id"] != clip_uuid]
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
