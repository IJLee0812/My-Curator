"""POST /v1/search — hybrid Milvus ANN + Postgres JSONB filter search (P3-2 / P3-4 / P4-7).

Query order: Milvus top-k ANN first (over the P4-7 dual-vector hybrid
collection) → clip_id = ANY($1) + GIN on dna_json in PG.  Never issues a full
PG scan without the Milvus candidate set.

P4-7: retrieval runs against ``clip_video`` + ``narrative-text`` vectors in one
collection.  The served modality is fixed server-side by
``CURATION_SEARCH_MODALITY`` (``text`` | ``video`` | ``hybrid``) — set to the
3-way benchmark winner; there is no per-request modality/weight knob.  The query
text is embedded once; for ``video`` / ``hybrid`` the same vector is used against
the video field via the shared Cosmos-Embed1 joint space (cross-modal).
Adjacent-window de-duplication is applied to the ranked results; the optional
``dedup_by_source`` flag collapses each source clip to its top hit.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from my_curator.adapters.embed.text_tower import CosmosEmbed1Encoder
from my_curator.adapters.storage.milvus import MilvusHybridRepository
from my_curator.adapters.storage.pg import PGRepository
from my_curator.domain.search_dedup import dedup_adjacent_windows, dedup_by_source

from ..deps import get_embedder, get_hybrid, get_pg

router = APIRouter()

# Served retrieval modality (P4-7).  Default is real-time hybrid: every query
# fuses the text + video towers via Milvus hybrid_search.  Overridable with the
# CURATION_SEARCH_MODALITY env var (text | video | hybrid).
_VALID_MODALITIES = {"text", "video", "hybrid"}
SEARCH_MODALITY = (os.environ.get("CURATION_SEARCH_MODALITY") or "hybrid").lower()
if SEARCH_MODALITY not in _VALID_MODALITIES:
    SEARCH_MODALITY = "hybrid"


class SearchFilters(BaseModel):
    weather: str | list[str] | None = None
    lighting: str | list[str] | None = None
    sensor_fidelity: str | list[str] | None = None
    road_type: str | list[str] | None = None
    lane_event: str | list[str] | None = None
    intersection_type: str | list[str] | None = None
    actor_class: str | list[str] | None = None
    actor_state: str | list[str] | None = None
    risk_level: str | list[str] | None = None
    ego_maneuver: str | list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=20, ge=1, le=1000)
    top_k: int = Field(default=1000, ge=1, le=10000)
    dedup_by_source: bool = False


class ClipResult(BaseModel):
    clip_id: str
    score: float
    dna_json: dict[str, Any] | None = None
    start_s: float | None = None
    end_s: float | None = None
    blob_uri: str | None = None
    is_gold: bool | None = None
    source_clip_id: str | None = None


class SearchResponse(BaseModel):
    results: list[ClipResult]
    total: int


async def _retrieve(
    hybrid: MilvusHybridRepository, query_vec: list[float], top_k: int
) -> list[dict[str, Any]]:
    """Milvus candidate retrieval for the served modality."""
    if SEARCH_MODALITY == "video":
        return await hybrid.search_video(query_vec, top_k=top_k)
    if SEARCH_MODALITY == "hybrid":
        # require_video=False so frameless (text-only) clips remain reachable via
        # the text tower — a text query should not drop 21% of the corpus.
        return await hybrid.hybrid_search(
            text_vec=query_vec, video_vec=query_vec, top_k=top_k, require_video=False
        )
    return await hybrid.search_text(query_vec, top_k=top_k)


@router.post("/v1/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    hybrid: MilvusHybridRepository = Depends(get_hybrid),
    pg: PGRepository = Depends(get_pg),
    embedder: CosmosEmbed1Encoder = Depends(get_embedder),
) -> SearchResponse:
    query_vec = await asyncio.to_thread(embedder.encode_text, req.query)
    milvus_results = await _retrieve(hybrid, query_vec, req.top_k)

    if not milvus_results:
        return SearchResponse(results=[], total=0)

    clip_ids: list[UUID] = [r["clip_id"] for r in milvus_results]
    score_map: dict[UUID, float] = {r["clip_id"]: r["score"] for r in milvus_results}

    filters_dict = {k: v for k, v in req.filters.model_dump().items() if v is not None}
    pg_results = await pg.filter_dna_by_ids(clip_ids, filters_dict, limit=len(clip_ids))

    ranked = sorted(pg_results, key=lambda r: score_map.get(r["clip_id"], 0.0), reverse=True)
    for r in ranked:
        r["score"] = score_map.get(r["clip_id"], 0.0)

    # Always collapse near-duplicate overlapping windows of the same source clip
    # (P4-5 1 s overlap); optionally collapse each source to its single top hit.
    ranked = dedup_adjacent_windows(ranked)
    if req.dedup_by_source:
        ranked = dedup_by_source(ranked)
    ranked = ranked[: req.limit]

    results = [
        ClipResult(
            clip_id=str(r["clip_id"]),
            score=r.get("score", 0.0),
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
