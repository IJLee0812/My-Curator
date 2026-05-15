"""POST /v1/search — hybrid Milvus ANN + Postgres JSONB filter search (P3-2 / P3-4).

Query order: Milvus top-k ANN first → clip_id = ANY($1) + GIN on dna_json in PG.
Never issues a full PG scan without the Milvus candidate set.

P3-4 extension: ``ClipResult`` carries the clip metadata the UI needs to
render result cards (start_s, end_s, blob_uri, is_gold, source_clip_id) so
the page does not have to issue an extra GET per result.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.storage.milvus import MilvusRepository
from src.storage.pg import PGRepository

from .deps import get_embedder, get_milvus, get_pg
from .embedder import CosmosEmbed1Encoder

router = APIRouter()


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


@router.post("/v1/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    milvus: MilvusRepository = Depends(get_milvus),
    pg: PGRepository = Depends(get_pg),
    embedder: CosmosEmbed1Encoder = Depends(get_embedder),
) -> SearchResponse:
    query_vec = await asyncio.to_thread(embedder.encode_text, req.query)
    milvus_results = await milvus.search(query_vec, top_k=req.top_k)

    if not milvus_results:
        return SearchResponse(results=[], total=0)

    clip_ids: list[UUID] = [r["clip_id"] for r in milvus_results]
    score_map: dict[UUID, float] = {r["clip_id"]: r["score"] for r in milvus_results}

    filters_dict = {k: v for k, v in req.filters.model_dump().items() if v is not None}
    pg_results = await pg.filter_dna_by_ids(clip_ids, filters_dict, limit=len(clip_ids))

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
