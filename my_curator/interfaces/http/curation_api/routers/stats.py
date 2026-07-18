"""GET /v1/stats — aggregate curation counts for the Dashboard (P3-4).

Surfaces the totals the React UI Dashboard renders without a mock data
fallback: total clip count, vector count from Milvus, review-queue
breakdown, and the derived `dna_pass_rate`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from my_curator.adapters.storage.milvus import MilvusHybridRepository
from my_curator.adapters.storage.pg import PGRepository

from ..deps import get_hybrid, get_pg

router = APIRouter()


class ReviewCounts(BaseModel):
    pending: int
    approved: int
    rejected: int
    rejected_schema_invalid: int


class StatsResponse(BaseModel):
    total_clips: int
    scenario_dna_count: int
    vector_count: int
    review: ReviewCounts
    dna_pass_rate: float | None


@router.get("/v1/stats", response_model=StatsResponse)
async def get_stats(
    pg: PGRepository = Depends(get_pg),
    hybrid: MilvusHybridRepository = Depends(get_hybrid),
) -> StatsResponse:
    pg_stats = await pg.get_stats()
    vector_count = await hybrid.count()
    return StatsResponse(
        total_clips=pg_stats["total_clips"],
        scenario_dna_count=pg_stats["scenario_dna_count"],
        vector_count=int(vector_count or 0),
        review=ReviewCounts(**pg_stats["review"]),
        dna_pass_rate=pg_stats["dna_pass_rate"],
    )
