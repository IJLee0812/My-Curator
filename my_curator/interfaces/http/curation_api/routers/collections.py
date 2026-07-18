"""GET /v1/collections — Milvus collection status (P3-2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from my_curator.adapters.storage.milvus import DIM, HYBRID_COLLECTION_NAME, MilvusHybridRepository

from ..deps import get_hybrid

router = APIRouter()


class CollectionInfo(BaseModel):
    collection_name: str
    vector_count: int
    dim: int
    index_type: str
    metric_type: str


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


@router.get("/v1/collections", response_model=CollectionsResponse)
async def list_collections(
    hybrid: MilvusHybridRepository = Depends(get_hybrid),
) -> CollectionsResponse:
    count = await hybrid.count()
    info = CollectionInfo(
        collection_name=HYBRID_COLLECTION_NAME,
        vector_count=count,
        dim=DIM,
        index_type="GPU_CAGRA",
        metric_type="IP",
    )
    return CollectionsResponse(collections=[info])
