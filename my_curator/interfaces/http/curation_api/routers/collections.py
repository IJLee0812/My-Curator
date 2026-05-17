"""GET /v1/collections — Milvus collection status (P3-2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from my_curator.adapters.storage.milvus import COLLECTION_NAME, DIM, MilvusRepository

from ..deps import get_milvus

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
    milvus: MilvusRepository = Depends(get_milvus),
) -> CollectionsResponse:
    count = await milvus.count()
    info = CollectionInfo(
        collection_name=COLLECTION_NAME,
        vector_count=count,
        dim=DIM,
        index_type="GPU_CAGRA",
        metric_type="IP",
    )
    return CollectionsResponse(collections=[info])
