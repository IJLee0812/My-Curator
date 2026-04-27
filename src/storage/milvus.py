"""Milvus DAL for My-Curator (pymilvus MilvusClient, sync-under-async).

Public surface:
  MilvusRepository.create()  — async factory; creates collection + GPU_CAGRA index if absent
  MilvusRepository.close()   — release client
  upsert / batch_upsert      — insert or update clip embeddings
  flush                      — force-seal growing segments (call before search in tests)
  search                     — ANN top-k (IP metric; caller must L2-normalise embeddings)
  delete / count             — maintenance helpers

Embedding contract: all vectors stored here MUST be L2-normalised (‖v‖ = 1).
IP(a, b) then equals cosine similarity.  Normalisation is the embedder's
responsibility (P3-1 Cosmos-Embed1 worker, spatial variant 336p).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from pymilvus import DataType, MilvusClient

COLLECTION_NAME = "clip_video_embed"
DIM = 1024  # Cosmos-Embed1 output dim (224p / 336p / 448p all produce 1024-d)

# GPU_CAGRA with cuVS defaults — adequate for <1M vectors.
# Tune intermediate_graph_degree / graph_degree / itopk_size at P3-2 (1M+ scale).
_INDEX_PARAMS: dict[str, Any] = {
    "index_type": "GPU_CAGRA",
    "metric_type": "IP",
    "params": {},
}
_SEARCH_PARAMS: dict[str, Any] = {"metric_type": "IP", "params": {}}


def _init_client(uri: str, collection_name: str, dim: int) -> MilvusClient:
    client = MilvusClient(uri=uri)
    if not client.has_collection(collection_name):
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("clip_id", DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", **_INDEX_PARAMS)
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
    return client


class MilvusRepository:
    def __init__(self, client: MilvusClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @classmethod
    async def create(
        cls,
        uri: str,
        *,
        collection_name: str = COLLECTION_NAME,
        dim: int = DIM,
    ) -> MilvusRepository:
        """Async factory — connects, creates collection + GPU_CAGRA index if absent."""
        client = await asyncio.to_thread(_init_client, uri, collection_name, dim)
        return cls(client, collection_name)

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)

    # ── writes ────────────────────────────────────────────────────────────────

    async def upsert(self, clip_id: UUID, embedding: list[float]) -> None:
        """Insert or replace a single clip embedding."""
        data = [{"clip_id": str(clip_id), "embedding": embedding}]
        await asyncio.to_thread(self._client.upsert, self._collection_name, data)

    async def batch_upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or replace multiple clip embeddings.

        Each record: {"clip_id": UUID, "embedding": list[float]}
        """
        data = [{"clip_id": str(r["clip_id"]), "embedding": r["embedding"]} for r in records]
        await asyncio.to_thread(self._client.upsert, self._collection_name, data)

    async def flush(self) -> None:
        """Force-seal growing segments so they become searchable immediately.

        Not required in production (Milvus auto-seals); useful in tests to
        avoid flaky search-after-insert races.
        """
        await asyncio.to_thread(self._client.flush, self._collection_name)

    # ── reads ─────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_vec: list[float],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest neighbours.

        Results are ordered by descending IP score (highest = most similar).
        Returns list of {"clip_id": UUID, "score": float}.
        """
        raw = await asyncio.to_thread(
            self._client.search,
            self._collection_name,
            [query_vec],
            anns_field="embedding",
            search_params=_SEARCH_PARAMS,
            limit=top_k,
            output_fields=["clip_id"],
        )
        return [
            {"clip_id": UUID(hit["entity"]["clip_id"]), "score": hit["distance"]} for hit in raw[0]
        ]

    async def delete(self, clip_id: UUID) -> None:
        """Remove a clip embedding by primary key."""
        await asyncio.to_thread(self._client.delete, self._collection_name, ids=[str(clip_id)])

    async def count(self) -> int:
        """Return the number of vectors currently in the collection."""
        stats = await asyncio.to_thread(self._client.get_collection_stats, self._collection_name)
        return int(stats["row_count"])
