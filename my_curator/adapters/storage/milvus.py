"""Milvus DAL for My-Curator (pymilvus MilvusClient, sync-under-async).

Public surface:
  MilvusRepository.create()  — async factory; creates collection + GPU_CAGRA index if absent
  MilvusRepository.close()   — release client
  upsert / batch_upsert      — insert or update clip embeddings
  flush                      — force-seal growing segments (call before search in tests)
  search                     — ANN top-k (IP metric; caller must L2-normalise embeddings)
  delete / count             — maintenance helpers

Embedding contract: all vectors stored here MUST be L2-normalised (‖v‖ = 1).
IP(a, b) then equals cosine similarity.  Cosmos-Embed1-336p outputs already
L2-normalised vectors; no additional normalisation step is required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker, WeightedRanker

COLLECTION_NAME = "clip_video_embed"
HYBRID_COLLECTION_NAME = "clip_hybrid_embed"  # P4-7: dual video + narrative-text vectors
DIM = 768  # Cosmos-Embed1-336p output dim

# GPU_CAGRA with cuVS defaults — adequate for <1M vectors.
# Tune intermediate_graph_degree / graph_degree / itopk_size at P3-2 (1M+ scale).
_INDEX_PARAMS: dict[str, Any] = {
    "index_type": "GPU_CAGRA",
    "metric_type": "IP",
    "params": {},
}
_SEARCH_PARAMS: dict[str, Any] = {"metric_type": "IP", "params": {}}


def _ensure_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    """Create the collection + GPU_CAGRA index if it is absent.

    Idempotent and cheap when the collection already exists (one has_collection
    RPC).  Callers invoke this before every write so that an external drop
    (e.g. a full DB reset run while this client is live) cannot silently break
    the writer: the next upsert transparently recreates the collection instead
    of failing with ``can't find collection`` forever.
    """
    if client.has_collection(collection_name):
        return
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


def _init_client(uri: str, collection_name: str, dim: int) -> MilvusClient:
    client = MilvusClient(uri=uri)
    _ensure_collection(client, collection_name, dim)
    return client


class MilvusRepository:
    def __init__(self, client: MilvusClient, collection_name: str, *, dim: int = DIM) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dim = dim

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
        return cls(client, collection_name, dim=dim)

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)

    async def _ensure(self) -> None:
        """Recreate the collection if it was dropped out from under this client."""
        await asyncio.to_thread(_ensure_collection, self._client, self._collection_name, self._dim)

    # ── writes ────────────────────────────────────────────────────────────────

    async def upsert(self, clip_id: UUID, embedding: list[float]) -> None:
        """Insert or replace a single clip embedding."""
        await self._ensure()
        data = [{"clip_id": str(clip_id), "embedding": embedding}]
        await asyncio.to_thread(self._client.upsert, self._collection_name, data)

    async def batch_upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or replace multiple clip embeddings.

        Each record: {"clip_id": UUID, "embedding": list[float]}
        """
        await self._ensure()
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
            consistency_level="Strong",
        )
        return [
            {"clip_id": UUID(hit["entity"]["clip_id"]), "score": hit["distance"]} for hit in raw[0]
        ]

    async def delete(self, clip_id: UUID) -> None:
        """Remove a clip embedding by primary key."""
        await asyncio.to_thread(self._client.delete, self._collection_name, ids=[str(clip_id)])

    async def count(self) -> int:
        """Return the number of live vectors in the collection.

        Uses query(count(*)) rather than get_collection_stats() because
        row_count includes soft-deleted rows until compaction completes.
        """
        result = await asyncio.to_thread(
            self._client.query,
            self._collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"])


# ── P4-7: dual-vector (video + narrative-text) hybrid collection ────────────────

_HAS_VIDEO_EXPR = "has_video == true"


def _ensure_hybrid_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    """Create the dual-vector hybrid collection + indexes if absent (idempotent).

    Schema: clip_id (PK) + text_embedding + video_embedding (both 768-d IP /
    GPU_CAGRA) + has_video (BOOL).  Milvus cannot add a vector field to an
    existing collection (add_collection_field is nullable-scalar only) and
    FLOAT_VECTOR is non-nullable, so frameless clips are stored with a
    zero-vector video placeholder and has_video=false, then excluded from
    video / hybrid ANN via ``has_video == true``.
    """
    if client.has_collection(collection_name):
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("clip_id", DataType.VARCHAR, max_length=36, is_primary=True)
    schema.add_field("text_embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("video_embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("has_video", DataType.BOOL)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="text_embedding", **_INDEX_PARAMS)
    index_params.add_index(field_name="video_embedding", **_INDEX_PARAMS)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def _init_hybrid_client(uri: str, collection_name: str, dim: int) -> MilvusClient:
    client = MilvusClient(uri=uri)
    _ensure_hybrid_collection(client, collection_name, dim)
    return client


class MilvusHybridRepository:
    """DAL for the P4-7 hybrid collection — one row per clip carrying both a
    video and a narrative-text 768-d vector, plus a ``has_video`` flag.

    All stored vectors MUST be L2-normalised (IP == cosine), except the
    zero-vector video placeholder written for frameless clips (never surfaced:
    it is filtered out of video / hybrid search by ``has_video == true``).
    """

    def __init__(self, client: MilvusClient, collection_name: str, *, dim: int = DIM) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dim = dim

    @classmethod
    async def create(
        cls,
        uri: str,
        *,
        collection_name: str = HYBRID_COLLECTION_NAME,
        dim: int = DIM,
    ) -> MilvusHybridRepository:
        client = await asyncio.to_thread(_init_hybrid_client, uri, collection_name, dim)
        return cls(client, collection_name, dim=dim)

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)

    async def _ensure(self) -> None:
        await asyncio.to_thread(
            _ensure_hybrid_collection, self._client, self._collection_name, self._dim
        )

    def _row(self, clip_id: UUID, text_vec: list[float], video_vec: list[float] | None) -> dict:
        has_video = video_vec is not None
        return {
            "clip_id": str(clip_id),
            "text_embedding": text_vec,
            "video_embedding": video_vec if has_video else [0.0] * self._dim,
            "has_video": has_video,
        }

    # ── writes ────────────────────────────────────────────────────────────────

    async def upsert(
        self, clip_id: UUID, *, text_vec: list[float], video_vec: list[float] | None = None
    ) -> None:
        """Insert or replace one clip's dual embedding (video_vec=None → text-only)."""
        await self._ensure()
        await asyncio.to_thread(
            self._client.upsert, self._collection_name, [self._row(clip_id, text_vec, video_vec)]
        )

    async def batch_upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or replace many rows.

        Each record: {"clip_id": UUID, "text_vec": list[float],
        "video_vec": list[float] | None}.
        """
        await self._ensure()
        data = [self._row(r["clip_id"], r["text_vec"], r.get("video_vec")) for r in records]
        await asyncio.to_thread(self._client.upsert, self._collection_name, data)

    async def flush(self) -> None:
        await asyncio.to_thread(self._client.flush, self._collection_name)

    # ── reads ─────────────────────────────────────────────────────────────────

    def _hits(self, raw: Any) -> list[dict[str, Any]]:
        return [
            {"clip_id": UUID(hit["entity"]["clip_id"]), "score": hit["distance"]} for hit in raw[0]
        ]

    async def search_text(self, query_vec: list[float], *, top_k: int = 10) -> list[dict[str, Any]]:
        """ANN over the narrative-text vector (all clips)."""
        raw = await asyncio.to_thread(
            self._client.search,
            self._collection_name,
            [query_vec],
            anns_field="text_embedding",
            search_params=_SEARCH_PARAMS,
            limit=top_k,
            output_fields=["clip_id"],
            consistency_level="Strong",
        )
        return self._hits(raw)

    async def search_video(
        self, query_vec: list[float], *, top_k: int = 10, require_video: bool = True
    ) -> list[dict[str, Any]]:
        """ANN over the video vector (frameless clips excluded when require_video)."""
        raw = await asyncio.to_thread(
            self._client.search,
            self._collection_name,
            [query_vec],
            anns_field="video_embedding",
            search_params=_SEARCH_PARAMS,
            limit=top_k,
            output_fields=["clip_id"],
            filter=_HAS_VIDEO_EXPR if require_video else "",
            consistency_level="Strong",
        )
        return self._hits(raw)

    async def hybrid_search(
        self,
        *,
        text_vec: list[float],
        video_vec: list[float],
        top_k: int = 10,
        weights: tuple[float, float] | None = (0.5, 0.5),
        rrf_k: int | None = None,
        require_video: bool = True,
    ) -> list[dict[str, Any]]:
        """Fuse text + video ANN with WeightedRanker (default) or RRFRanker.

        weights = (text_weight, video_weight).  Pass rrf_k to use RRFRanker
        instead.  Frameless clips are excluded from both sub-requests when
        require_video (a text-only fusion would otherwise favour their
        zero-score video half inconsistently).
        """
        expr = _HAS_VIDEO_EXPR if require_video else ""
        text_req = AnnSearchRequest(
            data=[text_vec],
            anns_field="text_embedding",
            param=_SEARCH_PARAMS,
            limit=top_k,
            expr=expr,
        )
        video_req = AnnSearchRequest(
            data=[video_vec],
            anns_field="video_embedding",
            param=_SEARCH_PARAMS,
            limit=top_k,
            expr=expr,
        )
        ranker = RRFRanker(rrf_k) if rrf_k is not None else WeightedRanker(*weights)
        raw = await asyncio.to_thread(
            self._client.hybrid_search,
            self._collection_name,
            [text_req, video_req],
            ranker,
            limit=top_k,
            output_fields=["clip_id"],
        )
        return self._hits(raw)

    async def count(self) -> int:
        result = await asyncio.to_thread(
            self._client.query,
            self._collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"])
