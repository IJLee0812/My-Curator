"""Unit tests for the P4-7 MilvusHybridRepository (dual video + text vectors).

No Milvus server / GPU / torch — a fake client records the schema, upsert rows,
and search/hybrid_search calls so the DAL logic can run on a bare venv.  The
native hybrid_search ranking itself is exercised in the compose integration
tests; here we assert row shaping, the has_video placeholder, and query wiring.
"""

from __future__ import annotations

import uuid

import pytest
from pymilvus import RRFRanker, WeightedRanker

pytestmark = pytest.mark.unit

_DIM = 8


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[str] = []

    def add_field(self, name, *_a, **_k) -> None:
        self.fields.append(name)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexed: list[str] = []

    def add_index(self, field_name, **_k) -> None:
        self.indexed.append(field_name)


def _hit(clip_id: str, score: float) -> dict:
    return {"entity": {"clip_id": clip_id}, "distance": score}


class _FakeHybridClient:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeSchema] = {}
        self.rows: list[dict] = []
        self.search_calls: list[dict] = []
        self.hybrid_calls: list[dict] = []
        self._canned = [_hit(str(uuid.uuid4()), 0.9), _hit(str(uuid.uuid4()), 0.8)]

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def create_schema(self, **_k) -> _FakeSchema:
        return _FakeSchema()

    def prepare_index_params(self) -> _FakeIndexParams:
        return _FakeIndexParams()

    def create_collection(self, *, collection_name, schema, index_params) -> None:
        self._collections[collection_name] = schema
        self._last_schema = schema
        self._last_index = index_params

    def upsert(self, collection_name, data) -> None:
        self.rows.extend(data)

    def search(self, collection_name, data, **kwargs) -> list:
        self.search_calls.append(kwargs)
        return [self._canned]

    def hybrid_search(self, collection_name, reqs, ranker, **kwargs) -> list:
        self.hybrid_calls.append({"reqs": reqs, "ranker": ranker, "kwargs": kwargs})
        return [self._canned]


def _repo(client):
    from my_curator.adapters.storage.milvus import MilvusHybridRepository

    return MilvusHybridRepository(client, "clip_hybrid_embed", dim=_DIM)


@pytest.mark.asyncio
async def test_schema_has_both_vectors_and_flag() -> None:
    client = _FakeHybridClient()
    await _repo(client).upsert(uuid.uuid4(), text_vec=[0.1] * _DIM, video_vec=[0.2] * _DIM)
    schema = client._collections["clip_hybrid_embed"]
    assert set(schema.fields) == {"clip_id", "text_embedding", "video_embedding", "has_video"}
    assert client._last_index.indexed == ["text_embedding", "video_embedding"]


@pytest.mark.asyncio
async def test_upsert_with_video_sets_flag_true() -> None:
    client = _FakeHybridClient()
    vid = [0.5] * _DIM
    await _repo(client).upsert(uuid.uuid4(), text_vec=[0.1] * _DIM, video_vec=vid)
    row = client.rows[-1]
    assert row["has_video"] is True
    assert row["video_embedding"] == vid


@pytest.mark.asyncio
async def test_upsert_text_only_uses_zero_placeholder() -> None:
    client = _FakeHybridClient()
    await _repo(client).upsert(uuid.uuid4(), text_vec=[0.1] * _DIM, video_vec=None)
    row = client.rows[-1]
    assert row["has_video"] is False
    assert row["video_embedding"] == [0.0] * _DIM


@pytest.mark.asyncio
async def test_batch_upsert_mixed() -> None:
    client = _FakeHybridClient()
    recs = [
        {"clip_id": uuid.uuid4(), "text_vec": [0.1] * _DIM, "video_vec": [0.2] * _DIM},
        {"clip_id": uuid.uuid4(), "text_vec": [0.3] * _DIM},  # no video
    ]
    await _repo(client).batch_upsert(recs)
    assert [r["has_video"] for r in client.rows] == [True, False]


@pytest.mark.asyncio
async def test_search_video_filters_has_video() -> None:
    client = _FakeHybridClient()
    repo = _repo(client)
    await repo.search_video([0.1] * _DIM, top_k=5, require_video=True)
    assert client.search_calls[-1]["filter"] == "has_video == true"
    assert client.search_calls[-1]["anns_field"] == "video_embedding"
    await repo.search_video([0.1] * _DIM, top_k=5, require_video=False)
    assert client.search_calls[-1]["filter"] == ""


@pytest.mark.asyncio
async def test_search_text_no_filter() -> None:
    client = _FakeHybridClient()
    res = await _repo(client).search_text([0.1] * _DIM, top_k=5)
    assert client.search_calls[-1]["anns_field"] == "text_embedding"
    assert len(res) == 2 and all("clip_id" in r and "score" in r for r in res)


@pytest.mark.asyncio
async def test_hybrid_search_weighted_default_and_rrf() -> None:
    client = _FakeHybridClient()
    repo = _repo(client)
    await repo.hybrid_search(text_vec=[0.1] * _DIM, video_vec=[0.2] * _DIM, top_k=5)
    call = client.hybrid_calls[-1]
    assert len(call["reqs"]) == 2
    assert isinstance(call["ranker"], WeightedRanker)
    await repo.hybrid_search(text_vec=[0.1] * _DIM, video_vec=[0.2] * _DIM, top_k=5, rrf_k=60)
    assert isinstance(client.hybrid_calls[-1]["ranker"], RRFRanker)
