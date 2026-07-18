"""Unit tests for the P4-7 ensure-collection guard on Milvus writes.

Regression guard for the operational defect where a full DB reset dropped the
Milvus collection while a long-running writer (embedder worker) stayed up: the
writer's client kept upserting into a now-missing collection and every write
failed with ``can't find collection`` forever.  ``MilvusRepository`` writes now
recreate the collection transparently if it is absent.

No Milvus server, GPU, or torch required — a fake client models the schema /
create / upsert surface and a "dropped collection" that raises on upsert.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.unit


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[str] = []

    def add_field(self, name, *_args, **_kwargs) -> None:
        self.fields.append(name)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexed: list[str] = []

    def add_index(self, field_name, **_kwargs) -> None:
        self.indexed.append(field_name)


class _FakeMilvusClient:
    """Minimal MilvusClient stand-in; upsert into a missing collection raises."""

    def __init__(self) -> None:
        self._collections: set[str] = set()
        self.upserts: list[tuple[str, list[dict]]] = []
        self.create_calls = 0

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def create_schema(self, **_kwargs) -> _FakeSchema:
        return _FakeSchema()

    def prepare_index_params(self) -> _FakeIndexParams:
        return _FakeIndexParams()

    def create_collection(self, *, collection_name: str, schema, index_params) -> None:
        self._collections.add(collection_name)
        self.create_calls += 1

    def upsert(self, collection_name: str, data: list[dict]) -> None:
        if collection_name not in self._collections:
            raise RuntimeError(f"can't find collection[collection={collection_name}]")
        self.upserts.append((collection_name, data))

    def drop(self, name: str) -> None:
        self._collections.discard(name)


def _repo(client: _FakeMilvusClient):
    from my_curator.adapters.storage.milvus import MilvusRepository

    return MilvusRepository(client, "clip_video_embed", dim=768)


@pytest.mark.asyncio
async def test_upsert_recreates_missing_collection() -> None:
    """A single upsert against an absent collection creates it, then writes."""
    client = _FakeMilvusClient()  # starts with NO collection (simulates post-reset)
    repo = _repo(client)

    await repo.upsert(uuid.uuid4(), [0.1] * 768)

    assert client.create_calls == 1
    assert len(client.upserts) == 1


@pytest.mark.asyncio
async def test_upsert_does_not_recreate_when_present() -> None:
    """No redundant create when the collection already exists."""
    client = _FakeMilvusClient()
    client._collections.add("clip_video_embed")
    repo = _repo(client)

    await repo.upsert(uuid.uuid4(), [0.1] * 768)
    await repo.batch_upsert([{"clip_id": uuid.uuid4(), "embedding": [0.2] * 768}])

    assert client.create_calls == 0
    assert len(client.upserts) == 2


@pytest.mark.asyncio
async def test_write_survives_collection_dropped_mid_run() -> None:
    """The exact defect: collection dropped under a live writer → next write recovers."""
    client = _FakeMilvusClient()
    repo = _repo(client)

    await repo.upsert(uuid.uuid4(), [0.1] * 768)  # creates + writes
    client.drop("clip_video_embed")  # external full reset

    await repo.batch_upsert([{"clip_id": uuid.uuid4(), "embedding": [0.3] * 768}])

    assert client.create_calls == 2  # recreated after the drop
    assert len(client.upserts) == 2
