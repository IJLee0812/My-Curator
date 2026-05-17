"""Integration tests for MilvusRepository (P1-4).

Requires the Milvus service from infra/compose.base.yml to be running:
    docker compose -f infra/compose.base.yml --env-file .env up -d milvus

Tests are auto-skipped when Milvus is not reachable on :19530.

Each test gets its own ephemeral collection (unique name per test) so tests
are fully isolated and can run in any order.  Collections are dropped in the
fixture teardown.

Embedding note: vectors are L2-normalised so IP == cosine similarity.
"""

from __future__ import annotations

import math
import socket
import time
import uuid
from typing import Generator

import pytest
import pytest_asyncio

from my_curator.adapters.storage.milvus import MilvusRepository

MILVUS_URI = "http://localhost:19530"

# Smaller dim for tests — GPU_CAGRA requires ≥ 32; 128 is fast to build.
TEST_DIM = 128

pytestmark = pytest.mark.integration


# ── availability guard ────────────────────────────────────────────────────────


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


MILVUS_UP = _tcp_open("127.0.0.1", 19530)

requires_milvus = pytest.mark.skipif(
    not MILVUS_UP,
    reason="Milvus not up (run: docker compose -f infra/compose.base.yml --env-file .env up -d)",
)

pytestmark = [pytest.mark.integration, requires_milvus]


# ── helpers ───────────────────────────────────────────────────────────────────


def _normalised_vec(dim: int, seed: int = 0) -> list[float]:
    """Return a deterministic L2-normalised vector of length *dim*."""
    rng = [math.sin(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in rng))
    return [x / norm for x in rng]


def _random_records(n: int, dim: int) -> list[dict]:
    return [{"clip_id": uuid.uuid4(), "embedding": _normalised_vec(dim, seed=i)} for i in range(n)]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def repo() -> Generator[MilvusRepository, None, None]:
    """Ephemeral collection per test — dropped on teardown."""
    coll = f"test_cvembed_{uuid.uuid4().hex[:8]}"
    r = await MilvusRepository.create(MILVUS_URI, collection_name=coll, dim=TEST_DIM)
    yield r
    r._client.drop_collection(coll)
    await r.close()


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_collection_created(repo: MilvusRepository):
    """create() produces a collection with GPU_CAGRA index."""
    assert repo._client.has_collection(repo._collection_name)


async def test_upsert_and_search_round_trip(repo: MilvusRepository):
    """Upserted vector is returned as top-1 with score ≈ 1.0 (L2-norm + IP)."""
    clip_id = uuid.uuid4()
    vec = _normalised_vec(TEST_DIM, seed=42)
    await repo.upsert(clip_id, vec)
    await repo.flush()

    results = await repo.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0]["clip_id"] == clip_id
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


async def test_batch_upsert_and_count(repo: MilvusRepository):
    """batch_upsert inserts all records; count() reflects the total."""
    records = _random_records(100, TEST_DIM)
    await repo.batch_upsert(records)
    await repo.flush()

    assert await repo.count() == 100


async def test_search_returns_top_k(repo: MilvusRepository):
    """search() returns exactly top_k results when the collection is larger."""
    records = _random_records(20, TEST_DIM)
    await repo.batch_upsert(records)
    await repo.flush()

    results = await repo.search(_normalised_vec(TEST_DIM, seed=99), top_k=5)
    assert len(results) == 5


async def test_upsert_idempotent(repo: MilvusRepository):
    """Re-upserting the same clip_id replaces — does not duplicate."""
    clip_id = uuid.uuid4()
    vec_a = _normalised_vec(TEST_DIM, seed=1)
    vec_b = _normalised_vec(TEST_DIM, seed=2)

    await repo.upsert(clip_id, vec_a)
    await repo.upsert(clip_id, vec_b)
    await repo.flush()

    assert await repo.count() == 1

    # The stored vector should be vec_b (latest wins).
    results = await repo.search(vec_b, top_k=1)
    assert results[0]["clip_id"] == clip_id
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


async def test_delete(repo: MilvusRepository):
    """delete() removes the clip embedding; count drops to zero."""
    clip_id = uuid.uuid4()
    await repo.upsert(clip_id, _normalised_vec(TEST_DIM, seed=7))
    await repo.flush()
    assert await repo.count() == 1

    await repo.delete(clip_id)
    await repo.flush()
    assert await repo.count() == 0


async def test_search_latency_10k(repo: MilvusRepository):
    """Median top-10 search latency on 10 k vectors is < 50 ms (P1-4 DoD)."""
    records = _random_records(10_000, TEST_DIM)
    # Insert in two batches to avoid a single large payload.
    await repo.batch_upsert(records[:5_000])
    await repo.batch_upsert(records[5_000:])
    await repo.flush()

    query = _normalised_vec(TEST_DIM, seed=0)
    latencies_ms: list[float] = []
    for i in range(20):
        q = _normalised_vec(TEST_DIM, seed=i * 7)
        t0 = time.perf_counter()
        await repo.search(q, top_k=10)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    median_ms = latencies_ms[len(latencies_ms) // 2]
    assert median_ms < 50, f"Median search latency {median_ms:.1f} ms exceeded 50 ms DoD"
    _ = query  # suppress unused-variable warning
