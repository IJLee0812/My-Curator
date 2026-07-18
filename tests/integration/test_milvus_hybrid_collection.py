"""Integration tests for the P4-7 MilvusHybridRepository (dual video + text).

Requires the Milvus service (infra/compose.base.yml) on :19530; auto-skipped
otherwise.  Each test uses its own ephemeral collection, dropped on teardown.

Validates the risky parts that unit tests with a fake client cannot: the
two-vector schema builds on GPU_CAGRA, ``has_video == true`` filtering works on
the video field, and native ``hybrid_search`` fuses both towers.

Embedding note: vectors are L2-normalised so IP == cosine similarity.
"""

from __future__ import annotations

import math
import socket
import uuid
from typing import Generator

import pytest
import pytest_asyncio

from my_curator.adapters.storage.milvus import MilvusHybridRepository

MILVUS_URI = "http://localhost:19530"
TEST_DIM = 128  # GPU_CAGRA requires >= 32


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


requires_milvus = pytest.mark.skipif(
    not _tcp_open("127.0.0.1", 19530),
    reason="Milvus not up (run: docker compose -f infra/compose.base.yml --env-file .env up -d)",
)

pytestmark = [pytest.mark.integration, requires_milvus]


def _vec(dim: int, seed: int = 0) -> list[float]:
    rng = [math.sin(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in rng))
    return [x / norm for x in rng]


@pytest_asyncio.fixture
async def repo() -> Generator[MilvusHybridRepository, None, None]:
    coll = f"test_hybrid_{uuid.uuid4().hex[:8]}"
    r = await MilvusHybridRepository.create(MILVUS_URI, collection_name=coll, dim=TEST_DIM)
    yield r
    r._client.drop_collection(coll)
    await r.close()


async def test_collection_has_both_vectors(repo: MilvusHybridRepository):
    assert repo._client.has_collection(repo._collection_name)
    desc = repo._client.describe_collection(repo._collection_name)
    names = {f["name"] for f in desc["fields"]}
    assert {"clip_id", "text_embedding", "video_embedding", "has_video"} <= names


async def test_dual_upsert_search_each_field(repo: MilvusHybridRepository):
    clip_id = uuid.uuid4()
    tvec, vvec = _vec(TEST_DIM, 1), _vec(TEST_DIM, 2)
    await repo.upsert(clip_id, text_vec=tvec, video_vec=vvec)
    await repo.flush()

    t = await repo.search_text(tvec, top_k=1)
    v = await repo.search_video(vvec, top_k=1)
    assert t[0]["clip_id"] == clip_id and t[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert v[0]["clip_id"] == clip_id and v[0]["score"] == pytest.approx(1.0, abs=1e-4)


async def test_video_search_excludes_frameless(repo: MilvusHybridRepository):
    with_video = uuid.uuid4()
    text_only = uuid.uuid4()
    await repo.batch_upsert(
        [
            {"clip_id": with_video, "text_vec": _vec(TEST_DIM, 3), "video_vec": _vec(TEST_DIM, 4)},
            {"clip_id": text_only, "text_vec": _vec(TEST_DIM, 5)},  # no video
        ]
    )
    await repo.flush()

    vids = {r["clip_id"] for r in await repo.search_video(_vec(TEST_DIM, 4), top_k=10)}
    assert with_video in vids
    assert text_only not in vids  # has_video == true filter excludes the placeholder

    # But the frameless clip IS reachable by text search.
    txts = {r["clip_id"] for r in await repo.search_text(_vec(TEST_DIM, 5), top_k=10)}
    assert text_only in txts


async def test_hybrid_search_fuses_and_excludes_frameless(repo: MilvusHybridRepository):
    target = uuid.uuid4()
    text_only = uuid.uuid4()
    await repo.batch_upsert(
        [
            {"clip_id": target, "text_vec": _vec(TEST_DIM, 6), "video_vec": _vec(TEST_DIM, 7)},
            {
                "clip_id": uuid.uuid4(),
                "text_vec": _vec(TEST_DIM, 8),
                "video_vec": _vec(TEST_DIM, 9),
            },
            {
                "clip_id": text_only,
                "text_vec": _vec(TEST_DIM, 6),
            },  # matches target's text, no video
        ]
    )
    await repo.flush()

    res = await repo.hybrid_search(
        text_vec=_vec(TEST_DIM, 6), video_vec=_vec(TEST_DIM, 7), top_k=10
    )
    ids = {r["clip_id"] for r in res}
    assert target in ids
    assert text_only not in ids  # require_video excludes frameless from hybrid


async def test_count_and_idempotent_upsert(repo: MilvusHybridRepository):
    clip_id = uuid.uuid4()
    await repo.upsert(clip_id, text_vec=_vec(TEST_DIM, 1), video_vec=_vec(TEST_DIM, 1))
    await repo.upsert(clip_id, text_vec=_vec(TEST_DIM, 2), video_vec=_vec(TEST_DIM, 2))
    await repo.flush()
    assert await repo.count() == 1
