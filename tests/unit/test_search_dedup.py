"""Unit tests for P4-7 search de-duplication + served-modality routing.

Pure logic + a fake hybrid repo — no Milvus / torch / server.
"""

from __future__ import annotations

import uuid

import pytest

from my_curator.domain.search_dedup import dedup_adjacent_windows, dedup_by_source

pytestmark = pytest.mark.unit


def _r(source, score, clip_id=None):
    return {"clip_id": clip_id or uuid.uuid4(), "source_clip_id": source, "score": score}


# ── dedup_adjacent_windows ──────────────────────────────────────────────────────


def test_collapses_near_equal_same_source():
    results = [_r("A", 0.90), _r("A", 0.9000001), _r("A", 0.60)]
    out = dedup_adjacent_windows(results)
    # the near-identical 0.90 window is collapsed; the distinct 0.60 window stays
    assert [round(r["score"], 3) for r in out] == [0.90, 0.60]


def test_keeps_distinct_scores_same_source():
    results = [_r("A", 0.90), _r("A", 0.70)]
    assert len(dedup_adjacent_windows(results)) == 2


def test_different_sources_not_collapsed_even_if_equal():
    results = [_r("A", 0.90), _r("B", 0.90)]
    assert len(dedup_adjacent_windows(results)) == 2


def test_items_without_source_or_score_kept():
    results = [_r(None, 0.9), {"clip_id": uuid.uuid4(), "source_clip_id": "A"}]  # no score
    assert len(dedup_adjacent_windows(results)) == 2


def test_preserves_rank_order():
    results = [_r("A", 0.9), _r("B", 0.8), _r("A", 0.5)]
    out = dedup_adjacent_windows(results)
    assert [r["score"] for r in out] == [0.9, 0.8, 0.5]


# ── dedup_by_source ─────────────────────────────────────────────────────────────


def test_by_source_keeps_top_per_source():
    a1, a2, b1 = _r("A", 0.9), _r("A", 0.7), _r("B", 0.8)
    out = dedup_by_source([a1, a2, b1])
    assert out == [a1, b1]  # A collapses to its top hit; order preserved


def test_by_source_keeps_none_source():
    r = _r(None, 0.5)
    assert dedup_by_source([r, r]) == [r, r]


# ── served-modality routing ─────────────────────────────────────────────────────


class _FakeHybrid:
    def __init__(self):
        self.calls = []

    async def search_text(self, qv, *, top_k):
        self.calls.append("text")
        return [{"clip_id": uuid.uuid4(), "score": 1.0}]

    async def search_video(self, qv, *, top_k):
        self.calls.append("video")
        return [{"clip_id": uuid.uuid4(), "score": 1.0}]

    async def hybrid_search(self, *, text_vec, video_vec, top_k, require_video=True):
        self.calls.append("hybrid")
        return [{"clip_id": uuid.uuid4(), "score": 1.0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["text", "video", "hybrid"])
async def test_retrieve_dispatches_by_modality(monkeypatch, modality):
    from my_curator.interfaces.http.curation_api.routers import search as search_mod

    monkeypatch.setattr(search_mod, "SEARCH_MODALITY", modality)
    fake = _FakeHybrid()
    await search_mod._retrieve(fake, [0.1] * 8, top_k=5)
    assert fake.calls == [modality]
