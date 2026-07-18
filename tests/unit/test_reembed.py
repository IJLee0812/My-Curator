"""Unit tests for the P4-7 corpus re-embed use-case (my_curator.application.reembed).

Fakes for PG / MinIO / towers / hybrid repo; ``load_frames`` is patched.  No
torch / GPU / Milvus / Postgres required.  Covers the schema-valid filter, the
dual-embed split (video when frames exist, text-only otherwise), resume, and the
video-error fallback to text-only.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from my_curator.application.reembed import reembed_corpus

pytestmark = pytest.mark.unit

_DIM = 8


class _FakePG:
    def __init__(self, rows):
        self._rows = rows

    async def list_reembed_source(self, *, session_id=None, limit=5000):
        return self._rows


class _FakeEncoder:
    def encode_text(self, text: str):
        return [0.1] * _DIM


class _FakeVideoModel:
    def embed(self, tensor):
        return [0.2] * _DIM


class _FakeHybridRepo:
    def __init__(self):
        self.upserts = []

    async def upsert(self, clip_id, *, text_vec, video_vec=None):
        self.upserts.append({"clip_id": clip_id, "text_vec": text_vec, "video_vec": video_vec})


class _FakeValidator:
    """schema-valid unless dna sets ``_invalid`` True."""

    def validate(self, dna):
        return (not dna.get("_invalid", False), [])


def _row(*, frames: bool = True, scene: bool = True, invalid: bool = False):
    dna = {"planner_logic": {"ego_maneuver": "stop"}}
    if scene:
        dna["scene_description"] = "a scene"
    if invalid:
        dna["_invalid"] = True
    return {
        "clip_id": uuid.uuid4(),
        "dna_json": dna,
        "frames_blob_uri": "frames/s/x" if frames else None,
        "source_clip_id": "src-1",
    }


async def _run(rows, **kw):
    repo = _FakeHybridRepo()
    with patch(
        "my_curator.adapters.storage.frame_loader.load_frames",
        new=AsyncMock(return_value=object()),
    ):
        stats = await reembed_corpus(
            pg=_FakePG(rows),
            minio=AsyncMock(),
            text_encoder=_FakeEncoder(),
            video_model=_FakeVideoModel(),
            hybrid_repo=repo,
            validator=_FakeValidator(),
            **kw,
        )
    return stats, repo


@pytest.mark.asyncio
async def test_dual_embed_split_and_filter():
    rows = [
        _row(frames=True, scene=True),  # → video
        _row(frames=False, scene=True),  # → text-only
        _row(frames=True, scene=False),  # → skipped (no scene_description)
        _row(frames=True, scene=True, invalid=True),  # → skipped (validator fails)
    ]
    stats, repo = await _run(rows)

    assert stats.total == 4
    assert stats.embedded == 2
    assert stats.with_video == 1
    assert stats.text_only == 1
    assert stats.skipped_invalid == 2
    assert len(repo.upserts) == 2
    # exactly one row carries a video vector, one is text-only (None)
    assert sorted([u["video_vec"] is None for u in repo.upserts]) == [False, True]


@pytest.mark.asyncio
async def test_resume_skips_processed():
    rows = [_row(frames=True), _row(frames=True)]
    already = {str(rows[0]["clip_id"])}
    seen: list[str] = []

    async def _on_processed(cid):
        seen.append(cid)

    stats, repo = await _run(rows, processed=already, on_processed=_on_processed)

    assert stats.skipped_resumed == 1
    assert stats.embedded == 1
    assert seen == [str(rows[1]["clip_id"])]  # callback only for the newly processed clip


@pytest.mark.asyncio
async def test_idempotent_upsert_by_clip_id():
    # Re-running over the same rows (no resume set) upserts by clip_id — the
    # fake records one upsert per clip per run; Milvus dedups by PK in reality.
    rows = [_row(frames=True)]
    _, repo1 = await _run(rows)
    _, repo2 = await _run(rows)
    assert repo1.upserts[0]["clip_id"] == repo2.upserts[0]["clip_id"]


@pytest.mark.asyncio
async def test_video_error_falls_back_to_text_only():
    rows = [_row(frames=True, scene=True)]
    repo = _FakeHybridRepo()
    with patch(
        "my_curator.adapters.storage.frame_loader.load_frames",
        new=AsyncMock(side_effect=RuntimeError("frame download failed")),
    ):
        stats = await reembed_corpus(
            pg=_FakePG(rows),
            minio=AsyncMock(),
            text_encoder=_FakeEncoder(),
            video_model=_FakeVideoModel(),
            hybrid_repo=repo,
            validator=_FakeValidator(),
        )

    assert stats.embedded == 1
    assert stats.text_only == 1
    assert stats.with_video == 0
    assert stats.video_errors == 1
    assert repo.upserts[0]["video_vec"] is None
