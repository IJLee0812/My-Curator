"""Thumbnail fallback to a sibling segment's frame (hotfix #84).

Segments of one source clip overlap, so a segment stored without frames of its
own is still covered by a neighbour. No DB, no MinIO — the repository call is
stubbed.
"""

from __future__ import annotations

import pytest

from my_curator.interfaces.http.curation_api.routers.clips import (
    FRAMES_PER_CLIP,
    _sibling_frame_key,
)


class _StubPG:
    def __init__(self, sibling: dict | None):
        self._sibling = sibling
        self.calls: list[tuple[str, float]] = []

    async def find_frames_sibling(self, source_clip_id: str, at_s: float):
        self.calls.append((source_clip_id, at_s))
        return self._sibling


def _row(**over) -> dict:
    return {"source_clip_id": "src-1", "start_s": 8.13, **over}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_picks_the_frame_matching_the_requested_instant():
    """8.13 s sits 80% into a 4.13-9.13 window → index 6 of 0..7."""
    pg = _StubPG({"frames_blob_uri": "frames/s/sib", "start_s": 4.13, "end_s": 9.13})
    assert await _sibling_frame_key(pg, _row()) == "frames/s/sib/frame_6.jpg"
    assert pg.calls == [("src-1", 8.13)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_window_start_maps_to_the_first_frame():
    pg = _StubPG({"frames_blob_uri": "frames/s/sib", "start_s": 8.13, "end_s": 13.13})
    assert await _sibling_frame_key(pg, _row()) == "frames/s/sib/frame_0.jpg"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_window_end_maps_to_the_last_frame():
    pg = _StubPG({"frames_blob_uri": "frames/s/sib", "start_s": 3.13, "end_s": 8.13})
    expected = f"frames/s/sib/frame_{FRAMES_PER_CLIP - 1}.jpg"
    assert await _sibling_frame_key(pg, _row()) == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zero_span_sibling_does_not_divide_by_zero():
    pg = _StubPG({"frames_blob_uri": "frames/s/sib", "start_s": 8.13, "end_s": 8.13})
    assert await _sibling_frame_key(pg, _row()) == "frames/s/sib/frame_0.jpg"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_sibling_with_frames_returns_none():
    """Caller must still 404 — the fallback never invents a frame."""
    assert await _sibling_frame_key(_StubPG(None), _row()) is None


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("row", [_row(source_clip_id=None), _row(start_s=None)])
async def test_rows_without_source_or_start_are_not_looked_up(row):
    pg = _StubPG({"frames_blob_uri": "frames/s/sib", "start_s": 0.0, "end_s": 5.0})
    assert await _sibling_frame_key(pg, row) is None
    assert pg.calls == []
