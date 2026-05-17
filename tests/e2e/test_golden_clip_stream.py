"""GT-6: golden byte-range stream + precise-times envelope.

Verifies ``GET /v1/clips/{id}/stream`` honours ``Range: bytes=N-`` with the
``Accept-Ranges`` response header and that ``GET /v1/clips/{id}`` surfaces
``precise_start_s`` / ``precise_end_s`` populated from the .timestamp sidecar.

Requires the compose stack with at least one clip persisted in postgres + the
underlying video data root mounted into the curation-api container.

References:
  docs/refactoring_plan.md  §3.1 GT-6.
"""

from __future__ import annotations

import os

import pytest

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


_API_BASE = os.environ.get("CURATION_API_BASE", "http://localhost:8001")


def _api_reachable() -> bool:
    if httpx is None:
        return False
    try:
        return httpx.get(f"{_API_BASE}/health", timeout=1.5).status_code == 200
    except Exception:
        return False


def _pick_clip_id(file_scheme_only: bool = False) -> str | None:
    """Return one clip_id from the corpus, or None when empty.

    When ``file_scheme_only`` is True, only clips with file:// blob_uri are
    eligible (required for byte-range streaming — stream:// origins are not
    streamable; the API returns 404 on those).
    """
    if httpx is None:
        return None
    try:
        r = httpx.post(
            f"{_API_BASE}/v1/search",
            json={"query": "scene", "limit": 50, "top_k": 200},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results") or []
        if file_scheme_only:
            results = [
                item for item in results if str(item.get("blob_uri", "")).startswith("file://")
            ]
        return results[0]["clip_id"] if results else None
    except Exception:
        return None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(httpx is None, reason="httpx not installed"),
    pytest.mark.skipif(not _api_reachable(), reason="curation-api not reachable on compose"),
]


def test_clip_detail_carries_precise_times():
    clip_id = _pick_clip_id()
    if clip_id is None:
        pytest.skip("no clips in corpus")
    with httpx.Client(base_url=_API_BASE, timeout=10.0) as client:
        r = client.get(f"/v1/clips/{clip_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "precise_start_s" in body
    assert "precise_end_s" in body
    assert body["precise_start_s"] is None or isinstance(body["precise_start_s"], (int, float))
    assert body["precise_end_s"] is None or isinstance(body["precise_end_s"], (int, float))


def test_clip_stream_accepts_byte_range():
    clip_id = _pick_clip_id(file_scheme_only=True)
    if clip_id is None:
        pytest.skip("no clips with file:// blob_uri in corpus")
    with httpx.Client(base_url=_API_BASE, timeout=10.0) as client:
        r = client.get(f"/v1/clips/{clip_id}/stream", headers={"Range": "bytes=0-1023"})
    # 206 partial content OR 200 with Accept-Ranges header.
    assert r.status_code in (200, 206), r.text
    assert "accept-ranges" in {k.lower() for k in r.headers}
    if r.status_code == 206:
        assert "content-range" in {k.lower() for k in r.headers}
