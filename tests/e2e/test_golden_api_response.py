"""GT-4: golden curation-api response envelope.

Verifies ``POST /v1/search`` returns a stable response envelope for five canned
queries.  The exact ranking is data-dependent and not asserted; the envelope
shape (top-level keys + per-item key set) is locked.

Requires the compose stack (curation-api + Milvus + Postgres + MinIO).
Skipped automatically when curation-api is unreachable.

References:
  docs/refactoring_plan.md  §3.1 GT-4.
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


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(httpx is None, reason="httpx not installed"),
    pytest.mark.skipif(not _api_reachable(), reason="curation-api not reachable on compose"),
]


_GOLDEN_QUERIES = [
    "a car on a clear day",
    "rainy night highway",
    "pedestrian crossing at intersection",
    "truck merging onto highway",
    "cyclist on urban road",
]

_EXPECTED_TOP_KEYS = {"results", "total"}
_EXPECTED_ITEM_KEYS = {"clip_id"}


@pytest.mark.parametrize(
    "query", _GOLDEN_QUERIES, ids=[q.replace(" ", "_") for q in _GOLDEN_QUERIES]
)
def test_golden_search_envelope(query):
    with httpx.Client(base_url=_API_BASE, timeout=10.0) as client:
        r = client.post("/v1/search", json={"query": query, "limit": 5, "top_k": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert _EXPECTED_TOP_KEYS.issubset(body.keys()), (
        f"Missing top-level keys: {_EXPECTED_TOP_KEYS - body.keys()}"
    )
    assert isinstance(body["results"], list)
    import uuid as _uuid

    for item in body["results"]:
        assert _EXPECTED_ITEM_KEYS.issubset(item.keys()), (
            f"Missing per-item keys: {_EXPECTED_ITEM_KEYS - item.keys()}"
        )
        # clip_id round-trips as UUID string.
        _uuid.UUID(item["clip_id"])


def test_golden_search_limit_respected():
    with httpx.Client(base_url=_API_BASE, timeout=10.0) as client:
        r = client.post(
            "/v1/search", json={"query": "a car on a clear day", "limit": 3, "top_k": 50}
        )
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 3
