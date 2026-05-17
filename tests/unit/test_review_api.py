"""Unit tests for review API endpoints (P3-5).

Uses a minimal FastAPI app with mocked PGRepository so no Postgres is needed.
"""

from __future__ import annotations

import datetime
import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from my_curator.interfaces.http.curation_api.routers.review import router

pytestmark = pytest.mark.unit


def _make_queue_row(clip_id: str, state: str = "pending") -> dict:
    return {
        "queue_id": 1,
        "clip_id": uuid.UUID(clip_id),
        "state": state,
        "reviewed_at": None,
        "reason": None,
        "created_at": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        "blob_uri": f"clips/sess/{clip_id}.mp4",
        "frames_blob_uri": None,
        "start_s": 0.0,
        "end_s": 5.0,
        "is_gold": False,
        "dna_json": None,
    }


@pytest.fixture()
def pg_mock():
    return AsyncMock()


@pytest.fixture()
def client(pg_mock):
    app = FastAPI()
    app.include_router(router)
    app.state.pg = pg_mock
    return TestClient(app)


# ── GET /v1/review ─────────────────────────────────────────────────────────────


def test_list_review_queue_empty(client, pg_mock):
    pg_mock.get_review_queue.return_value = []
    resp = client.get("/v1/review")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_list_review_queue_returns_items(client, pg_mock):
    clip_id = str(uuid.uuid4())
    pg_mock.get_review_queue.return_value = [_make_queue_row(clip_id, "pending")]
    resp = client.get("/v1/review?status=pending&limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["clip_id"] == clip_id
    assert item["state"] == "pending"
    assert item["reviewed_at"] is None
    pg_mock.get_review_queue.assert_awaited_once_with(status="pending", limit=10)


def test_list_review_queue_no_status_filter(client, pg_mock):
    pg_mock.get_review_queue.return_value = []
    client.get("/v1/review")
    pg_mock.get_review_queue.assert_awaited_once_with(status=None, limit=50)


# ── PATCH /v1/clips/{id}/review ────────────────────────────────────────────────


def test_review_clip_approve(client, pg_mock):
    clip_id = str(uuid.uuid4())
    pg_mock.set_review_status.return_value = None
    resp = client.patch(f"/v1/clips/{clip_id}/review", json={"action": "approve"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["clip_id"] == clip_id
    assert body["state"] == "approved"
    pg_mock.set_review_status.assert_awaited_once_with(uuid.UUID(clip_id), "approved")


def test_review_clip_reject(client, pg_mock):
    clip_id = str(uuid.uuid4())
    pg_mock.set_review_status.return_value = None
    resp = client.patch(f"/v1/clips/{clip_id}/review", json={"action": "reject"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["clip_id"] == clip_id
    assert body["state"] == "rejected"


def test_review_clip_invalid_action(client, pg_mock):
    clip_id = str(uuid.uuid4())
    resp = client.patch(f"/v1/clips/{clip_id}/review", json={"action": "flag"})
    assert resp.status_code == 422


def test_review_clip_invalid_uuid(client, pg_mock):
    resp = client.patch("/v1/clips/not-a-uuid/review", json={"action": "approve"})
    assert resp.status_code == 422
    pg_mock.set_review_status.assert_not_awaited()


def test_review_clip_not_found(client, pg_mock):
    import asyncpg

    clip_id = str(uuid.uuid4())
    pg_mock.set_review_status.side_effect = asyncpg.ForeignKeyViolationError("fk violation")
    resp = client.patch(f"/v1/clips/{clip_id}/review", json={"action": "approve"})
    assert resp.status_code == 404
    assert "clip not found" in resp.json()["detail"]


def test_review_clip_missing_body(client, pg_mock):
    clip_id = str(uuid.uuid4())
    resp = client.patch(f"/v1/clips/{clip_id}/review", json={})
    assert resp.status_code == 422
