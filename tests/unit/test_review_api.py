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
    pg_mock.count_review_queue.return_value = 0
    resp = client.get("/v1/review")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["page"] == 1
    assert body["size"] == 30


def test_list_review_queue_returns_items(client, pg_mock):
    clip_id = str(uuid.uuid4())
    pg_mock.get_review_queue.return_value = [_make_queue_row(clip_id, "pending")]
    pg_mock.count_review_queue.return_value = 42
    resp = client.get("/v1/review?status=pending&page=2&size=50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 42  # tab total, not the page length
    assert body["page"] == 2
    assert body["size"] == 50
    item = body["items"][0]
    assert item["clip_id"] == clip_id
    assert item["state"] == "pending"
    assert item["reviewed_at"] is None
    # page 2 @ size 50 → offset 50
    pg_mock.get_review_queue.assert_awaited_once_with(
        status="pending", risk=None, limit=50, offset=50
    )
    pg_mock.count_review_queue.assert_awaited_once_with(status="pending", risk=None)


def test_list_review_queue_no_status_filter(client, pg_mock):
    pg_mock.get_review_queue.return_value = []
    pg_mock.count_review_queue.return_value = 0
    client.get("/v1/review")
    pg_mock.get_review_queue.assert_awaited_once_with(status=None, risk=None, limit=30, offset=0)


def test_list_review_queue_risk_filter_reaches_both_queries(client, pg_mock):
    """The risk filter must narrow the count too, or pagination would offer
    pages the filtered list cannot fill."""
    pg_mock.get_review_queue.return_value = []
    pg_mock.count_review_queue.return_value = 1
    resp = client.get("/v1/review?status=pending&risk=critical")
    assert resp.status_code == 200
    pg_mock.get_review_queue.assert_awaited_once_with(
        status="pending", risk="critical", limit=30, offset=0
    )
    pg_mock.count_review_queue.assert_awaited_once_with(status="pending", risk="critical")


def test_list_review_queue_rejects_unknown_risk(client, pg_mock):
    resp = client.get("/v1/review?risk=catastrophic")
    assert resp.status_code == 422
    pg_mock.get_review_queue.assert_not_awaited()


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


# ── SQL predicate builder (pure, no DB) ────────────────────────────────────────


class TestReviewFilters:
    """PGRepository._review_filters composes the status and risk predicates."""

    @staticmethod
    def _build(status, risk):
        from my_curator.adapters.storage.pg import PGRepository

        return PGRepository._review_filters(status, risk)

    def test_no_filters_is_no_where_clause(self):
        assert self._build(None, None) == ("", [])
        assert self._build("all", "all") == ("", [])

    def test_risk_only_opens_the_where_clause(self):
        where, params = self._build(None, "critical")
        assert where.startswith("WHERE ")
        assert "planner_logic" in where and "$1" in where
        assert params == ["critical"]

    def test_status_and_risk_are_anded_with_distinct_placeholders(self):
        where, params = self._build("pending", "elevated")
        assert where.count("WHERE") == 1
        assert " AND " in where
        assert "$1" in where and "$2" in where
        assert params == ["pending", "elevated"]

    def test_schema_invalid_status_binds_no_param_so_risk_takes_dollar_one(self):
        """The schema_invalid tab inlines its state literal, so a risk filter
        must still bind at $1 rather than assuming a status param exists."""
        where, params = self._build("schema_invalid", "nominal")
        assert "rejected_schema_invalid" in where
        assert params == ["nominal"]
        assert "$1" in where and "$2" not in where
