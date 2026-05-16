"""E2E tests for the P3-5 Verify-by-Exception review workflow.

Requires the full compose stack:
  docker compose -f infra/compose.base.yml -f infra/compose.curate.yml \\
      --env-file .env up -d

Run with:
    pytest tests/e2e/test_review_workflow.py -m e2e -v
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

BASE_URL = os.environ.get("CURATION_API_URL", "http://localhost:8001")

pytestmark = pytest.mark.e2e


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c


@pytest.fixture(scope="module")
def review_clip_id(client):
    """Ingest a clip and wait for it to appear in the review queue.

    Returns the clip_id string once the clip row exists in PG.
    Skips if the CurationConsumer does not write within 30 s.
    """
    clip_id = str(uuid.uuid4())
    session_id = "e2e-review-session"
    payload = {
        "clip_id": clip_id,
        "session_id": session_id,
        "blob_uri": f"clips/{session_id}/{clip_id}.mp4",
        "start_s": 0.0,
        "end_s": 4.0,
        "dna_version": "0.1",
        "dna_json": {
            "dna_version": "0.1",
            "clip_id": clip_id,
            "timestamp_range": {"start_s": 0.0, "end_s": 4.0},
            "odd": {"weather": "clear", "lighting": "day", "sensor_fidelity": ["camera"]},
            "topology": {"road_type": "highway", "lane_event": "normal", "intersection_type": "none"},
            "actor_dynamics": [],
            "planner_logic": {
                "ego_maneuver": "lane_keep",
                "risk_level": "nominal",
                "causal_trigger_actor_index": None,
            },
            "confidence": {"overall": 0.9, "scout_agreement": 0.85, "hallucination_flags": []},
            "provenance": {
                "scout_models": ["Cosmos-Reason2-8B-FP8"],
                "scout_prompt_hash": "deadbeef",
                "judge_model": None,
                "judge_prompt_hash": None,
                "pipeline_version": "0.1.0",
                "is_synthetic": False,
                "reference_standards": [],
            },
        },
        "scout_prompt_hash": "deadbeef",
        "pipeline_version": "0.1.0",
    }

    resp = client.post("/v1/ingest", json=payload)
    assert resp.status_code == 200, resp.text

    deadline = time.time() + 30
    while time.time() < deadline:
        r = client.get("/v1/review", params={"status": "pending", "limit": 200})
        if r.status_code == 200:
            ids = [i["clip_id"] for i in r.json().get("items", [])]
            if clip_id in ids:
                return clip_id
        time.sleep(1)

    pytest.skip(f"Clip {clip_id} not in review queue after 30 s — CurationConsumer may be down")


# ── tests ─────────────────────────────────────────────────────────────────────


def test_review_queue_lists_pending(client, review_clip_id):
    """GET /v1/review?status=pending includes the ingested clip."""
    resp = client.get("/v1/review", params={"status": "pending", "limit": 200})
    assert resp.status_code == 200
    body = resp.json()
    ids = [i["clip_id"] for i in body["items"]]
    assert review_clip_id in ids, f"Expected {review_clip_id} in pending queue, got: {ids}"


def test_clip_detail_review_status_pending(client, review_clip_id):
    """GET /v1/clips/{id} includes review_status=pending before any action."""
    resp = client.get(f"/v1/clips/{review_clip_id}")
    assert resp.status_code == 200
    assert resp.json()["review_status"] == "pending"


def test_approve_clip(client, review_clip_id):
    """PATCH /v1/clips/{id}/review approve → state becomes approved."""
    resp = client.patch(
        f"/v1/clips/{review_clip_id}/review",
        json={"action": "approve"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["clip_id"] == review_clip_id
    assert body["state"] == "approved"


def test_clip_detail_review_status_approved(client, review_clip_id):
    """GET /v1/clips/{id} reflects approved status after PATCH."""
    resp = client.get(f"/v1/clips/{review_clip_id}")
    assert resp.status_code == 200
    assert resp.json()["review_status"] == "approved"


def test_approved_appears_in_approved_queue(client, review_clip_id):
    """GET /v1/review?status=approved includes the newly approved clip."""
    resp = client.get("/v1/review", params={"status": "approved", "limit": 200})
    assert resp.status_code == 200
    ids = [i["clip_id"] for i in resp.json()["items"]]
    assert review_clip_id in ids


def test_approved_not_in_pending_queue(client, review_clip_id):
    """Approved clip must no longer appear in the pending tab."""
    resp = client.get("/v1/review", params={"status": "pending", "limit": 200})
    assert resp.status_code == 200
    ids = [i["clip_id"] for i in resp.json()["items"]]
    assert review_clip_id not in ids


def test_reject_overwrites_approve(client, review_clip_id):
    """PATCH reject after approve: UPSERT must overwrite state to rejected."""
    resp = client.patch(
        f"/v1/clips/{review_clip_id}/review",
        json={"action": "reject"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "rejected"

    detail = client.get(f"/v1/clips/{review_clip_id}")
    assert detail.json()["review_status"] == "rejected"


def test_rejected_appears_in_rejected_queue(client, review_clip_id):
    """GET /v1/review?status=rejected includes the clip after reject overwrites approve."""
    resp = client.get("/v1/review", params={"status": "rejected", "limit": 200})
    assert resp.status_code == 200
    ids = [i["clip_id"] for i in resp.json()["items"]]
    assert review_clip_id in ids


def test_invalid_action_returns_422(client, review_clip_id):
    """PATCH with unknown action must return 422."""
    resp = client.patch(
        f"/v1/clips/{review_clip_id}/review",
        json={"action": "flag"},
    )
    assert resp.status_code == 422


def test_invalid_uuid_returns_422(client):
    """PATCH with a non-UUID clip_id must return 422."""
    resp = client.patch("/v1/clips/not-a-uuid/review", json={"action": "approve"})
    assert resp.status_code == 422


def test_unknown_clip_returns_404(client):
    """PATCH for a clip_id that does not exist in PG must return 404."""
    resp = client.patch(
        f"/v1/clips/{uuid.uuid4()}/review",
        json={"action": "approve"},
    )
    assert resp.status_code == 404
