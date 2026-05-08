"""E2E tests for curation-api (P3-2).

Requires the full compose stack:
  docker compose -f infra/compose.base.yml -f infra/compose.curate.yml \\
      --env-file .env up -d

The curation-api container must be healthy (model warm-up complete) before
these tests are collected.  All tests are marked ``e2e``.

Run with:
    pytest tests/e2e/test_curation_api.py -m e2e -v
"""

from __future__ import annotations

import json
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
def seeded_clip_id(client):
    """Ingest one real clip into the pipeline and return its clip_id.

    Publishes to Kafka; CurationConsumer writes PG + embedding worker
    writes Milvus.  We poll /v1/clips/{id} until the row appears (up to
    30 s) so subsequent search tests have data to find.
    """
    clip_id = str(uuid.uuid4())
    session_id = "e2e-session-001"
    payload = {
        "clip_id": clip_id,
        "session_id": session_id,
        "blob_uri": f"clips/{session_id}/{clip_id}.mp4",
        "start_s": 0.0,
        "end_s": 5.0,
        "dna_version": "0.1",
        "dna_json": {
            "dna_version": "0.1",
            "clip_id": clip_id,
            "timestamp_range": {"start_s": 0.0, "end_s": 5.0},
            "odd": {
                "weather": "clear",
                "lighting": "daylight",
                "sensor_fidelity": ["camera"],
            },
            "topology": {
                "road_type": "urban",
                "lane_event": "straight",
                "intersection_type": "none",
            },
            "actor_dynamics": [
                {
                    "actor_class": "vehicle",
                    "state": "moving",
                    "distance_bucket": "near",
                    "confidence": 0.9,
                    "grounded_by_yolo26": True,
                }
            ],
            "planner_logic": {
                "ego_maneuver": "lane_keep",
                "risk_level": "low",
                "causal_trigger_actor_index": None,
            },
            "confidence": 0.85,
            "provenance": {"scout_model": "Cosmos-Reason2-8B-FP8"},
        },
        "scout_prompt_hash": "aabbccdd",
        "pipeline_version": "0.1.0",
    }
    resp = client.post("/v1/ingest", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["published"] is True

    deadline = time.time() + 30
    while time.time() < deadline:
        r = client.get(f"/v1/clips/{clip_id}")
        if r.status_code == 200:
            return clip_id
        time.sleep(1)

    pytest.skip(f"Clip {clip_id} not found in PG within 30 s — CurationConsumer may be down")


# ── tests ─────────────────────────────────────────────────────────────────────


def test_health_ready(client):
    """/health returns 200 after model warm-up."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_returns_results(client, seeded_clip_id):
    """POST /v1/search with a natural-language query returns ≥1 result."""
    resp = client.post(
        "/v1/search",
        json={"query": "urban daylight clear weather driving", "limit": 10},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] >= 1
    result = body["results"][0]
    assert "clip_id" in result
    assert "score" in result
    assert isinstance(result["score"], float)


def test_search_dna_filter(client, seeded_clip_id):
    """DNA filter excludes clips that do not match the requested field value."""
    resp_match = client.post(
        "/v1/search",
        json={
            "query": "urban driving",
            "filters": {"weather": "clear"},
            "limit": 100,
        },
    )
    assert resp_match.status_code == 200
    matching_ids = {r["clip_id"] for r in resp_match.json()["results"]}

    resp_no_match = client.post(
        "/v1/search",
        json={
            "query": "urban driving",
            "filters": {"weather": "fog"},
            "limit": 100,
        },
    )
    assert resp_no_match.status_code == 200
    fog_ids = {r["clip_id"] for r in resp_no_match.json()["results"]}

    assert seeded_clip_id in matching_ids
    assert seeded_clip_id not in fog_ids


def test_search_latency_p95(client, seeded_clip_id):
    """10 consecutive steady-state queries must have p95 latency < 500 ms."""
    latencies: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.post("/v1/search", json={"query": "urban driving", "limit": 20})
        latencies.append(time.perf_counter() - t0)
        assert resp.status_code == 200

    latencies.sort()
    p95_ms = latencies[int(len(latencies) * 0.95)] * 1000
    assert p95_ms < 500, f"p95 latency {p95_ms:.1f} ms exceeds 500 ms SLA"


def test_ingest_publishes_to_kafka(client):
    """POST /v1/ingest returns published=True for a well-formed payload."""
    clip_id = str(uuid.uuid4())
    payload = {
        "clip_id": clip_id,
        "session_id": "e2e-kafka-test",
        "blob_uri": f"clips/e2e-kafka-test/{clip_id}.mp4",
        "start_s": 0.0,
        "end_s": 3.0,
        "dna_version": "0.1",
        "dna_json": {"dna_version": "0.1", "clip_id": clip_id},
        "scout_prompt_hash": "deadbeef",
        "pipeline_version": "0.1.0",
    }
    resp = client.post("/v1/ingest", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["published"] is True
    assert body["topic"] == "curation.clip.scouted"
    assert body["clip_id"] == clip_id


def test_clips_detail(client, seeded_clip_id):
    """GET /v1/clips/{id} returns DNA JSON and a presigned MinIO URL."""
    resp = client.get(f"/v1/clips/{seeded_clip_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["clip_id"] == seeded_clip_id
    assert body["dna_json"] is not None
    assert body["presigned_url"].startswith("http")
