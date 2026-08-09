"""Integration tests for CurationConsumer.

Two test classes:
  TestCurationConsumerMocked   — AsyncMock DALs; no external services needed.
  TestCurationConsumerIntegration — real Postgres (compose.base.yml).

Run only mocked tests (CI):
  pytest tests/integration/test_curation_consumer.py -m "not integration"

Run full suite (requires compose.base.yml running):
  pytest tests/integration/test_curation_consumer.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from my_curator.application.consumers.curation_consumer import (
    PIPELINE_VERSION,
    CurationConsumer,
    _compute_prompt_hash,
    _parse_dna_json,
)

# ─── helpers ──────────────────────────────────────────────────────────────────

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "scout_cosmos_reason2.v1.md"

_VALID_DNA = {
    "scene_summary": "A clear highway stretch.",
    "road_type": "highway",
    "road_features": {
        "num_lanes": 4,
        "lane_markings": "dashed_white",
        "road_surface": "dry_asphalt",
        "road_condition": "good",
    },
    "weather": "clear",
    "visibility": "good",
    "traffic_density": "sparse",
    "key_objects": [],
    "ego_vehicle": {"action": "going_straight", "estimated_speed": "fast (>60km/h)"},
    "potential_risks": [],
}

_CURATION_META = {
    "temperature": 0.7,
    "seed": 44,
    "latency_ms": 310.0,
    "partial_sampling": False,
    "n_samples": 3,
    "needs_review": False,
    "reason": None,
}


def _make_scouted_msg(json_valid: bool = True, source_clip_id: str | None = None) -> dict:
    msg: dict = {
        "stream_id": 0,
        "timestamp": 1714000000.0,
        "segment": {"start_time": 0.0, "end_time": 5.0, "duration": 5.0},
        "result": json.dumps(_VALID_DNA) if json_valid else "plain text result",
        "curation": _CURATION_META,
        "metadata": {"source": "vllm-ds-plugin", "version": "1.0", "json_valid": json_valid},
    }
    if source_clip_id is not None:
        msg["source_clip_id"] = source_clip_id
    return msg


# A schema-complete v0.2 DNA. _V2_PROMPT_HASH resolves to dna_version 0.2.0 so the
# consumer's schema gate validates against scenario_dna_v0_2.schema.json.
_V2_PROMPT_HASH = "82e083d75bf378b8"
_VALID_DNA_V02: dict = {
    "dna_version": "0.2.0",
    "clip_id": "12345678-1234-5678-1234-567812345678",
    "timestamp_range": {"start_s": 0.0, "end_s": 5.0},
    "scene_description": (
        "Clear-day cruise on a two-lane primary road with no active intersection. "
        "Ego maintains a steady following gap with no notable actor interactions."
    ),
    "odd": {"weather": "clear", "lighting": "day", "sensor_fidelity": ["clean"]},
    "topology": {"road_type": "primary", "lane_event": "normal", "intersection_type": "none"},
    "actor_dynamics": [],
    "planner_logic": {
        "ego_maneuver": "cruise",
        "risk_level": "nominal",
        "risk_level_rationale": "Clear primary road, no actors within 50 m, ego cruise within limit.",
        "safety_event": {
            "has_event": False,
            "event_type": "none",
            "collision_type": None,
            "severity_estimate": None,
        },
    },
    "confidence": {"overall": 0.9, "scout_agreement": 1.0, "hallucination_flags": []},
    "provenance": {
        "scout_models": ["cosmos-reason2-8b"],
        "scout_prompt_hash": "abcd1234ef567890",
        "pipeline_version": "p2-6",
        "is_synthetic": False,
        "reference_standards": ["ASAM OSI v3.x", "OpenDRIVE v1.5M"],
    },
}


def _make_scouted_msg_v2() -> dict:
    """Scouted message carrying a schema-complete v0.2 DNA (validates clean)."""
    msg = _make_scouted_msg(json_valid=True)
    msg["result"] = json.dumps(_VALID_DNA_V02)
    return msg


def _make_ingest_msg(session_id: str = "ingest-session-001") -> dict:
    """Simulate a /v1/ingest Kafka payload (P3-2 format)."""
    import uuid as _uuid

    return {
        "clip_id": str(_uuid.uuid4()),
        "session_id": session_id,
        "blob_uri": f"clips/{session_id}/test.mp4",
        "start_s": 0.0,
        "end_s": 5.0,
        "dna_version": "0.1",
        "dna_json": _VALID_DNA,
        "scout_prompt_hash": "aabbccddaabbccdd",
        "pipeline_version": "0.1.0",
        "frame_count": None,
        "is_synthetic": False,
        "judge_prompt_hash": None,
        "curation_meta": None,
    }


def _make_needs_review_msg(reason: str = "partial_batch") -> dict:
    return {
        "stream_id": 1,
        "timestamp": 1714000005.0,
        "segment": {"start_time": 5.0, "end_time": 10.0, "duration": 5.0},
        "result": "partial output text",
        "curation": {
            **_CURATION_META,
            "partial_sampling": True,
            "needs_review": True,
            "reason": reason,
        },
        "metadata": {"source": "vllm-ds-plugin", "version": "1.0", "json_valid": False},
    }


def _mock_consumer(pg=None, prompt_hash="abcd1234abcd1234") -> CurationConsumer:
    if pg is None:
        pg = AsyncMock()
    return CurationConsumer(pg, prompt_hash, session_id="test-session-001")


# ─── pure-Python helper tests ─────────────────────────────────────────────────


@pytest.mark.integration
class TestParseDnaJson:
    def test_valid_json_parsed(self):
        dna, _ = _parse_dna_json(json.dumps({"scene_summary": "x"}), {})
        assert dna["scene_summary"] == "x"

    def test_invalid_json_wraps_raw_text(self):
        dna, _ = _parse_dna_json("just a sentence", {})
        assert dna["raw_text"] == "just a sentence"

    def test_curation_meta_returned_separately(self):
        """_curation is no longer merged into dna_json — it is returned as the second tuple element."""
        dna, meta = _parse_dna_json("{}", {"temperature": 0.7})
        assert "_curation" not in dna
        assert meta["temperature"] == 0.7

    def test_code_fence_stripped(self):
        inner = json.dumps({"scene_summary": "fenced"})
        dna, _ = _parse_dna_json(f"```json\n{inner}\n```", {})
        assert dna["scene_summary"] == "fenced"

    def test_non_dict_json_wrapped(self):
        dna, _ = _parse_dna_json("[1, 2, 3]", {})
        assert "raw_text" in dna


@pytest.mark.integration
class TestComputePromptHash:
    def test_hash_is_16_hex_chars(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("hello")
        h = _compute_prompt_hash(f)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_bytes(b"content")
        assert _compute_prompt_hash(f) == _compute_prompt_hash(f)

    def test_hash_matches_sha256(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_bytes(b"data")
        expected = hashlib.sha256(b"data").hexdigest()[:16]
        assert _compute_prompt_hash(f) == expected

    def test_prompt_file_exists(self):
        assert _PROMPT_PATH.exists(), f"Missing: {_PROMPT_PATH}"

    def test_prompt_hash_stable(self):
        h = _compute_prompt_hash(_PROMPT_PATH)
        assert len(h) == 16


# ─── segment time refinement (sidecar clamp at persistence boundary) ──────────


def _write_sidecar(video_root: Path, rel_video: str, fps: int, n_frames: int) -> None:
    ts_path = (video_root / rel_video).with_suffix(".timestamp")
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"FPS,{fps}", "Size,1920,1080"]
    rows += [f"{i},{1714000000000000 + i * 1_000_000 // fps}" for i in range(n_frames)]
    ts_path.write_text("\n".join(rows) + "\n")


@pytest.mark.integration
class TestSegmentTimeRefinement:
    """Sidecar present → persisted start_s/end_s are frame-aligned and clamped
    to the video's real duration (10 s @30 fps → end 299/30 s, not 13.1 s)."""

    _REL = "sess-01/00001/video/00001.mp4"

    def _msg(self, base: dict) -> dict:
        base["segment"] = {"start_time": 8.1, "end_time": 13.1, "duration": 5.0}
        base["source_video_path"] = self._REL
        return base

    async def test_scouted_end_clamped_to_video_duration(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIDEO_DATA_ROOT", str(tmp_path))
        _write_sidecar(tmp_path, self._REL, fps=30, n_frames=300)
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", self._msg(_make_scouted_msg()))
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["start_s"] == pytest.approx(8.1)
        assert kwargs["end_s"] == pytest.approx(299 / 30)

    async def test_scouted_dna_timestamp_range_matches_clamped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIDEO_DATA_ROOT", str(tmp_path))
        _write_sidecar(tmp_path, self._REL, fps=30, n_frames=300)
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", self._msg(_make_scouted_msg()))
        tr = pg.write_clip_with_dna.call_args.kwargs["dna_json"]["timestamp_range"]
        assert tr["end_s"] == pytest.approx(299 / 30)

    async def test_needs_review_end_clamped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIDEO_DATA_ROOT", str(tmp_path))
        _write_sidecar(tmp_path, self._REL, fps=30, n_frames=300)
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.needs_review", self._msg(_make_needs_review_msg()))
        assert pg.write_clip_with_dna.call_args.kwargs["end_s"] == pytest.approx(299 / 30)

    async def test_no_sidecar_keeps_raw_times(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIDEO_DATA_ROOT", str(tmp_path))
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", self._msg(_make_scouted_msg()))
        assert pg.write_clip_with_dna.call_args.kwargs["end_s"] == pytest.approx(13.1)

    async def test_no_video_root_keeps_raw_times(self, monkeypatch):
        monkeypatch.delenv("VIDEO_DATA_ROOT", raising=False)
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", self._msg(_make_scouted_msg()))
        assert pg.write_clip_with_dna.call_args.kwargs["end_s"] == pytest.approx(13.1)


# ─── mocked DAL tests ─────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCurationConsumerMocked:
    async def test_scouted_calls_write_clip_with_dna(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        pg.write_clip_with_dna.assert_awaited_once()

    async def test_scouted_write_clip_with_dna_args(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["session_id"] == "test-session-001"
        assert kwargs["blob_uri"] == "stream://0/0.00-5.00"
        assert kwargs["dna_version"] == "0.1.0"
        assert kwargs["pipeline_version"] == PIPELINE_VERSION
        assert isinstance(kwargs["clip_id"], UUID)

    async def test_scouted_pg_failure_increments_errors(self):
        pg = AsyncMock()
        pg.write_clip_with_dna.side_effect = RuntimeError("PG down")
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        assert consumer.errors == 1

    async def test_scouted_json_invalid_inserts_review_queue(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg(json_valid=False))
        pg.insert_review_queue.assert_awaited_once()
        kwargs = pg.insert_review_queue.call_args.kwargs
        assert kwargs["state"] == "rejected_schema_invalid"

    async def test_scouted_json_valid_inserts_pending(self):
        # Schema-complete v0.2 DNA under a hash resolving to 0.2.0 → passes the
        # consumer schema gate → pending.
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg, prompt_hash=_V2_PROMPT_HASH)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg_v2())
        pg.insert_review_queue.assert_awaited_once()
        assert pg.insert_review_queue.call_args.kwargs["state"] == "pending"

    async def test_scouted_json_valid_but_schema_incomplete_inserts_rejected(self):
        # Parses as JSON (json_valid=True) but misses required nested fields
        # (odd, planner_logic, …). The schema gate must flag it as
        # rejected_schema_invalid rather than storing it as pending.
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg, prompt_hash=_V2_PROMPT_HASH)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg(json_valid=True))
        pg.insert_review_queue.assert_awaited_once()
        assert pg.insert_review_queue.call_args.kwargs["state"] == "rejected_schema_invalid"

    async def test_needs_review_calls_write_clip_with_dna(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.needs_review", _make_needs_review_msg())
        pg.write_clip_with_dna.assert_awaited_once()

    async def test_needs_review_inserts_review_queue_pending(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.needs_review", _make_needs_review_msg("partial_batch"))
        pg.insert_review_queue.assert_awaited_once()
        kwargs = pg.insert_review_queue.call_args.kwargs
        assert kwargs["state"] == "pending"
        assert kwargs["reason"] == "partial_batch"

    async def test_processed_counter_increments(self):
        consumer = _mock_consumer()
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        await consumer.handle("curation.clip.needs_review", _make_needs_review_msg())
        assert consumer.processed == 2

    async def test_unknown_topic_not_counted(self):
        consumer = _mock_consumer()
        await consumer.handle("some.other.topic", {})
        assert consumer.processed == 0

    async def test_errors_not_counted_on_success(self):
        consumer = _mock_consumer()
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        assert consumer.errors == 0

    # ── ingest-format (P3-2) tests ────────────────────────────────────────────

    async def test_scouted_ingest_calls_write_clip_with_dna(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_ingest_msg())
        pg.write_clip_with_dna.assert_awaited_once()

    async def test_scouted_ingest_uses_message_session_id(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_ingest_msg(session_id="my-session"))
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["session_id"] == "my-session"

    async def test_scouted_ingest_uses_precomputed_dna(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_ingest_msg())
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["dna_json"] == _VALID_DNA
        assert kwargs["dna_version"] == "0.1"

    async def test_scouted_ingest_calls_insert_session(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_ingest_msg(session_id="s1"))
        pg.insert_session.assert_awaited_once()
        kwargs = pg.insert_session.call_args.kwargs
        assert kwargs["session_id"] == "s1"

    async def test_scouted_ingest_pg_failure_increments_errors(self):
        pg = AsyncMock()
        pg.write_clip_with_dna.side_effect = RuntimeError("PG down")
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_ingest_msg())
        assert consumer.errors == 1

    async def test_scouted_ingest_counted_in_processed(self):
        consumer = _mock_consumer()
        await consumer.handle("curation.clip.scouted", _make_ingest_msg())
        assert consumer.processed == 1

    # ── P3-4: source_clip_id thread-through ───────────────────────────────────

    async def test_scouted_threads_source_clip_id(self):
        """source_clip_id from the Kafka message reaches write_clip_with_dna (P3-4)."""
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle(
            "curation.clip.scouted",
            _make_scouted_msg(source_clip_id="00042"),
        )
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["source_clip_id"] == "00042"

    async def test_scouted_no_source_clip_id_passes_none(self):
        """When the message omits source_clip_id, the consumer passes None (P3-4)."""
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["source_clip_id"] is None

    async def test_scouted_ingest_threads_source_clip_id(self):
        """The /v1/ingest path also threads source_clip_id when present (P3-4)."""
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        msg = _make_ingest_msg()
        msg["source_clip_id"] = "00099"
        await consumer.handle("curation.clip.scouted", msg)
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["source_clip_id"] == "00099"

    async def test_needs_review_threads_source_clip_id(self):
        """The needs-review handler threads source_clip_id too (P3-4)."""
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        msg = _make_needs_review_msg()
        msg["source_clip_id"] = "00007"
        await consumer.handle("curation.clip.needs_review", msg)
        kwargs = pg.write_clip_with_dna.call_args.kwargs
        assert kwargs["source_clip_id"] == "00007"


# ─── real-service integration tests ───────────────────────────────────────────


@pytest.mark.integration
class TestCurationConsumerIntegration:
    """Requires compose.base.yml running (Postgres).

    Start stack:  docker compose -f infra/compose.base.yml --env-file .env up -d
    """

    @pytest.fixture
    async def pg(self):
        from my_curator.adapters.storage.pg import PGRepository, dsn_from_env

        repo = await PGRepository.create(dsn_from_env())
        yield repo
        await repo.close()

    @pytest.fixture
    async def session_id(self, pg):
        sid = "integ-test-session-p24"
        await pg.insert_session(
            session_id=sid,
            dataset="test-dataset",
            subset="val",
            dataset_version="0.0.1",
            recorded_at=datetime.now(timezone.utc),
            source_kind="synthetic",
        )
        yield sid
        await pg._pool.execute("DELETE FROM sessions WHERE session_id = $1", sid)

    @pytest.fixture
    def consumer(self, pg, session_id) -> CurationConsumer:
        prompt_hash = _compute_prompt_hash(_PROMPT_PATH)
        return CurationConsumer(pg, prompt_hash, session_id)

    async def test_scouted_creates_scenario_dna_row(self, consumer, pg):
        before = await pg.query_dna_by_json('$.scene_summary == "A clear highway stretch."')
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        after = await pg.query_dna_by_json('$.scene_summary == "A clear highway stretch."')
        assert len(after) == len(before) + 1

    async def test_scouted_creates_clip_row(self, consumer, pg):
        """PG `clips` is the system of record — count must grow on every scouted message."""
        before = await pg._pool.fetchval("SELECT count(*) FROM clips")
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        after = await pg._pool.fetchval("SELECT count(*) FROM clips")
        assert after == before + 1

    async def test_scouted_json_invalid_creates_review_queue_row(self, consumer, pg):
        await consumer.handle("curation.clip.scouted", _make_scouted_msg(json_valid=False))
        rows = await pg._pool.fetch(
            "SELECT state FROM review_queue WHERE state = 'rejected_schema_invalid' ORDER BY created_at DESC LIMIT 1"
        )
        assert rows, "No rejected_schema_invalid row found"
        assert rows[0]["state"] == "rejected_schema_invalid"

    async def test_needs_review_creates_pending_row(self, consumer, pg):
        await consumer.handle("curation.clip.needs_review", _make_needs_review_msg())
        rows = await pg._pool.fetch(
            "SELECT state, reason FROM review_queue WHERE state = 'pending' ORDER BY created_at DESC LIMIT 1"
        )
        assert rows, "No pending row found"
        assert rows[0]["reason"] == "partial_batch"
