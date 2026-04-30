"""Integration tests for CurationConsumer.

Two test classes:
  TestCurationConsumerMocked   — AsyncMock DALs; no external services needed.
  TestCurationConsumerIntegration — real Postgres + Milvus (compose.base.yml).

Run only mocked tests (CI):
  pytest tests/integration/test_curation_consumer.py -m "not integration"

Run full suite (requires compose.base.yml running):
  pytest tests/integration/test_curation_consumer.py
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from src.bus.kafka import (
    DNA_VERSION,
    PIPELINE_VERSION,
    ZERO_VECTOR,
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


def _make_scouted_msg(json_valid: bool = True) -> dict:
    return {
        "stream_id": 0,
        "timestamp": 1714000000.0,
        "segment": {"start_time": 0.0, "end_time": 5.0, "duration": 5.0},
        "result": json.dumps(_VALID_DNA) if json_valid else "plain text result",
        "curation": _CURATION_META,
        "metadata": {"source": "vllm-ds-plugin", "version": "1.0", "json_valid": json_valid},
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


def _mock_consumer(pg=None, milvus=None, prompt_hash="abcd1234abcd1234") -> CurationConsumer:
    if pg is None:
        pg = AsyncMock()
    if milvus is None:
        milvus = AsyncMock()
    return CurationConsumer(pg, milvus, prompt_hash, session_id="test-session-001")


# ─── pure-Python helper tests ─────────────────────────────────────────────────


class TestParseDnaJson:
    def test_valid_json_parsed(self):
        result = _parse_dna_json(json.dumps({"scene_summary": "x"}), {})
        assert result["scene_summary"] == "x"

    def test_invalid_json_wraps_raw_text(self):
        result = _parse_dna_json("just a sentence", {})
        assert result["raw_text"] == "just a sentence"

    def test_curation_meta_merged(self):
        result = _parse_dna_json("{}", {"temperature": 0.7})
        assert result["_curation"] == {"temperature": 0.7}

    def test_code_fence_stripped(self):
        inner = json.dumps({"scene_summary": "fenced"})
        result = _parse_dna_json(f"```json\n{inner}\n```", {})
        assert result["scene_summary"] == "fenced"

    def test_non_dict_json_wrapped(self):
        result = _parse_dna_json("[1, 2, 3]", {})
        assert "raw_text" in result


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


# ─── mocked DAL tests ─────────────────────────────────────────────────────────


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
        assert kwargs["dna_version"] == DNA_VERSION
        assert kwargs["pipeline_version"] == PIPELINE_VERSION
        assert isinstance(kwargs["clip_id"], UUID)

    async def test_scouted_calls_milvus_upsert(self):
        milvus = AsyncMock()
        consumer = _mock_consumer(milvus=milvus)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        milvus.upsert.assert_awaited_once()
        call_args = milvus.upsert.call_args
        assert call_args.args[1] == ZERO_VECTOR

    async def test_scouted_skips_milvus_on_pg_failure(self):
        pg = AsyncMock()
        pg.write_clip_with_dna.side_effect = RuntimeError("PG down")
        milvus = AsyncMock()
        consumer = _mock_consumer(pg=pg, milvus=milvus)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        milvus.upsert.assert_not_awaited()
        assert consumer.errors == 1

    async def test_scouted_json_invalid_inserts_review_queue(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg(json_valid=False))
        pg.insert_review_queue.assert_awaited_once()
        kwargs = pg.insert_review_queue.call_args.kwargs
        assert kwargs["state"] == "rejected_schema_invalid"

    async def test_scouted_json_valid_no_review_queue(self):
        pg = AsyncMock()
        consumer = _mock_consumer(pg=pg)
        await consumer.handle("curation.clip.scouted", _make_scouted_msg(json_valid=True))
        pg.insert_review_queue.assert_not_awaited()

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

    async def test_needs_review_does_not_call_milvus(self):
        milvus = AsyncMock()
        consumer = _mock_consumer(milvus=milvus)
        await consumer.handle("curation.clip.needs_review", _make_needs_review_msg())
        milvus.upsert.assert_not_awaited()

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


# ─── real-service integration tests ───────────────────────────────────────────


@pytest.mark.integration
class TestCurationConsumerIntegration:
    """Requires compose.base.yml running (Postgres + Milvus).

    Start stack:  docker compose -f infra/compose.base.yml --env-file .env up -d
    """

    @pytest.fixture
    async def pg(self):
        from src.storage.pg import PGRepository, dsn_from_env

        repo = await PGRepository.create(dsn_from_env())
        yield repo
        await repo.close()

    @pytest.fixture
    async def milvus(self):
        from src.storage.milvus import MilvusRepository

        uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
        repo = await MilvusRepository.create(uri)
        yield repo
        await repo.close()

    @pytest.fixture
    async def session_id(self, pg) -> str:
        sid = "integ-test-session-p24"
        await pg.insert_session(
            session_id=sid,
            dataset="test-dataset",
            subset="val",
            dataset_version="0.0.1",
            recorded_at=datetime.now(timezone.utc),
            source_kind="synthetic",
        )
        return sid

    @pytest.fixture
    def consumer(self, pg, milvus, session_id) -> CurationConsumer:
        prompt_hash = _compute_prompt_hash(_PROMPT_PATH)
        return CurationConsumer(pg, milvus, prompt_hash, session_id)

    async def test_scouted_creates_scenario_dna_row(self, consumer, pg):
        before = await pg.query_dna_by_json('$.scene_summary == "A clear highway stretch."')
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        after = await pg.query_dna_by_json('$.scene_summary == "A clear highway stretch."')
        assert len(after) == len(before) + 1

    async def test_scouted_increases_milvus_count(self, consumer, milvus):
        before = await milvus.count()
        await consumer.handle("curation.clip.scouted", _make_scouted_msg())
        await milvus.flush()
        after = await milvus.count()
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

    async def test_needs_review_no_milvus_write(self, consumer, milvus):
        before = await milvus.count()
        await consumer.handle("curation.clip.needs_review", _make_needs_review_msg())
        await milvus.flush()
        after = await milvus.count()
        assert after == before
