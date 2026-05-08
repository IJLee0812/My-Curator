"""End-to-end pipeline tests.

These tests launch main.py via subprocess and verify the pipeline runs
to completion without errors.  They require a real NVIDIA GPU and a
GStreamer / DeepStream environment, so they are automatically skipped
on any host that lacks either.

Run inside the DS 9.0 Docker container:
    pytest tests/e2e -v

Note: GStreamer presence is detected via binary (gst-launch-1.0 / gst-inspect-1.0)
rather than Python imports to avoid false-positives from the integration-test mocks.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

WORKSPACE = Path("/workspace")
MAIN_PY = WORKSPACE / "main.py"
SAMPLE_MP4 = WORKSPACE / "assets" / "videos" / "sample.mp4"
# Unified config — same file drives both Pure VLM and VLM+Detect modes.
CONFIG_DRIVING = WORKSPACE / "configs" / "config_driving_scene.yaml"
# Detect mode requires the YOLO26 ONNX model to already be present.
YOLO26_ONNX = WORKSPACE / "models" / "yolo26m.onnx"
SCHEMA_PATH = WORKSPACE / "schemas" / "scenario_dna_v0_1.schema.json"
_KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")


def _nvidia_gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _gstreamer_available() -> bool:
    """Detect GStreamer via binary — not Python import (which may be mocked)."""
    return shutil.which("gst-launch-1.0") is not None or shutil.which("gst-inspect-1.0") is not None


_IN_DOCKER_WORKSPACE = MAIN_PY.exists()


def _kafka_available() -> bool:
    """TCP reachability check for Kafka broker (proxy for compose stack availability)."""
    import socket

    try:
        host, port = _KAFKA_BROKER.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _nvidia_gpu_available(), reason="No NVIDIA GPU detected")
requires_gstreamer = pytest.mark.skipif(
    not _gstreamer_available(), reason="GStreamer not available"
)
requires_workspace = pytest.mark.skipif(
    not _IN_DOCKER_WORKSPACE,
    reason="Not running inside Docker workspace (/workspace/main.py not found)",
)
requires_yolo26m_onnx = pytest.mark.skipif(
    not YOLO26_ONNX.exists(),
    reason=f"Detect mode requires {YOLO26_ONNX} (export via scripts/export_yolo26.py)",
)
requires_compose = pytest.mark.skipif(
    not _kafka_available(),
    reason=f"Kafka not reachable at {_KAFKA_BROKER} — compose stack likely down",
)


def _run_pipeline(*extra_args, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(MAIN_PY)] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@requires_workspace
@requires_gpu
@requires_gstreamer
@pytest.mark.e2e
class TestPureVLMPipeline:
    def test_dry_run_exits_zero(self, tmp_path):
        """Pure VLM pipeline with --dry-run completes without error."""
        result = _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            "--output",
            str(tmp_path / "output.json"),
            "--dry-run",
        )
        assert result.returncode == 0, result.stderr

    def test_output_json_created(self, tmp_path):
        """--output flag creates a JSON results file."""
        out = tmp_path / "output.json"
        _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            "--output",
            str(out),
            "--dry-run",
        )
        assert out.exists()

    def test_output_json_has_segments(self, tmp_path):
        """Output JSON contains at least one segment entry."""
        out = tmp_path / "output.json"
        _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            "--output",
            str(out),
            "--dry-run",
        )
        data = json.loads(out.read_text())
        assert data.get("total_segments", 0) > 0


@requires_workspace
@requires_gpu
@requires_gstreamer
@requires_yolo26m_onnx
@pytest.mark.e2e
class TestDetectModePipeline:
    # First run triggers a TRT engine build (several minutes) plus VLM
    # model load and inference on all segments of the sample clip.
    # Subsequent runs reuse the cached engine and are much faster.
    DETECT_TIMEOUT_SEC = 1800

    def test_detect_dry_run_exits_zero(self, tmp_path):
        """VLM+Detect pipeline with --dry-run completes without error."""
        result = _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            "--detect",
            "--output",
            str(tmp_path / "output.json"),
            "--dry-run",
            timeout=self.DETECT_TIMEOUT_SEC,
        )
        assert result.returncode == 0, result.stderr

    def test_detect_output_json_created(self, tmp_path):
        """Detect mode --output creates a JSON results file."""
        out = tmp_path / "output_detect.json"
        _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            "--detect",
            "--output",
            str(out),
            "--dry-run",
            timeout=self.DETECT_TIMEOUT_SEC,
        )
        assert out.exists()


@requires_workspace
@requires_gpu
@requires_gstreamer
@requires_compose
@pytest.mark.e2e
class TestDNARoundTripSmoke:
    """Phase-2 E2E smoke: sample.mp4 → DS pipeline → Kafka → CurationConsumer → PG + Milvus.

    Requires the full compose stack (Kafka, Postgres, Milvus) in addition to a
    GPU and GStreamer environment. Skips gracefully when any dependency is absent.
    """

    PIPELINE_TIMEOUT_SEC = 600
    CONSUMER_TIMEOUT_SEC = 60

    def _session_id(self) -> str:
        return f"e2e-smoke-{uuid.uuid4().hex[:8]}"

    def _pg_dsn(self) -> str:
        return (
            f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}"
            f"@{os.environ.get('PG_HOST', 'localhost')}"
            f":{os.environ.get('PG_PORT', '5432')}"
            f"/{os.environ.get('PG_DB', 'curation')}"
        )

    def _run_consumer(self, session_id: str) -> subprocess.CompletedProcess:
        """Run CurationConsumer subprocess with a short timeout to drain the queue."""
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "src.bus.kafka",
                "--session-id",
                session_id,
                "--dataset",
                "e2e_smoke",
                "--subset",
                "test",
                "--dataset-version",
                "0",
                "--broker",
                _KAFKA_BROKER,
                "--timeout",
                str(self.CONSUMER_TIMEOUT_SEC * 1000),
            ],
            capture_output=True,
            text=True,
            timeout=self.CONSUMER_TIMEOUT_SEC + 10,
            cwd=str(WORKSPACE),
        )

    def test_kafka_consumer_writes_schema_valid_dna(self):
        """sample.mp4 → DS pipeline → Kafka → CurationConsumer → schema-valid DNA row in PG."""
        import jsonschema

        session_id = self._session_id()
        schema = json.loads(SCHEMA_PATH.read_text())

        # Run DS pipeline (no --dry-run: publishes to Kafka)
        pipeline_result = _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            timeout=self.PIPELINE_TIMEOUT_SEC,
        )
        assert pipeline_result.returncode == 0, pipeline_result.stderr

        # Run CurationConsumer to consume published messages and write to PG
        self._run_consumer(session_id)

        # Verify schema-valid DNA row exists in Postgres
        async def _assert_pg_row() -> None:
            import asyncpg

            conn = await asyncpg.connect(self._pg_dsn())
            try:
                rows = await conn.fetch(
                    "SELECT dna_json FROM scenario_dna"
                    " WHERE pipeline_version = $1"
                    " AND (dna_json->>'dna_version') IS NOT NULL"
                    " LIMIT 10",
                    "p2-6",
                )
                assert rows, "No schema-valid scenario_dna rows found for pipeline_version='p2-6'"
                for row in rows:
                    raw = row["dna_json"]
                    dna = raw if isinstance(raw, dict) else json.loads(raw)
                    jsonschema.validate(dna, schema)
            finally:
                await conn.close()

        asyncio.run(_assert_pg_row())

    def test_pg_clips_count_increments(self):
        """Postgres `clips` row count increases after CurationConsumer ingests scouted messages.

        PG `clips` is the system of record for ingested clips (Milvus is now
        written exclusively by EmbedderWorker / /v1/ingest, not the consumer).
        """
        session_id = self._session_id()

        async def _count() -> int:
            import asyncpg

            conn = await asyncpg.connect(self._pg_dsn())
            try:
                return await conn.fetchval("SELECT count(*) FROM clips")
            finally:
                await conn.close()

        count_before = asyncio.run(_count())

        pipeline_result = _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            timeout=self.PIPELINE_TIMEOUT_SEC,
        )
        assert pipeline_result.returncode == 0, pipeline_result.stderr
        self._run_consumer(session_id)

        count_after = asyncio.run(_count())
        assert count_after > count_before, (
            f"PG clips count did not increase: before={count_before} after={count_after}"
        )


@requires_workspace
@requires_gpu
@requires_gstreamer
@requires_compose
@pytest.mark.e2e
class TestEmbedderWorkerE2E:
    """P3-2: verify embedder-worker (compose daemon) replaces the zero-vector stub
    so DS pipeline clips become searchable via curation-api.

    Requires the full compose stack (base + curate + pipeline) up, including
    the ``embedder-worker`` service from ``infra/compose.curate.yml``.  Without
    that daemon running, DS pipeline clips remain zero-vectored and these
    tests fail intentionally.

    Class-scoped fixture runs the DS pipeline once and shares the resulting
    set of newly-embedded clip_ids with the per-test assertions.
    """

    PIPELINE_TIMEOUT_SEC = 600
    EMBED_WAIT_SEC = 180

    @pytest.fixture(scope="class")
    def new_clip_ids(self) -> set[str]:
        """Run DS pipeline + consumer once; poll Milvus until embedder-worker
        produces ≥1 new non-zero vector.  Returns the set of newly-embedded clip_ids.
        """
        import time

        import numpy as np
        from pymilvus import MilvusClient

        milvus_uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
        client = MilvusClient(uri=milvus_uri)

        def _nonzero_clip_ids() -> set[str]:
            rows = client.query(
                collection_name="clip_video_embed",
                filter="",
                output_fields=["clip_id", "embedding"],
                limit=10000,
            )
            return {
                row["clip_id"]
                for row in rows
                if row.get("embedding") and float(np.linalg.norm(row["embedding"])) > 1e-3
            }

        before = _nonzero_clip_ids()

        pipeline_result = _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            timeout=self.PIPELINE_TIMEOUT_SEC,
        )
        assert pipeline_result.returncode == 0, pipeline_result.stderr

        consumer_session = f"e2e-emb-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.bus.kafka",
                "--session-id",
                consumer_session,
                "--dataset",
                "e2e_emb",
                "--subset",
                "test",
                "--dataset-version",
                "0",
                "--broker",
                _KAFKA_BROKER,
                "--timeout",
                "60000",
            ],
            capture_output=True,
            text=True,
            timeout=70,
            cwd=str(WORKSPACE),
        )

        deadline = time.time() + self.EMBED_WAIT_SEC
        new_ids: set[str] = set()
        while time.time() < deadline:
            new_ids = _nonzero_clip_ids() - before
            if new_ids:
                break
            time.sleep(3)

        if not new_ids:
            pytest.fail(
                f"No new non-zero embeddings within {self.EMBED_WAIT_SEC}s — "
                "embedder-worker may not be running as a compose service."
            )
        return new_ids

    def test_embedder_worker_replaces_zero_vector(self, new_clip_ids: set[str]) -> None:
        """Embedder-worker overwrote ≥1 zero-vector stub with a real Cosmos-Embed1 vector."""
        assert len(new_clip_ids) >= 1, (
            f"Expected ≥1 new non-zero embedding from embedder-worker, got {len(new_clip_ids)}"
        )

    def test_ds_pipeline_clip_searchable(self, new_clip_ids: set[str]) -> None:
        """A newly-embedded DS pipeline clip is retrievable via /v1/search/video self-query.

        The query embedding is derived from the clip's own MinIO frames, so the
        clip itself must rank in the top-N (cosine self-similarity ≈ 1.0).
        """
        import httpx

        target_id = next(iter(new_clip_ids))
        api_url = os.environ.get("CURATION_API_URL", "http://localhost:8001")
        with httpx.Client(base_url=api_url, timeout=30.0) as http:
            resp = http.post(
                "/v1/search/video",
                json={"clip_id": target_id, "limit": 5},
            )
            assert resp.status_code == 200, resp.text
            result_ids = [r["clip_id"] for r in resp.json()["results"]]

        assert target_id in result_ids, (
            f"Clip {target_id} not found in /v1/search/video self-query results: {result_ids}"
        )
