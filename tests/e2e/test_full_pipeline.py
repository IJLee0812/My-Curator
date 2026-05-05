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

    def test_milvus_entity_count_increments(self):
        """Milvus clip_video_embed entity count increases by at least 1 after pipeline ingest."""
        session_id = self._session_id()

        async def _count_before() -> int:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=os.environ.get("MILVUS_URI", "http://localhost:19530"))
            result = client.query(
                collection_name="clip_video_embed",
                filter="",
                output_fields=["count(*)"],
            )
            return result[0]["count(*)"] if result else 0

        count_before = asyncio.run(_count_before())

        # Run DS pipeline then consumer
        pipeline_result = _run_pipeline(
            str(SAMPLE_MP4),
            "-c",
            str(CONFIG_DRIVING),
            timeout=self.PIPELINE_TIMEOUT_SEC,
        )
        assert pipeline_result.returncode == 0, pipeline_result.stderr
        self._run_consumer(session_id)

        async def _count_after() -> int:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=os.environ.get("MILVUS_URI", "http://localhost:19530"))
            result = client.query(
                collection_name="clip_video_embed",
                filter="",
                output_fields=["count(*)"],
            )
            return result[0]["count(*)"] if result else 0

        count_after = asyncio.run(_count_after())
        assert count_after > count_before, (
            f"Milvus entity count did not increase: before={count_before} after={count_after}"
        )
