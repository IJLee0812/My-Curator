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

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path("/workspace")
MAIN_PY = WORKSPACE / "main.py"
SAMPLE_MP4 = WORKSPACE / "assets" / "videos" / "sample.mp4"
# Unified config — same file drives both Pure VLM and VLM+Detect modes.
CONFIG_DRIVING = WORKSPACE / "configs" / "config_driving_scene.yaml"
# Detect mode requires the YOLO26 ONNX model to already be present.
YOLO26_ONNX = WORKSPACE / "models" / "yolo26m.onnx"


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


def _run_pipeline(*extra_args, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(MAIN_PY)] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@requires_workspace
@requires_gpu
@requires_gstreamer
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
