"""Shared fixtures and markers for DeepStream-VLM tests."""

import os
import shutil
import subprocess
import sys

import pytest

# ── Project root & path setup (must run at import time, before test collection) ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _subdir in ("plugin", "src"):
    _p = os.path.join(PROJECT_ROOT, _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Environment detection ──
def _has_nvidia_gpu() -> bool:
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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _has_gstreamer() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return True
    except Exception:
        return False


GPU_AVAILABLE = _has_nvidia_gpu()
GST_AVAILABLE = _has_gstreamer()

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="No NVIDIA GPU detected")
requires_gstreamer = pytest.mark.skipif(not GST_AVAILABLE, reason="GStreamer not available")


# ── YAML config fixtures ──
@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a minimal valid YAML config for testing."""
    content = """\
model:
  path: "/workspace/models/hub/test-model"
  max_model_len: 4096
  gpu_memory_utilization: 0.5
  gpu_id: 1
  video_mode: 1
  tensor_format: "pytorch"
segment:
  length_sec: 5
  overlap_sec: 1
  selection_fps: 2
inference:
  max_tokens: 512
  temperature: 0.3
  top_p: 0.85
  top_k: 40
  repetition_penalty: 1.2
  system_prompt: "You are a test assistant."
  user_prompt: "Describe {num_frames} frames from stream {stream_id}."
  stream_prompts:
    0:
      user_prompt: "Stream 0 override prompt."
      temperature: 0.1
pipeline:
  queue_maxsize: 4
  max_wait_timeout: 60
video:
  default_fps_numerator: 25
  default_fps_denominator: 1
detection_hints:
  enabled: true
  min_confidence: 0.5
"""
    p = tmp_path / "test_config.yaml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def empty_config_yaml(tmp_path):
    """Config with no sections — all defaults."""
    p = tmp_path / "empty_config.yaml"
    p.write_text("{}\n")
    return str(p)


@pytest.fixture
def config_no_hints(tmp_path):
    """Config without detection_hints section."""
    content = """\
model:
  path: "/workspace/models/hub/test-model"
inference:
  user_prompt: "Describe the scene."
"""
    p = tmp_path / "config_no_hints.yaml"
    p.write_text(content)
    return str(p)
