"""GT-S1: GPU smoke — DS pipeline Kafka envelope shape.

Runs the DS pipeline on ``assets/videos/sample.mp4`` (dry-run, console output
only) and asserts the final Kafka message envelope shape matches the locked
key set + type set.  Values are non-deterministic (vLLM temperature sampling)
so per-field equality is not asserted.

Marked ``gpu`` — nightly run, skipped by default in the unit/integration loop.

References:
  docs/refactoring_plan.md  §3.1 GT-S1, §4 R-0.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from collections.abc import Iterable

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SAMPLE_VIDEO = _REPO_ROOT / "assets" / "videos" / "sample.mp4"
_DS_CONTAINER = os.environ.get("DS_CONTAINER", "my-curator-ds9-vlm-dev")


_EXPECTED_TOP_KEYS = {
    "stream_id",
    "timestamp",
    "segment",
    "result",
    "metadata",
    "clip_id",
}


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "ps"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.skipif(not _SAMPLE_VIDEO.exists(), reason="sample video missing"),
    pytest.mark.skipif(not _docker_available(), reason="docker not available"),
]


def _extract_kafka_messages(stdout: str) -> Iterable[dict]:
    """Yield JSON message objects from publisher dry-run console output."""
    capturing = False
    buf: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("Value:"):
            capturing = True
            buf = [line[len("Value:") :].strip()]
            continue
        if capturing:
            buf.append(line)
            # Heuristic: closing brace on its own line marks end of dump.
            if line.strip() == "}":
                try:
                    yield json.loads("\n".join(buf))
                except json.JSONDecodeError:
                    pass
                capturing = False
                buf = []


def test_ds_smoke_envelope_shape():
    """Run sample.mp4 through DS dry-run; assert envelope of last published message."""
    cmd = [
        "docker",
        "exec",
        _DS_CONTAINER,
        "python3",
        "/workspace/main.py",
        "/workspace/assets/videos/sample.mp4",
        "-c",
        "/workspace/configs/config_driving_scene.yaml",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"DS smoke exited {result.returncode}\nstdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-1000:]}"
    )

    messages = list(_extract_kafka_messages(result.stdout))
    assert messages, "no Kafka messages dumped by publisher"

    last = messages[-1]
    assert _EXPECTED_TOP_KEYS.issubset(last.keys()), (
        f"Envelope drift: missing {_EXPECTED_TOP_KEYS - last.keys()}; got {sorted(last.keys())}"
    )
    assert isinstance(last["segment"], dict)
    assert {"start_time", "end_time", "duration"}.issubset(last["segment"].keys())
    assert isinstance(last["metadata"], dict)
    assert "json_valid" in last["metadata"]
    assert isinstance(last["metadata"]["json_valid"], bool)
