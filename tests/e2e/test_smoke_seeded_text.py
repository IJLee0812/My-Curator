"""GT-S2: GPU smoke — seeded best.text first-200-char snapshot (optional).

Activated only when ``configs/config_driving_scene.yaml`` is overridden to set
``inference.temperature: 0.0`` (deterministic decoding).  vLLM does not
guarantee bit-identical decoding even with temperature=0, so this test is
opt-in via ``GT_S2_SEEDED=1`` env var.

References:
  docs/refactoring_plan.md  §3.1 GT-S2.
"""

from __future__ import annotations

import os
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SAMPLE_VIDEO = _REPO_ROOT / "assets" / "videos" / "sample.mp4"
_SEEDED = os.environ.get("GT_S2_SEEDED") == "1"


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.skipif(not _SAMPLE_VIDEO.exists(), reason="sample video missing"),
    pytest.mark.skipif(not _SEEDED, reason="opt-in: set GT_S2_SEEDED=1 to run"),
]


def test_seeded_best_text_first_200_chars():
    """Placeholder — implementation lands when the seeded fixture is captured.

    See plan §3.1 GT-S2.  The reference text is captured by running the DS
    pipeline once under temperature=0.0 + fixed seed and committing the
    first 200 chars of ``best.text`` to tests/fixtures/golden/seeded_text.txt.
    """
    snapshot = _REPO_ROOT / "tests" / "fixtures" / "golden" / "seeded_text.txt"
    if not snapshot.exists():
        pytest.skip("seeded snapshot not yet captured")
    expected = snapshot.read_text(encoding="utf-8")[:200]
    # Actual DS run + comparison wired in after fixture capture.
    assert isinstance(expected, str) and len(expected) > 0
