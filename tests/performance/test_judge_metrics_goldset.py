"""Report-only Judge CAR/FOR/pass-through gate on the gold set (P4-6).

Consumes a predictions file produced by ``scripts/gen_judge_predictions.py`` (which runs
the Judge over the gold clips' v0.2 DNA). The file is a JSON list of
``{"clip_id", "scout", "final", "gt"}`` records. Skips when the file is absent (the
authentic file requires the gold corpus to be re-curated to v0.2 first — see the script).

Initial gates are RELAXED / report-only: the test logs each gate's pass/fail and only
hard-asserts that the metrics are computable, never failing the build on a gate miss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from my_curator.domain.judge.metrics import JudgeRecord, compute_metrics

pytestmark = pytest.mark.performance

log = logging.getLogger(__name__)

_PREDICTIONS = Path(__file__).parent / "judge_predictions_goldset.json"

# Initial report-only gates (§ P4-6 issue). Logged, not enforced.
GATE_CAR_MIN = 0.60
GATE_FOR_MAX = 0.15
GATE_NOMINAL_PASSTHROUGH_MIN = 0.80


def _load_records() -> list[JudgeRecord]:
    if not _PREDICTIONS.exists():
        pytest.skip(
            f"no predictions file at {_PREDICTIONS.name} — run scripts/gen_judge_predictions.py "
            "after the gold corpus is re-curated to v0.2"
        )
    data = json.loads(_PREDICTIONS.read_text(encoding="utf-8"))
    return [JudgeRecord(scout=r["scout"], final=r["final"], gt=r.get("gt")) for r in data]


def test_judge_gates_report_only():
    records = _load_records()
    assert records, "predictions file is empty"
    m = compute_metrics(records)

    log.info("Judge gold-set metrics: %s", m)
    # Report-only: log whether each relaxed gate is met; do NOT fail the build.
    for name, value, ok in (
        ("CAR", m["car"], m["car"] is None or m["car"] >= GATE_CAR_MIN),
        ("FOR", m["for"], m["for"] is None or m["for"] <= GATE_FOR_MAX),
        (
            "nominal_passthrough",
            m["nominal_passthrough_rate"],
            m["nominal_passthrough_rate"] is None
            or m["nominal_passthrough_rate"] >= GATE_NOMINAL_PASSTHROUGH_MIN,
        ),
    ):
        log.info("gate %-20s value=%s met=%s", name, value, ok)

    # Hard assertions only on computability (report-only phase).
    assert m["n"] == len(records)
    assert "car" in m and "for" in m and "nominal_passthrough_rate" in m
