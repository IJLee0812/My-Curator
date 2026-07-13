"""Text-only LLM-as-Judge critic domain logic (P4-6).

Pure and host-testable (no I/O/SDK at import):
  - verdict.py  — parse the critic's raw text into a structured Verdict.
  - decision.py — N-sample majority-vote override + safety_event consistency flag.
  - metrics.py  — CAR / FOR / nominal-pass-through vs gold.

The Judge re-scores ONLY ``planner_logic.risk_level`` and ``scene_description``.
"""

from __future__ import annotations

from my_curator.domain.judge.decision import (
    JudgeDecision,
    decide,
    safety_event_inconsistency,
)
from my_curator.domain.judge.metrics import JudgeRecord, compute_metrics
from my_curator.domain.judge.verdict import (
    CONFIDENCE_LEVELS,
    RISK_LEVELS,
    Verdict,
    effective_risk,
    parse_verdict,
    strip_thinking,
)

__all__ = [
    "CONFIDENCE_LEVELS",
    "RISK_LEVELS",
    "Verdict",
    "JudgeDecision",
    "JudgeRecord",
    "compute_metrics",
    "decide",
    "effective_risk",
    "parse_verdict",
    "safety_event_inconsistency",
    "strip_thinking",
]
