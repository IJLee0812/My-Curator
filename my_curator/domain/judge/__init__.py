"""Text-only LLM-as-Judge critic domain logic (P4-6).

Pure and host-testable (no I/O/SDK at import):
  - verdict.py  — parse the critic's raw text into a structured Verdict.
  - decision.py — N-sample majority-vote override + safety_event consistency flag.
  - metrics.py  — CAR / FOR / nominal-pass-through vs gold.
  - prompt.py   — judge prompt loading, hashing, and user-prompt construction.

The Judge re-scores ONLY ``planner_logic.risk_level`` and ``scene_description``.
"""

from __future__ import annotations

from my_curator.domain.judge.decision import (
    JudgeDecision,
    decide,
    safety_event_inconsistency,
)
from my_curator.domain.judge.metrics import JudgeRecord, compute_metrics
from my_curator.domain.judge.prompt import (
    JUDGE_PROMPT_FILE,
    JUDGE_PROMPT_HASHES,
    assert_judge_prompt_registered,
    build_judge_user_prompt,
    judge_prompt_hash,
    load_system_prompt,
)
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
    "JUDGE_PROMPT_FILE",
    "JUDGE_PROMPT_HASHES",
    "RISK_LEVELS",
    "Verdict",
    "JudgeDecision",
    "JudgeRecord",
    "assert_judge_prompt_registered",
    "build_judge_user_prompt",
    "compute_metrics",
    "decide",
    "effective_risk",
    "judge_prompt_hash",
    "load_system_prompt",
    "parse_verdict",
    "safety_event_inconsistency",
    "strip_thinking",
]
