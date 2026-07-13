"""N-sample majority-vote override rule + safety_event consistency flag (P4-6)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from my_curator.domain.judge.verdict import KEEP, Verdict, effective_risk


@dataclass(frozen=True)
class JudgeDecision:
    """Aggregated decision over N critic samples for one clip."""

    scout_risk: str
    final_risk: str
    flipped: bool
    votes: dict[str, int]
    agreement: int
    n: int
    rationale: str | None
    scene_override: str | None
    confidence: str | None


def _majority_threshold(n: int) -> int:
    return n // 2 + 1


def decide(
    scout_risk: str,
    verdicts: Sequence[Verdict],
    *,
    majority: int | None = None,
) -> JudgeDecision:
    """Flip risk_level only when the modal effective risk differs from ``scout_risk`` AND
    reaches the majority threshold; otherwise KEEP. No verdicts → a KEEP decision."""
    n = len(verdicts)
    if n == 0:
        return JudgeDecision(
            scout_risk=scout_risk,
            final_risk=scout_risk,
            flipped=False,
            votes={},
            agreement=0,
            n=0,
            rationale=None,
            scene_override=None,
            confidence=None,
        )

    threshold = majority if majority is not None else _majority_threshold(n)
    effective = [effective_risk(v, scout_risk) for v in verdicts]
    tally = Counter(effective)
    modal, agreement = tally.most_common(1)[0]

    flipped = modal != scout_risk and agreement >= threshold
    final_risk = modal if flipped else scout_risk

    rationale: str | None = None
    if flipped:
        for v in verdicts:
            if effective_risk(v, scout_risk) == modal and v.rationale:
                rationale = v.rationale
                break

    scene_props = [v.scene for v in verdicts if v.scene not in (None, KEEP)]
    scene_override = scene_props[0] if len(scene_props) >= threshold else None

    conf_counts = Counter(v.confidence for v in verdicts if v.confidence)
    confidence = conf_counts.most_common(1)[0][0] if conf_counts else None

    return JudgeDecision(
        scout_risk=scout_risk,
        final_risk=final_risk,
        flipped=flipped,
        votes=dict(tally),
        agreement=agreement,
        n=n,
        rationale=rationale,
        scene_override=scene_override,
        confidence=confidence,
    )


def safety_event_inconsistency(dna: dict[str, Any]) -> str | None:
    """Flag ``event_type=collision`` with ``collision_type=null`` for review (schema-legal,
    a positively-indeterminate collision — never mutated)."""
    if not isinstance(dna, dict):
        return None
    se = (
        dna.get("planner_logic", {}).get("safety_event")
        if isinstance(dna.get("planner_logic"), dict)
        else None
    )
    if not isinstance(se, dict):
        return None
    if se.get("event_type") == "collision" and se.get("collision_type") is None:
        return "event_type=collision with collision_type=null (indeterminate collision type)"
    return None
