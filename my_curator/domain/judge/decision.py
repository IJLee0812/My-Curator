"""N-sample majority-vote override rule + safety_event consistency flag (P4-6).

Conservatism is structural: KEEP is the default, and ``risk_level`` flips only when a
majority of the N self-consistency samples agree on a label differing from the Scout's.
The self-reported ``CONFIDENCE`` tag is carried through for audit logging only — never a gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from my_curator.domain.judge.verdict import KEEP, Verdict, effective_risk


@dataclass(frozen=True)
class JudgeDecision:
    """Aggregated decision over N critic samples for one clip.

    ``final_risk`` == ``scout_risk`` unless ``flipped``. ``rationale`` is present only when
    flipped; ``scene_override`` and ``confidence`` are report-only.
    """

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
    """Votes required for a majority: ``n // 2 + 1`` (2 for N=3, 3 for N=5)."""
    return n // 2 + 1


def decide(
    scout_risk: str,
    verdicts: Sequence[Verdict],
    *,
    majority: int | None = None,
) -> JudgeDecision:
    """Aggregate N critic samples into one override decision.

    Flip ``risk_level`` only when the modal effective risk differs from ``scout_risk`` AND
    reaches the majority threshold; otherwise KEEP. ``scene_description`` is report-only.
    With no verdicts a KEEP decision is returned.
    """
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
    """Read-only flag for ``event_type=collision`` with ``collision_type=null``.

    The v0.2 schema permits this (a positively-indeterminate collision: off-screen /
    occluded / partial frame), so it is surfaced for human review, not treated as an error.
    """
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
