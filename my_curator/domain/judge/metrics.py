"""Judge quality metrics vs gold — CAR / FOR / nominal-pass-through (P4-6).

CAR: of overridden clips with a gold label, fraction where the Judge matches gold.
FOR: of clips the Scout already got right, fraction the Judge needlessly flipped.
nominal pass-through: of Scout-``nominal`` clips, fraction left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class JudgeRecord:
    """One clip's before/after/gold labels. ``gt`` is None when no gold exists."""

    scout: str
    final: str
    gt: str | None = None


def _rate(numer: int, denom: int) -> float | None:
    """numer/denom, or None when the denominator is zero (undefined, not 0.0)."""
    return (numer / denom) if denom else None


def compute_metrics(records: Iterable[JudgeRecord | dict]) -> dict:
    """Compute CAR / FOR / nominal-pass-through over a set of judged clips."""
    recs = [r if isinstance(r, JudgeRecord) else JudgeRecord(**r) for r in records]

    overrides = car_matches = 0
    scout_correct = false_overrides = 0
    scout_nominal = nominal_passthrough = 0

    for r in recs:
        flipped = r.final != r.scout
        if flipped and r.gt is not None:
            overrides += 1
            if r.final == r.gt:
                car_matches += 1
        if r.gt is not None and r.scout == r.gt:
            scout_correct += 1
            if flipped:
                false_overrides += 1
        if r.scout == "nominal":
            scout_nominal += 1
            if not flipped:
                nominal_passthrough += 1

    return {
        "n": len(recs),
        "overrides": overrides,
        "car_matches": car_matches,
        "car": _rate(car_matches, overrides),
        "scout_correct": scout_correct,
        "false_overrides": false_overrides,
        "for": _rate(false_overrides, scout_correct),
        "scout_nominal": scout_nominal,
        "nominal_passthrough": nominal_passthrough,
        "nominal_passthrough_rate": _rate(nominal_passthrough, scout_nominal),
    }
