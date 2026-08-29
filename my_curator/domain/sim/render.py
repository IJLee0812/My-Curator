"""Which segment gets rendered, and what the batch did.

A source clip holds about three segments while a render produces one set of videos, so
something has to choose. The rule is the phase decision: the highest-risk segment, ties
broken by the DNA's own confidence, overridable by index.

Pure logic — segment rows come from the caller, and nothing here touches a simulator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from my_curator.domain.sim.compilation import RISK_ORDER

RENDERED = "rendered"
FAILED = "failed"

#: Segments whose DNA carries ego interaction are the ones worth showing: most of the
#: corpus stages either scripted background behaviour or an empty road.
EGO_INTERACTION = "ego_interaction"


@dataclass(frozen=True)
class SegmentRef:
    """One candidate segment of one source clip."""

    clip_id: str
    source_clip_id: str
    segment_index: int
    risk_level: str
    confidence: float
    blob_uri: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class RenderOutcome:
    """What happened to one segment."""

    clip_id: str
    source_clip_id: str
    segment_index: int
    risk_level: str
    status: str
    town: str | None = None
    road_id: int | None = None
    lane_id: int | None = None
    duration_s: float | None = None
    keys: tuple[str, ...] = ()
    failure_reason: str | None = None

    @property
    def rendered(self) -> bool:
        return self.status == RENDERED

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source_clip_id": self.source_clip_id,
            "segment_index": self.segment_index,
            "risk_level": self.risk_level,
            "status": self.status,
            "town": self.town,
            "road_id": self.road_id,
            "lane_id": self.lane_id,
            "duration_s": self.duration_s,
            "keys": list(self.keys),
            "failure_reason": self.failure_reason,
        }


def _risk_rank(risk: str) -> int:
    return RISK_ORDER.index(risk) if risk in RISK_ORDER else len(RISK_ORDER)


def select_segment(segments: list[SegmentRef], *, index: int | None = None) -> SegmentRef | None:
    """The segment to render for one source clip.

    With *index* the choice is the operator's; without it the highest-risk segment wins,
    ties broken by confidence and then by position so the result is stable.
    """
    if not segments:
        return None
    if index is not None:
        return next((s for s in segments if s.segment_index == index), None)
    return min(segments, key=lambda s: (_risk_rank(s.risk_level), -s.confidence, s.segment_index))


def group_by_town(plans: list[tuple[SegmentRef, str]]) -> list[tuple[str, list[SegmentRef]]]:
    """Order a batch so each town is booted once.

    Map switching segfaults the simulator, so a batch runs town by town; towns are
    ordered by size to get the most renders out of the fewest server lifetimes.
    """
    grouped: dict[str, list[SegmentRef]] = {}
    for segment, town in plans:
        grouped.setdefault(town, []).append(segment)
    return sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))


@dataclass
class RenderReport:
    attempted: int = 0
    rendered: int = 0
    by_risk: list[dict[str, Any]] = field(default_factory=list)
    towns: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rendered_pct(self) -> float:
        return round(100.0 * self.rendered / self.attempted, 1) if self.attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "rendered": self.rendered,
            "rendered_pct": self.rendered_pct,
            "by_risk": self.by_risk,
            "towns": self.towns,
            "failures": self.failures,
        }

    def render_text(self) -> str:
        lines = [
            "Scenario render",
            f"  attempted : {self.attempted}",
            f"  rendered  : {self.rendered} ({self.rendered_pct}%)",
            "",
            "  by risk level:",
        ]
        for row in self.by_risk:
            lines.append(
                f"    {row['risk_level']:<9} {row['rendered']:>4}/{row['attempted']:<4}"
                f"   ({row['rendered_pct']}%)"
            )
        if self.towns:
            lines.append("")
            lines.append("  staged in:")
            for row in self.towns:
                lines.append(f"    {row['town']:<9} {row['count']:>4}")
        if self.failures:
            lines.append("")
            lines.append("  failures:")
            for row in self.failures:
                lines.append(f"    {row['reason']:<26} {row['count']:>4}")
        return "\n".join(lines)


def build_render_report(outcomes: list[RenderOutcome]) -> RenderReport:
    report = RenderReport(
        attempted=len(outcomes),
        rendered=sum(1 for o in outcomes if o.rendered),
    )
    for risk in sorted({o.risk_level for o in outcomes}, key=_risk_rank):
        rows = [o for o in outcomes if o.risk_level == risk]
        rendered = sum(1 for o in rows if o.rendered)
        report.by_risk.append(
            {
                "risk_level": risk,
                "attempted": len(rows),
                "rendered": rendered,
                "rendered_pct": round(100.0 * rendered / len(rows), 1) if rows else 0.0,
            }
        )
    report.towns = [
        {"town": town, "count": count}
        for town, count in sorted(Counter(o.town for o in outcomes if o.town).items())
    ]
    report.failures = [
        {"reason": reason, "count": count}
        for reason, count in Counter(
            o.failure_reason for o in outcomes if o.failure_reason
        ).most_common()
    ]
    return report
