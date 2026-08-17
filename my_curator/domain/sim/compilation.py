"""Corpus-level accounting for the compilation pass.

The coverage report answers "how much of the corpus is mappable"; this one answers "how
much of it became a valid scenario, staged where". Both exist so a headline percentage
cannot be read without the breakdown underneath it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from my_curator.domain.sim.spec import Degradation

RISK_ORDER = ("critical", "elevated", "nominal", "unknown")


@dataclass(frozen=True)
class CompiledSegment:
    """One segment's outcome, whether or not it produced a valid document."""

    clip_id: str
    risk_level: str
    town: str | None = None
    road_id: int | None = None
    lane_id: int | None = None
    is_valid: bool = False
    errors: tuple[str, ...] = ()
    road_degradations: tuple[Degradation, ...] = ()
    failure: str | None = None

    @property
    def compiled(self) -> bool:
        return self.failure is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "risk_level": self.risk_level,
            "town": self.town,
            "road_id": self.road_id,
            "lane_id": self.lane_id,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "road_degradations": [d.to_dict() for d in self.road_degradations],
            "failure": self.failure,
        }


@dataclass
class CompilationReport:
    total: int = 0
    compiled: int = 0
    valid: int = 0
    by_risk: list[dict[str, Any]] = field(default_factory=list)
    towns: list[dict[str, Any]] = field(default_factory=list)
    road_degradations: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    invalid: list[dict[str, Any]] = field(default_factory=list)

    @property
    def compiled_pct(self) -> float:
        return round(100.0 * self.compiled / self.total, 1) if self.total else 0.0

    @property
    def valid_pct(self) -> float:
        return round(100.0 * self.valid / self.total, 1) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_segments": self.total,
            "compiled": self.compiled,
            "valid": self.valid,
            "compiled_pct": self.compiled_pct,
            "valid_pct": self.valid_pct,
            "by_risk": self.by_risk,
            "towns": self.towns,
            "road_degradations": self.road_degradations,
            "failures": self.failures,
            "invalid": self.invalid,
        }

    def render_text(self) -> str:
        lines = [
            "OpenSCENARIO compilation",
            f"  segments      : {self.total}",
            f"  compiled      : {self.compiled} ({self.compiled_pct}%)",
            f"  XSD-valid     : {self.valid} ({self.valid_pct}%)",
            "",
            "  by risk level:",
        ]
        for row in self.by_risk:
            lines.append(
                f"    {row['risk_level']:<9} {row['valid']:>4}/{row['total']:<4} valid"
                f"   ({row['valid_pct']}%)"
            )
        lines.append("")
        lines.append("  staged in:")
        for row in self.towns:
            lines.append(f"    {row['town']:<9} {row['count']:>4}")
        if self.road_degradations:
            lines.append("")
            lines.append("  road-selection compromises:")
            for row in self.road_degradations:
                lines.append(f"    {row['code']:<26} {row['count']:>4}  {row['requested']}")
        if self.failures:
            lines.append("")
            lines.append("  failures:")
            for row in self.failures:
                lines.append(f"    {row['reason']:<26} {row['count']:>4}")
        if self.invalid:
            lines.append("")
            lines.append(f"  XSD-invalid: {len(self.invalid)} (first error per segment)")
            for row in self.invalid[:5]:
                lines.append(f"    {row['clip_id']}: {row['error']}")
        return "\n".join(lines)


def build_compilation_report(segments: list[CompiledSegment]) -> CompilationReport:
    report = CompilationReport(
        total=len(segments),
        compiled=sum(1 for s in segments if s.compiled),
        valid=sum(1 for s in segments if s.is_valid),
    )

    risks = sorted({s.risk_level for s in segments}, key=_risk_rank)
    for risk in risks:
        rows = [s for s in segments if s.risk_level == risk]
        valid = sum(1 for s in rows if s.is_valid)
        report.by_risk.append(
            {
                "risk_level": risk,
                "total": len(rows),
                "compiled": sum(1 for s in rows if s.compiled),
                "valid": valid,
                "valid_pct": round(100.0 * valid / len(rows), 1) if rows else 0.0,
            }
        )

    report.towns = [
        {"town": town, "count": count}
        for town, count in sorted(Counter(s.town for s in segments if s.town).items())
    ]

    degradations = Counter(
        (d.code.value, d.requested) for s in segments for d in s.road_degradations
    )
    report.road_degradations = [
        {"code": code, "requested": requested, "count": count}
        for (code, requested), count in degradations.most_common()
    ]

    report.failures = [
        {"reason": reason, "count": count}
        for reason, count in Counter(s.failure for s in segments if s.failure).most_common()
    ]

    report.invalid = [
        {"clip_id": s.clip_id, "error": s.errors[0].splitlines()[0] if s.errors else ""}
        for s in segments
        if s.compiled and not s.is_valid
    ]
    return report


def _risk_rank(risk: str) -> int:
    return RISK_ORDER.index(risk) if risk in RISK_ORDER else len(RISK_ORDER)
