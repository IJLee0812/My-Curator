"""Corpus-level coverage report over mapping results.

Answers three questions: what fraction of the corpus can be re-staged at all and why the
rest cannot; how much of what is stageable is faithful versus a recorded compromise; and
whether the prioritized elevated/critical segments fare differently from the nominal bulk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from my_curator.domain.sim.mapper import MappingResult
from my_curator.domain.sim.spec import ControlMode, SimSpec

#: Report order: the segments the phase cares about first.
RISK_ORDER: tuple[str, ...] = ("critical", "elevated", "nominal", "unknown")

#: How much of a staged scene is actually *happening*, worst to best. Coverage alone
#: overstates demo value: a segment can map perfectly and still render as an empty road.
SCENE_CONTENT_ORDER: tuple[str, ...] = (
    "ego_only",
    "ambient_only",
    "scripted_actors",
    "ego_interaction",
)


def classify_scene_content(spec: SimSpec) -> str:
    """Bucket a spec by how much staged activity it contains."""
    if not spec.actors:
        return "ego_only"
    if any(a.control_mode is ControlMode.EVENT for a in spec.actors):
        return "ego_interaction"
    if any(a.control_mode is ControlMode.SCRIPTED for a in spec.actors):
        return "scripted_actors"
    return "ambient_only"


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


@dataclass(frozen=True)
class RiskBreakdown:
    risk_level: str
    total: int
    mapped: int
    degraded: int

    @property
    def excluded(self) -> int:
        return self.total - self.mapped

    @property
    def coverage_pct(self) -> float:
        return _pct(self.mapped, self.total)

    @property
    def clean_pct(self) -> float:
        return _pct(self.mapped - self.degraded, self.mapped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "total": self.total,
            "mapped": self.mapped,
            "excluded": self.excluded,
            "degraded": self.degraded,
            "coverage_pct": self.coverage_pct,
            "clean_pct": self.clean_pct,
        }


@dataclass(frozen=True)
class CoverageReport:
    total: int
    mapped: int
    degraded: int
    by_risk: tuple[RiskBreakdown, ...]
    exclusion_counts: dict[str, int]
    exclusion_examples: dict[str, str]
    degradation_counts: dict[str, int]
    degradation_notes: dict[str, str]
    town_candidate_counts: dict[str, int]
    scene_content_counts: dict[str, int] = field(default_factory=dict)
    anomalies: dict[str, int] = field(default_factory=dict)

    @property
    def excluded(self) -> int:
        return self.total - self.mapped

    @property
    def coverage_pct(self) -> float:
        return _pct(self.mapped, self.total)

    @property
    def clean_pct(self) -> float:
        return _pct(self.mapped - self.degraded, self.mapped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_segments": self.total,
            "mapped": self.mapped,
            "excluded": self.excluded,
            "coverage_pct": self.coverage_pct,
            "degraded": self.degraded,
            "clean_pct": self.clean_pct,
            "by_risk": [b.to_dict() for b in self.by_risk],
            "exclusions": [
                {"reason": r, "count": c, "example": self.exclusion_examples.get(r, "")}
                for r, c in sorted(self.exclusion_counts.items(), key=lambda kv: -kv[1])
            ],
            "degradations": [
                {"code": c, "count": n, "note": self.degradation_notes.get(c, "")}
                for c, n in sorted(self.degradation_counts.items(), key=lambda kv: -kv[1])
            ],
            "town_candidates": dict(
                sorted(self.town_candidate_counts.items(), key=lambda kv: -kv[1])
            ),
            "scene_content": {
                bucket: self.scene_content_counts.get(bucket, 0)
                for bucket in reversed(SCENE_CONTENT_ORDER)
            },
            "schema_anomalies": dict(sorted(self.anomalies.items(), key=lambda kv: -kv[1])),
        }

    def render_text(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [
            f"segments            {self.total}",
            f"mappable            {self.mapped} ({self.coverage_pct}%)",
            f"excluded            {self.excluded}",
            f"of mappable, clean  {self.mapped - self.degraded} ({self.clean_pct}%)",
            f"of mappable, degraded {self.degraded}",
            "",
            f"{'risk':10s} {'total':>6s} {'mapped':>7s} {'cover':>7s} {'degraded':>9s}",
        ]
        for b in self.by_risk:
            lines.append(
                f"{b.risk_level:10s} {b.total:6d} {b.mapped:7d} "
                f"{b.coverage_pct:6.1f}% {b.degraded:9d}"
            )
        if self.scene_content_counts:
            lines += ["", "staged scene content (mapped segments)"]
            for bucket in reversed(SCENE_CONTENT_ORDER):
                count = self.scene_content_counts.get(bucket, 0)
                lines.append(f"  {count:4d}  {bucket}")
        if self.exclusion_counts:
            lines += ["", "exclusion reasons"]
            for reason, count in sorted(self.exclusion_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {count:4d}  {reason}")
                lines.append(f"        e.g. {self.exclusion_examples.get(reason, '')}")
        if self.degradation_counts:
            lines += ["", "degradations (segments affected)"]
            for code, count in sorted(self.degradation_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {count:4d}  {code}")
        if self.anomalies:
            lines += ["", "out-of-schema values carried through"]
            for anomaly, count in sorted(self.anomalies.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {count:4d}  {anomaly}")
        return "\n".join(lines)


def build_coverage_report(results: list[MappingResult]) -> CoverageReport:
    """Aggregate per-segment mapping results into the corpus report."""
    totals: Counter[str] = Counter()
    mapped_by_risk: Counter[str] = Counter()
    degraded_by_risk: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    exclusion_examples: dict[str, str] = {}
    degradation_counts: Counter[str] = Counter()
    degradation_notes: dict[str, str] = {}
    town_counts: Counter[str] = Counter()
    scene_content: Counter[str] = Counter()
    anomalies: Counter[str] = Counter()

    for result in results:
        risk = result.spec.risk_level if result.spec else "unknown"
        totals[risk] += 1
        if result.spec is not None:
            mapped_by_risk[risk] += 1
            if result.spec.is_degraded:
                degraded_by_risk[risk] += 1
            town_counts["+".join(result.spec.world.road.candidate_towns) or "<none>"] += 1
            scene_content[classify_scene_content(result.spec)] += 1
        for reason, detail in result.exclusions:
            exclusion_counts[reason.value] += 1
            exclusion_examples.setdefault(reason.value, detail)
        # A code is counted once per segment, not once per occurrence, so the number
        # reads as "how many segments are compromised this way".
        for code in {d.code for d in result.degradations}:
            degradation_counts[code.value] += 1
            degradation_notes.setdefault(code.value, code.note)
        for anomaly in result.anomalies:
            anomalies[anomaly] += 1

    seen_risks = [r for r in RISK_ORDER if r in totals]
    seen_risks += sorted(r for r in totals if r not in RISK_ORDER)
    by_risk = tuple(
        RiskBreakdown(risk, totals[risk], mapped_by_risk[risk], degraded_by_risk[risk])
        for risk in seen_risks
    )

    return CoverageReport(
        total=sum(totals.values()),
        mapped=sum(mapped_by_risk.values()),
        degraded=sum(degraded_by_risk.values()),
        by_risk=by_risk,
        exclusion_counts=dict(exclusion_counts),
        exclusion_examples=exclusion_examples,
        degradation_counts=dict(degradation_counts),
        degradation_notes=degradation_notes,
        town_candidate_counts=dict(town_counts),
        scene_content_counts=dict(scene_content),
        anomalies=dict(anomalies),
    )
