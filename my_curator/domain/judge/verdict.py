"""Parse the Judge critic's raw text output into a structured Verdict (P4-6).

The critic emits a trailing label block; parsing is line-anchored (``^KEY:`` under
re.MULTILINE) so an empty field never bleeds into the next line, and the
``<think>…</think>`` chain-of-thought is stripped before parsing:

    VERDICT_RISK: <nominal|elevated|critical> | KEEP
    RATIONALE:    <one line; only when risk_level changes>
    VERDICT_SCENE: KEEP | <corrected scene_description>
    CONFIDENCE:   high | medium | low
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RISK_LEVELS: tuple[str, ...] = ("nominal", "elevated", "critical")
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")
KEEP = "KEEP"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _field(body: str, key: str) -> str | None:
    """Value of a line-anchored ``KEY: value`` line (tolerates ``**KEY:**`` markdown), or None."""
    m = re.search(rf"^[ \t>*]*{key}[ \t*]*:[ \t*]*(.+?)[ \t*]*$", body, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


@dataclass(frozen=True)
class Verdict:
    """One critic sample's parsed verdict.

    ``risk`` is a RISK_LEVELS member, the ``"KEEP"`` sentinel, or None (absent/unparseable
    → pass-through). ``confidence`` is logged only, never a gate.
    """

    risk: str | None
    rationale: str | None
    scene: str | None
    confidence: str | None
    raw: str


def strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` blocks and return the remaining text, trimmed."""
    return _THINK_RE.sub("", text or "").strip()


def parse_verdict(text: str) -> Verdict:
    """Parse a raw critic response into a :class:`Verdict` (never raises)."""
    body = strip_thinking(text)

    raw_risk = _field(body, "VERDICT_RISK")
    risk: str | None
    if raw_risk is None:
        risk = None
    elif raw_risk.strip().upper().startswith(KEEP):
        risk = KEEP
    else:
        low = raw_risk.strip().lower()
        risk = low if low in RISK_LEVELS else None

    raw_rat = _field(body, "RATIONALE")
    rationale = None if (raw_rat is None or raw_rat in {"-", "->"}) else raw_rat

    raw_scene = _field(body, "VERDICT_SCENE")
    if raw_scene is None:
        scene = None
    elif raw_scene.strip().upper().startswith(KEEP):
        scene = KEEP
    else:
        scene = raw_scene.strip().strip('"')

    raw_conf = _field(body, "CONFIDENCE")
    confidence = raw_conf.lower() if (raw_conf and raw_conf.lower() in CONFIDENCE_LEVELS) else None

    return Verdict(risk=risk, rationale=rationale, scene=scene, confidence=confidence, raw=body)


def effective_risk(verdict: Verdict, scout_risk: str) -> str:
    """Resolve a verdict to a concrete risk label.

    ``KEEP`` and an absent/unparseable verdict both resolve to the Scout's label
    (the conservative default); a stated level is returned as-is.
    """
    if verdict.risk in (None, KEEP):
        return scout_risk
    return verdict.risk
