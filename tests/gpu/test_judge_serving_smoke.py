"""GPU smoke test: judge-critic serving stability + verdict-format parse (P4-6).

Formalises the 10-scenario stability check. Requires a running judge-critic
(``docker compose ... --profile judge up -d judge-critic``); auto-skips when the
server at ``JUDGE_CRITIC_URL`` (default http://localhost:8002) is unreachable.

Run: ``pytest tests/gpu/test_judge_serving_smoke.py -m gpu``.
"""

from __future__ import annotations

import urllib.request

import pytest

from my_curator.adapters.judge.qwen_text_critic import QwenTextCritic, base_url_from_env
from my_curator.domain.judge.prompt import build_judge_user_prompt, load_system_prompt
from my_curator.domain.judge.verdict import CONFIDENCE_LEVELS, RISK_LEVELS, parse_verdict

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_KEEP = "KEEP"


def _server_up() -> bool:
    try:
        urllib.request.urlopen(f"{base_url_from_env()}/health", timeout=2)
        return True
    except Exception:
        return False


def _dna(risk, scene, event="none", collision=None, severity=None):
    return {
        "scene_description": scene,
        "planner_logic": {
            "risk_level": risk,
            "risk_level_rationale": "r",
            "safety_event": {
                "has_event": event != "none",
                "event_type": event,
                "collision_type": collision,
                "severity_estimate": severity,
            },
        },
    }


# A compact diverse + adversarial battery (correct / clearly-wrong / borderline).
SCENARIOS = [
    _dna("nominal", "Clear-day motorway cruise, no actors within 50 m."),
    _dna(
        "critical",
        "Ego's bumper contacts the lead's rear.",
        event="collision",
        collision="rear_end",
        severity="minor",
    ),
    _dna("nominal", "Ego collides with the stopped vehicle ahead in the rain.", event="none"),
    _dna(
        "elevated",
        "Cyclist briefly enters the ego lane edge in light rain; no contact.",
        event="near_miss",
        severity="no_harm",
    ),
]


@pytest.mark.skipif(not _server_up(), reason="judge-critic server not reachable")
async def test_serving_returns_parseable_verdicts():
    system = load_system_prompt()
    critic = QwenTextCritic(timeout_s=30.0, retries=1)
    parsed_ok = 0
    try:
        for dna in SCENARIOS:
            raw = await critic.critique(system, build_judge_user_prompt(dna))
            v = parse_verdict(raw)
            # verdict block is present and well-formed
            assert v.risk in (*RISK_LEVELS, _KEEP), f"unparseable risk: {raw[-200:]!r}"
            if v.confidence is not None:
                assert v.confidence in CONFIDENCE_LEVELS
            parsed_ok += 1
    finally:
        await critic.aclose()
    assert parsed_ok == len(SCENARIOS)
