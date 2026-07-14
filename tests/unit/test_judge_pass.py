"""Orchestration tests for judge_pass (P4-6) with a fake critic + AsyncMock PG."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from my_curator.application.pipeline.judge_pass import ClipJudgement, judge_pass

pytestmark = pytest.mark.unit

CLIP = UUID("11111111-0000-0000-0000-000000000001")
SYS = "system prompt"
HASH = "d06aef8a3365f0b2"


class FakeCritic:
    """Returns a fixed response for every critique call (or raises to simulate failure)."""

    def __init__(self, response: str | None = None, *, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.calls = 0

    async def critique(self, system: str, user: str) -> str:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._response


def _dna(risk="elevated", *, scene="A scene.", event="none", collision=None, severity=None):
    return {
        "scene_description": scene,
        "planner_logic": {
            "risk_level": risk,
            "risk_level_rationale": "orig rationale",
            "safety_event": {
                "has_event": event != "none",
                "event_type": event,
                "collision_type": collision,
                "severity_estimate": severity,
            },
        },
        "provenance": {"judge_model": None, "judge_prompt_hash": None},
    }


def _repo():
    r = AsyncMock()
    r.insert_judge_override = AsyncMock(return_value=1)
    r.apply_judge_override_dna = AsyncMock(return_value=None)
    return r


async def _run(repo, critic, records, **kw):
    return await judge_pass(
        repo=repo, critic=critic, system_prompt=SYS, prompt_hash=HASH, records=records, **kw
    )


async def test_keep_stamps_provenance_no_override_rows():
    repo = _repo()
    critic = FakeCritic("VERDICT_RISK: KEEP\nVERDICT_SCENE: KEEP\nCONFIDENCE: high")
    out = await _run(repo, critic, [{"clip_id": CLIP, "dna_json": _dna("elevated")}])
    jm = out["judgements"][0]
    assert isinstance(jm, ClipJudgement)
    assert jm.flipped is False and jm.final_risk == "elevated" and jm.n_ok == 3
    assert critic.calls == 3  # N=3 parallel samples
    repo.insert_judge_override.assert_not_awaited()  # nothing overridden
    repo.apply_judge_override_dna.assert_awaited_once()  # provenance still stamped
    written = repo.apply_judge_override_dna.call_args.kwargs["dna_json"]
    assert written["provenance"]["judge_model"] == "qwen3-8b-awq"
    assert written["provenance"]["judge_prompt_hash"] == HASH


async def test_majority_flip_writes_risk_and_logs_override():
    repo = _repo()
    critic = FakeCritic(
        "VERDICT_RISK: critical\nRATIONALE: R1: any collision = critical\nCONFIDENCE: high"
    )
    out = await _run(
        repo, critic, [{"clip_id": CLIP, "dna_json": _dna("nominal"), "gt": "critical"}]
    )
    jm = out["judgements"][0]
    assert jm.flipped is True and jm.final_risk == "critical"
    written = repo.apply_judge_override_dna.call_args.kwargs["dna_json"]
    assert written["planner_logic"]["risk_level"] == "critical"
    assert written["planner_logic"]["risk_level_rationale"] == "R1: any collision = critical"
    # override row logged for risk_level
    fields = [c.kwargs["field"] for c in repo.insert_judge_override.call_args_list]
    assert "risk_level" in fields
    assert out["metrics"]["car"] == pytest.approx(1.0)  # matched gold


async def test_scene_override_written_and_logged():
    repo = _repo()
    critic = FakeCritic('VERDICT_RISK: KEEP\nVERDICT_SCENE: "Corrected scene."\nCONFIDENCE: medium')
    out = await _run(repo, critic, [{"clip_id": CLIP, "dna_json": _dna("elevated")}])
    jm = out["judgements"][0]
    assert jm.scene_overridden is True and jm.flipped is False
    written = repo.apply_judge_override_dna.call_args.kwargs["dna_json"]
    assert written["scene_description"] == "Corrected scene."
    fields = [c.kwargs["field"] for c in repo.insert_judge_override.call_args_list]
    assert fields == ["scene_description"]


async def test_safety_event_inconsistency_flagged_not_mutated():
    repo = _repo()
    critic = FakeCritic("VERDICT_RISK: KEEP\nVERDICT_SCENE: KEEP\nCONFIDENCE: high")
    dna = _dna("critical", event="collision", collision=None, severity="minor")
    out = await _run(repo, critic, [{"clip_id": CLIP, "dna_json": dna}])
    jm = out["judgements"][0]
    assert jm.safety_flag is not None
    fields = [c.kwargs["field"] for c in repo.insert_judge_override.call_args_list]
    assert "safety_event_consistency" in fields
    written = repo.apply_judge_override_dna.call_args.kwargs["dna_json"]
    # enum NOT mutated
    assert written["planner_logic"]["safety_event"]["collision_type"] is None
    assert written["planner_logic"]["safety_event"]["event_type"] == "collision"


async def test_dry_run_writes_nothing():
    repo = _repo()
    critic = FakeCritic("VERDICT_RISK: critical\nRATIONALE: R1\nCONFIDENCE: high")
    out = await _run(repo, critic, [{"clip_id": CLIP, "dna_json": _dna("nominal")}], dry_run=True)
    assert out["dry_run"] is True
    assert out["judgements"][0].flipped is True  # decision still computed
    repo.apply_judge_override_dna.assert_not_awaited()
    repo.insert_judge_override.assert_not_awaited()
    assert out["overrides_written"] == 0


async def test_all_samples_fail_falls_back_to_keep():
    repo = _repo()
    critic = FakeCritic(raises=RuntimeError("critic down"))
    out = await _run(repo, critic, [{"clip_id": CLIP, "dna_json": _dna("elevated")}])
    jm = out["judgements"][0]
    assert jm.n_ok == 0 and jm.flipped is False and jm.final_risk == "elevated"


async def test_caller_records_not_mutated():
    repo = _repo()
    critic = FakeCritic("VERDICT_RISK: critical\nRATIONALE: R1\nCONFIDENCE: high")
    rec = {"clip_id": CLIP, "dna_json": _dna("nominal")}
    await _run(repo, critic, [rec])
    # deepcopy isolation: original record's DNA is untouched
    assert rec["dna_json"]["planner_logic"]["risk_level"] == "nominal"


async def test_metrics_aggregate_over_multiple_clips():
    repo = _repo()
    critic = FakeCritic("VERDICT_RISK: KEEP\nCONFIDENCE: high")
    recs = [
        {"clip_id": CLIP, "dna_json": _dna("nominal"), "gt": "nominal"},
        {
            "clip_id": UUID("22222222-0000-0000-0000-000000000002"),
            "dna_json": _dna("elevated"),
            "gt": "elevated",
        },
    ]
    out = await _run(repo, critic, recs)
    assert out["metrics"]["n"] == 2
    assert out["metrics"]["for"] == pytest.approx(0.0)  # nothing needlessly flipped
