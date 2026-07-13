"""Offline Judge-critic pass orchestration (P4-6).

For each stored v0.2 DNA record: build the critic prompt, fire N self-consistency
samples **in parallel**, majority-vote the risk_level, apply any override to the DNA
(risk_level + synced rationale, and report-only scene_description), log every override
to ``judge_overrides``, flag safety_event inconsistencies (read-only), and stamp
``provenance.judge_model`` / ``judge_prompt_hash``. Never blocks inline publish.

Failed/timed-out samples are dropped; if every sample fails the clip falls back to
Scout-only (KEEP) since ``decide([])`` returns a pass-through decision.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import UUID

from my_curator.domain.judge.decision import decide, safety_event_inconsistency
from my_curator.domain.judge.metrics import JudgeRecord, compute_metrics
from my_curator.domain.judge.prompt import build_judge_user_prompt
from my_curator.domain.judge.verdict import Verdict, parse_verdict

DEFAULT_MODEL_NAME = "qwen3-8b-awq"
DEFAULT_N_SAMPLES = 3


@dataclass
class ClipJudgement:
    """Per-clip outcome of the pass."""

    clip_id: Any
    scout_risk: str | None
    final_risk: str | None
    flipped: bool = False
    scene_overridden: bool = False
    safety_flag: str | None = None
    n_ok: int = 0
    confidence: str | None = None
    fields_written: list[str] = field(default_factory=list)


async def _sample_verdicts(
    critic: Any, system_prompt: str, user_prompt: str, n: int
) -> list[Verdict]:
    """Fire N critique calls concurrently; return parsed verdicts (failures dropped)."""
    results = await asyncio.gather(
        *(critic.critique(system_prompt, user_prompt) for _ in range(n)),
        return_exceptions=True,
    )
    return [parse_verdict(r) for r in results if not isinstance(r, BaseException)]


def _s(value: Any) -> str | None:
    """Coerce an override value to text for the audit log (None stays None)."""
    return None if value is None else str(value)


async def judge_pass(
    *,
    repo: Any,
    critic: Any,
    system_prompt: str,
    prompt_hash: str,
    records: Sequence[dict],
    model_name: str = DEFAULT_MODEL_NAME,
    n_samples: int = DEFAULT_N_SAMPLES,
    dry_run: bool = False,
) -> dict:
    """Run the Judge over ``records`` (``[{clip_id, dna_json, gt?}]``; optional ``gt`` is
    the gold risk_level, used only for metrics).

    Fires ``n_samples`` critiques per clip, majority-votes risk_level, and (unless
    ``dry_run``) writes overrides + provenance via ``repo``. Returns
    ``{"judgements": [ClipJudgement...], "metrics": {...}, "overrides_written": int, "dry_run": bool}``.
    """
    judgements: list[ClipJudgement] = []
    metric_records: list[JudgeRecord] = []
    overrides_written = 0

    for rec in records:
        clip_id = rec["clip_id"]
        dna = copy.deepcopy(rec["dna_json"])
        pl = dna.get("planner_logic") if isinstance(dna.get("planner_logic"), dict) else {}
        scout_risk = pl.get("risk_level")
        gt = rec.get("gt")

        user_prompt = build_judge_user_prompt(dna)
        verdicts = await _sample_verdicts(critic, system_prompt, user_prompt, n_samples)
        decision = decide(scout_risk, verdicts) if scout_risk is not None else None

        jm = ClipJudgement(
            clip_id=clip_id,
            scout_risk=scout_risk,
            final_risk=(decision.final_risk if decision else scout_risk),
            n_ok=len(verdicts),
            confidence=(decision.confidence if decision else None),
        )

        pending: list[tuple[str, Any, Any, Any]] = []  # (field, scout_value, judge_value, gt_value)

        if decision and decision.flipped:
            jm.flipped = True
            jm.final_risk = decision.final_risk
            pending.append(("risk_level", scout_risk, decision.final_risk, gt))
            if not dry_run:
                dna["planner_logic"]["risk_level"] = decision.final_risk
                if decision.rationale:
                    dna["planner_logic"]["risk_level_rationale"] = decision.rationale
            jm.fields_written.append("risk_level")

        if decision and decision.scene_override:
            jm.scene_overridden = True
            old_scene = dna.get("scene_description")
            pending.append(("scene_description", old_scene, decision.scene_override, None))
            if not dry_run:
                dna["scene_description"] = decision.scene_override
            jm.fields_written.append("scene_description")

        flag = safety_event_inconsistency(dna)
        if flag:
            jm.safety_flag = flag
            pending.append(("safety_event_consistency", None, flag, None))

        if not dry_run:
            prov = dna.setdefault("provenance", {})
            prov["judge_model"] = model_name
            prov["judge_prompt_hash"] = prompt_hash
            await repo.apply_judge_override_dna(
                clip_id=clip_id, dna_json=dna, judge_prompt_hash=prompt_hash
            )
            for fld, sv, jv, gtv in pending:
                await repo.insert_judge_override(
                    clip_id=clip_id,
                    field=fld,
                    scout_value=_s(sv),
                    judge_value=_s(jv),
                    gt_value=_s(gtv),
                )
                overrides_written += 1

        judgements.append(jm)
        if scout_risk is not None:
            metric_records.append(JudgeRecord(scout=scout_risk, final=jm.final_risk, gt=gt))

    return {
        "judgements": judgements,
        "metrics": compute_metrics(metric_records),
        "overrides_written": overrides_written,
        "dry_run": dry_run,
    }


def clip_id_as_uuid(value: Any) -> UUID:
    """Best-effort UUID coercion for clip_id values coming from PG or JSON."""
    return value if isinstance(value, UUID) else UUID(str(value))
