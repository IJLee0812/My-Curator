"""CLI entrypoint for the offline Judge-critic pass (P4-6).

Scope (validation-first default is --gold-set):
  --gold-set [PATH]   judge the P4-3 gold set (measures CAR/FOR against gold labels)
  --session SESSION   judge all v0.2 clips of one ingest session (production default)
  --all-v0.2          judge every v0.2 DNA row (backfill / prompt-version re-run)

The judge-critic vLLM server must be running on GPU 0 (compose ``--profile judge``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from my_curator.adapters.judge.qwen_text_critic import QwenTextCritic, base_url_from_env
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.application.pipeline.judge_pass import DEFAULT_N_SAMPLES, judge_pass
from my_curator.domain.judge.prompt import (
    assert_judge_prompt_registered,
    judge_prompt_hash,
    load_system_prompt,
)

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GOLD = str(_REPO_ROOT / "tests" / "performance" / "gold_set_v2.json")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline LLM-as-Judge text critic pass (Qwen3-8B-AWQ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = p.add_mutually_exclusive_group()
    scope.add_argument(
        "--gold-set",
        nargs="?",
        const=DEFAULT_GOLD,
        metavar="PATH",
        help=f"Judge the gold set (default path: {DEFAULT_GOLD})",
    )
    scope.add_argument("--session", metavar="SESSION_ID", help="Judge one session's v0.2 clips")
    scope.add_argument(
        "--all-v0.2", dest="all_v02", action="store_true", help="Judge every v0.2 DNA row"
    )
    p.add_argument("--judge-url", default=base_url_from_env(), help="judge-critic base URL")
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="self-consistency votes"
    )
    p.add_argument("--limit", type=int, default=1000, help="max clips to judge")
    p.add_argument("--dry-run", action="store_true", help="compute + report, write nothing")
    return p


def load_gold_gt(path: str) -> dict[str, str]:
    """Return ``{clip_id: gold risk_level}`` from a gold_set_v2.json file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c["clip_id"]: c["risk_level"] for c in data.get("clips", [])}


def attach_gt(rows: list[dict], gt_map: dict[str, str]) -> list[dict]:
    """Attach a ``gt`` (gold risk_level) to each row, matched by str(clip_id)."""
    for r in rows:
        r["gt"] = gt_map.get(str(r["clip_id"]))
    return rows


async def _select_records(repo: Any, args: argparse.Namespace) -> list[dict]:
    """Resolve the CLI scope into judge_pass records (default: gold set)."""
    if args.session:
        return await repo.list_v02_dna(session_id=args.session, limit=args.limit)
    if args.all_v02:
        return await repo.list_v02_dna(limit=args.limit)
    gold_path = args.gold_set or DEFAULT_GOLD
    gt_map = load_gold_gt(gold_path)
    from uuid import UUID

    clip_ids = [UUID(cid) for cid in gt_map]
    rows = await repo.list_v02_dna(clip_ids=clip_ids, limit=args.limit)
    return attach_gt(rows, gt_map)


async def _run(args: argparse.Namespace) -> None:
    prompt_hash = judge_prompt_hash()
    assert_judge_prompt_registered(prompt_hash)
    system_prompt = load_system_prompt()

    repo = await PGRepository.create(dsn_from_env())
    critic = QwenTextCritic(args.judge_url)
    try:
        records = await _select_records(repo, args)
        if not records:
            log.warning("No v0.2 DNA rows matched the selected scope — nothing to judge.")
            return
        log.info("Judging %d clip(s), N=%d, dry_run=%s", len(records), args.n_samples, args.dry_run)
        result = await judge_pass(
            repo=repo,
            critic=critic,
            system_prompt=system_prompt,
            prompt_hash=prompt_hash,
            records=records,
            n_samples=args.n_samples,
            dry_run=args.dry_run,
        )
    finally:
        await critic.aclose()
        await repo.close()

    flips = sum(1 for j in result["judgements"] if j.flipped)
    scenes = sum(1 for j in result["judgements"] if j.scene_overridden)
    flags = sum(1 for j in result["judgements"] if j.safety_flag)
    m = result["metrics"]
    log.info(
        "Done — clips=%d flips=%d scene_overrides=%d safety_flags=%d overrides_written=%d",
        m["n"],
        flips,
        scenes,
        flags,
        result["overrides_written"],
    )
    log.info(
        "Gates (report-only) — CAR=%s FOR=%s nominal_passthrough=%s",
        m["car"],
        m["for"],
        m["nominal_passthrough_rate"],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run(_build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
