"""Generate the gold-set Judge predictions file for the P4-6 performance gate.

Runs the Judge (N-sample majority vote, dry-run — no DB writes) over the gold clips'
**v0.2** DNA and dumps ``[{clip_id, scout, final, gt}]`` to
``tests/performance/judge_predictions_goldset.json``, which
``tests/performance/test_judge_metrics_goldset.py`` then scores (CAR/FOR/pass-through).

Two sources of v0.2 DNA (pick one):
  * ``--from-pipeline-output DIR``: per-clip JSON produced by a Scout-v2 pipeline dry-run
    (``run_pipeline ... --detect --dry-run --output DIR``). Segments are content-matched
    to the gold set by (source_clip_id, start_s) — no Postgres needed, corpus untouched.
  * default: read v0.2 rows for the gold clip_ids from Postgres (requires the corpus to
    have been re-curated to v0.2).

judge-critic must be running on GPU 0 (``docker compose ... --profile judge up -d judge-critic``).

Usage:
  python3 -m scripts.gen_judge_predictions --from-pipeline-output /tmp/gold_v02_out
  python3 -m scripts.gen_judge_predictions              # Postgres source
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from uuid import UUID

from my_curator.adapters.judge.qwen_text_critic import QwenTextCritic, base_url_from_env
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.application.pipeline.judge_pass import judge_pass
from my_curator.cli.run_judge_pass import DEFAULT_GOLD, attach_gt, load_gold_gt
from my_curator.domain.judge.prompt import judge_prompt_hash, load_system_prompt

log = logging.getLogger(__name__)

_OUT = Path(__file__).parent.parent / "tests" / "performance" / "judge_predictions_goldset.json"
_FENCE = re.compile(r"```json\s*(.*?)```", re.DOTALL)
_MATCH_TOL_S = 0.3  # start_s match tolerance between pipeline segment and gold segment


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate gold-set Judge predictions (P4-6)")
    p.add_argument("--gold-set", default=DEFAULT_GOLD)
    p.add_argument(
        "--from-pipeline-output",
        default=None,
        metavar="DIR",
        help="Directory of per-clip pipeline dry-run JSON (content-matched to gold)",
    )
    p.add_argument("--judge-url", default=base_url_from_env())
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--out", default=str(_OUT))
    return p


def _parse_dna(result_text: str) -> dict | None:
    """Extract a v0.2 DNA dict from a Scout ``result`` string (```json fence or raw {...})."""
    fences = _FENCE.findall(result_text or "")
    raw = fences[-1] if fences else (result_text or "")
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        s, e = raw.rfind("{"), raw.rfind("}")
        if s != -1 and e > s:
            try:
                obj = json.loads(raw[s : e + 1])
                return obj if isinstance(obj, dict) else None
            except (json.JSONDecodeError, ValueError):
                return None
        return None


def records_from_pipeline_output(out_dir: str, gold_path: str) -> tuple[list[dict], list[str]]:
    """Build judge records by content-matching pipeline segments to gold clips.

    Match key: (source_clip_id == gold video stem) AND |start_time - gold.start_s| < tol.
    Returns (records, unmatched_gold_clip_ids).
    """
    gold = [
        (Path(c["video"]).stem, float(c["start_s"]), c["clip_id"], c["risk_level"])
        for c in json.loads(Path(gold_path).read_text(encoding="utf-8"))["clips"]
    ]
    records: list[dict] = []
    matched: set[str] = set()
    for f in sorted(Path(out_dir).glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        for seg in doc.get("segments", []):
            src = seg.get("source_clip_id")
            st = seg.get("segment", {}).get("start_time")
            if src is None or st is None:
                continue
            st = float(st)
            for g_src, g_start, g_clip, g_gt in gold:
                if g_src == src and abs(g_start - st) < _MATCH_TOL_S and g_clip not in matched:
                    dna = _parse_dna(seg.get("result", ""))
                    if dna is not None:
                        records.append({"clip_id": g_clip, "dna_json": dna, "gt": g_gt})
                        matched.add(g_clip)
                    break
    unmatched = [c[2] for c in gold if c[2] not in matched]
    return records, unmatched


async def _records_from_pg(gold_path: str) -> list[dict]:
    gt_map = load_gold_gt(gold_path)
    repo = await PGRepository.create(dsn_from_env())
    try:
        rows = await repo.list_v02_dna(clip_ids=[UUID(c) for c in gt_map], limit=len(gt_map) or 1)
        return attach_gt(rows, gt_map)
    finally:
        await repo.close()


async def _run(args: argparse.Namespace) -> None:
    if args.from_pipeline_output:
        records, unmatched = records_from_pipeline_output(args.from_pipeline_output, args.gold_set)
        log.info(
            "Matched %d gold segments from pipeline output; %d unmatched",
            len(records),
            len(unmatched),
        )
        if unmatched:
            log.warning("Unmatched gold clip_ids (%d): %s", len(unmatched), unmatched[:10])
    else:
        records = await _records_from_pg(args.gold_set)

    if not records:
        log.warning("No v0.2 DNA records resolved — nothing written.")
        return

    critic = QwenTextCritic(args.judge_url)
    try:
        result = await judge_pass(
            repo=None,
            critic=critic,
            system_prompt=load_system_prompt(),
            prompt_hash=judge_prompt_hash(),
            records=records,
            n_samples=args.n_samples,
            dry_run=True,  # prediction generation only — never mutate DB
        )
    finally:
        await critic.aclose()

    preds = [
        {"clip_id": str(j.clip_id), "scout": j.scout_risk, "final": j.final_risk, "gt": r.get("gt")}
        for j, r in zip(result["judgements"], records, strict=True)
    ]
    Path(args.out).write_text(json.dumps(preds, indent=2), encoding="utf-8")
    log.info("Wrote %d predictions -> %s", len(preds), args.out)
    log.info("Metrics (report-only): %s", result["metrics"])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run(_build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
