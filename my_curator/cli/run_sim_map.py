"""CLI entrypoint for the DNA -> sim-spec mapping pass.

Read-only against Postgres, no GPU and no CARLA: loads curated Scenario DNA, maps
every segment through the pure-domain mapper, and prints/writes the coverage report.

    python3 -m my_curator.cli.run_sim_map --all --json /tmp/coverage.json
    python3 -m my_curator.cli.run_sim_map --clip <uuid> --print-spec
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.domain.scout.versioning import CURRENT_DNA_VERSION
from my_curator.domain.sim import build_coverage_report, map_dna

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map curated Scenario DNA onto CARLA sim specs and report coverage",
    )
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--session", metavar="SESSION_ID", help="Map one session's segments")
    scope.add_argument("--clip", metavar="CLIP_ID", help="Map a single segment by clip_id")
    scope.add_argument(
        "--all", dest="all_segments", action="store_true", help="Map every segment (default)"
    )
    p.add_argument(
        "--dna-version",
        default=CURRENT_DNA_VERSION,
        help=f"Scenario DNA version to map (default: {CURRENT_DNA_VERSION})",
    )
    p.add_argument("--limit", type=int, default=2000, help="max segments to load")
    p.add_argument("--json", metavar="PATH", help="write the coverage report as JSON")
    p.add_argument(
        "--specs-dir",
        metavar="DIR",
        help="write one <clip_id>.json SimSpec per mapped segment",
    )
    p.add_argument(
        "--print-spec",
        action="store_true",
        help="print the full SimSpec for each mapped segment (use with --clip)",
    )
    return p


async def _load_rows(repo: PGRepository, args: argparse.Namespace) -> list[dict]:
    version = args.dna_version
    if args.clip:
        return await repo.list_dna(dna_version=version, clip_ids=[UUID(args.clip)], limit=1)
    if args.session:
        return await repo.list_dna(dna_version=version, session_id=args.session, limit=args.limit)
    return await repo.list_dna(dna_version=version, limit=args.limit)


def _as_dna(row: dict) -> dict:
    """``dna_json`` arrives as jsonb (dict) or as text depending on the driver codec."""
    dna = row["dna_json"]
    if isinstance(dna, str):
        dna = json.loads(dna)
    dna.setdefault("clip_id", str(row["clip_id"]))
    return dna


async def _run(args: argparse.Namespace) -> None:
    repo = await PGRepository.create(dsn_from_env())
    try:
        rows = await _load_rows(repo, args)
    finally:
        await repo.close()

    if not rows:
        log.warning("No DNA rows matched the selected scope — nothing to map.")
        return

    results = [map_dna(_as_dna(row)) for row in rows]
    report = build_coverage_report(results)

    print(report.render_text())

    if args.print_spec:
        for result in results:
            if result.spec is not None:
                print(f"\n--- {result.clip_id} ---")
                print(json.dumps(result.spec.to_dict(), indent=2))
            else:
                reasons = "; ".join(f"{r.value}: {d}" for r, d in result.exclusions)
                print(f"\n--- {result.clip_id} --- EXCLUDED: {reasons}")

    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        log.info("Coverage report written to %s", args.json)

    if args.specs_dir:
        out = Path(args.specs_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for result in results:
            if result.spec is None:
                continue
            (out / f"{result.clip_id}.json").write_text(
                json.dumps(result.spec.to_dict(), indent=2), encoding="utf-8"
            )
            written += 1
        log.info("Wrote %d SimSpec file(s) to %s", written, out)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run(_build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
