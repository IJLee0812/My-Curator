"""CLI entrypoint for the OpenSCENARIO compilation pass.

Maps curated DNA to a ``SimSpec``, resolves the spec's road query against the town road
index, compiles the pair to an OpenSCENARIO 1.0 document and validates it. Read-only
against Postgres apart from the index it reads; no GPU and no simulator.

    python3 -m my_curator.cli.run_sim_compile --all --out-dir "$SIM_ARTIFACT_DIR"
    python3 -m my_curator.cli.run_sim_compile --clip <uuid> --print
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from my_curator.adapters.sim.xosc_writer import serialize, validate, write
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.domain.scout.versioning import CURRENT_DNA_VERSION
from my_curator.domain.sim import map_dna
from my_curator.domain.sim.compilation import CompiledSegment, build_compilation_report
from my_curator.domain.sim.road_index import RoadCandidate, select_road
from my_curator.domain.sim.xosc_compiler import KPH_TO_MPS, compile_scenario

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compile curated DNA into OpenSCENARIO scenarios")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--session", metavar="SESSION_ID", help="compile one session's segments")
    scope.add_argument("--clip", metavar="CLIP_ID", help="compile a single segment")
    scope.add_argument(
        "--all", dest="all_segments", action="store_true", help="compile every segment (default)"
    )
    p.add_argument(
        "--dna-version",
        default=CURRENT_DNA_VERSION,
        help=f"Scenario DNA version to compile (default: {CURRENT_DNA_VERSION})",
    )
    p.add_argument("--limit", type=int, default=2000, help="max segments to load")
    p.add_argument("--out-dir", metavar="DIR", help="write one <clip_id>.xosc per segment")
    p.add_argument("--report", metavar="PATH", help="write the compilation report as JSON")
    p.add_argument("--print", action="store_true", help="print the compiled document")
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="do not check documents against the OpenSCENARIO XSD",
    )
    return p


def _as_dna(row: dict) -> dict:
    dna = row["dna_json"]
    if isinstance(dna, str):
        dna = json.loads(dna)
    dna.setdefault("clip_id", str(row["clip_id"]))
    return dna


def _as_candidate(row: dict) -> RoadCandidate:
    return RoadCandidate(
        town=row["town"],
        road_id=row["road_id"],
        lane_id=row["lane_id"],
        lane_section_s=row["lane_section_s"],
        lane_section_end_s=row["lane_section_end_s"],
        driving_lanes=row["driving_lanes"],
        speed_kph=row["speed_kph"],
        lane_types=frozenset(row["lane_types"]),
        junction_forms=frozenset(row["junction_forms"]),
        in_junction=row["in_junction"],
    )


async def _load(repo: PGRepository, args: argparse.Namespace) -> list[dict]:
    version = args.dna_version
    if args.clip:
        return await repo.list_dna(dna_version=version, clip_ids=[UUID(args.clip)], limit=1)
    if args.session:
        return await repo.list_dna(dna_version=version, session_id=args.session, limit=args.limit)
    return await repo.list_dna(dna_version=version, limit=args.limit)


def _compile_one(
    dna: dict,
    candidates: list[RoadCandidate],
    out_dir: Path | None,
    validate_docs: bool,
    print_doc: bool,
) -> CompiledSegment:
    result = map_dna(dna)
    clip_id = result.clip_id
    if result.spec is None:
        reasons = "; ".join(r.value for r, _ in result.exclusions) or "unmapped"
        return CompiledSegment(clip_id=clip_id, risk_level="unknown", failure=reasons)

    spec = result.spec
    # Enough road for the warm-up and the segment at the ego's target speed.
    min_length = (spec.warmup_s + spec.duration_s) * spec.ego.target_speed_kph * KPH_TO_MPS
    selection = select_road(spec.world.road, candidates, seed=clip_id, min_length_m=min_length)
    if selection is None:
        return CompiledSegment(
            clip_id=clip_id, risk_level=spec.risk_level, failure="road_index_empty"
        )

    root = compile_scenario(spec, selection)
    outcome = validate(root, clip_id) if validate_docs else None

    if print_doc:
        print(serialize(root))
    if out_dir is not None:
        write(root, out_dir / f"{clip_id}.xosc")

    return CompiledSegment(
        clip_id=clip_id,
        risk_level=spec.risk_level,
        town=selection.candidate.town,
        road_id=selection.candidate.road_id,
        lane_id=selection.candidate.lane_id,
        is_valid=outcome.is_valid if outcome else True,
        errors=outcome.errors if outcome else (),
        road_degradations=selection.degradations,
    )


async def _run(args: argparse.Namespace) -> None:
    repo = await PGRepository.create(dsn_from_env())
    try:
        rows = await _load(repo, args)
        index_rows = await repo.list_sim_road_index()
    finally:
        await repo.close()

    if not rows:
        log.warning("No DNA rows matched the selected scope — nothing to compile.")
        return
    if not index_rows:
        log.error(
            "sim_road_index is empty — run `python3 -m my_curator.cli.build_road_index` first."
        )
        return

    candidates = [_as_candidate(r) for r in index_rows]
    log.info("Loaded %d road candidate(s) from the index", len(candidates))

    out_dir = Path(args.out_dir) if args.out_dir else None
    segments = [
        _compile_one(_as_dna(row), candidates, out_dir, not args.skip_validation, args.print)
        for row in rows
    ]

    report = build_compilation_report(segments)
    print(report.render_text())

    if out_dir is not None:
        log.info("Scenario files written to %s", out_dir)
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        log.info("Compilation report written to %s", args.report)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run(_build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
