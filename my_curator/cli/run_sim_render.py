"""CLI entrypoint for the render pass: a curated clip becomes three videos.

The host half of a render. It picks the segment, compiles its scenario, brings the
simulator up on the right town, hands the staging over to the container, uploads what comes
back and records the attempt — successful or not — in the render ledger.

    python3 -m my_curator.cli.run_sim_render --clip <source_clip_id>
    python3 -m my_curator.cli.run_sim_render --all --ego-interaction --report out.json

A batch groups its work by town so each town is booted once, and keeps going when one
segment fails; a single render stops at the first problem.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from my_curator.adapters.sim import simulator_service
from my_curator.adapters.sim.xosc_writer import write
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.domain.scout.versioning import CURRENT_DNA_VERSION
from my_curator.domain.sim import map_dna, select_road
from my_curator.domain.sim.coverage import classify_scene_content
from my_curator.domain.sim.reasons import RenderFailure
from my_curator.domain.sim.render import (
    EGO_INTERACTION,
    FAILED,
    RENDERED,
    RenderOutcome,
    SegmentRef,
    build_render_report,
    group_by_town,
    select_segment,
)
from my_curator.domain.sim.road_index import RoadCandidate
from my_curator.domain.sim.xosc_compiler import KPH_TO_MPS, compile_scenario

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_BUCKET = "clips"
SYNTHETIC_PREFIX = "synthetic"

#: Where the container sees the two bind mounts the render needs.
CONTAINER_SIM_DIR = "/opt/sim"
CONTAINER_RENDER_DIR = "/opt/render"
CONTAINER_VIDEO_DIR = "/video"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-stage curated segments in CARLA and record them")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--clip", metavar="SOURCE_CLIP_ID", help="render one source clip")
    scope.add_argument("--all", dest="all_clips", action="store_true", help="render a batch")
    p.add_argument("--segment", type=int, help="segment index override (default: highest risk)")
    p.add_argument(
        "--ego-interaction",
        action="store_true",
        help="restrict a batch to segments that stage an ego interaction",
    )
    p.add_argument("--dna-version", default=CURRENT_DNA_VERSION)
    p.add_argument("--limit", type=int, default=5000, help="max segments to load")
    p.add_argument("--max-renders", type=int, help="stop a batch after this many attempts")
    p.add_argument("--sim-dir", help="where scenarios are written (default: $SIM_ARTIFACT_DIR)")
    p.add_argument("--render-dir", help="where videos are written (default: $SIM_RENDER_DIR)")
    p.add_argument("--report", metavar="PATH", help="write the render report as JSON")
    p.add_argument("--no-upload", action="store_true", help="leave videos on disk only")
    p.add_argument("--keep-simulator", action="store_true", help="leave the simulator running")
    p.add_argument("--dry-run", action="store_true", help="compile and plan, render nothing")
    return p


def _artifact_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = os.environ.get("DATA_ROOT", "")
    sim_dir = Path(args.sim_dir or os.environ.get("SIM_ARTIFACT_DIR") or f"{data_root}/sim")
    render_dir = Path(args.render_dir or os.environ.get("SIM_RENDER_DIR") or f"{data_root}/render")
    sim_dir.mkdir(parents=True, exist_ok=True)
    # Created before the simulator mounts it, and group/other-writable: the container
    # writes the videos as its own user and the host reads them back to upload. Left to
    # Docker, this appears as a root-owned mount point neither side can write.
    render_dir.mkdir(parents=True, exist_ok=True)
    render_dir.chmod(0o777)
    return sim_dir, render_dir


def _as_dna(row: dict) -> dict:
    dna = dict(row["dna_json"])
    dna.setdefault("clip_id", str(row["clip_id"]))
    return dna


def _segment_ref(row: dict, dna: dict) -> SegmentRef:
    planner = dna.get("planner_logic") or {}
    confidence = (dna.get("confidence") or {}).get("overall")
    return SegmentRef(
        clip_id=str(row["clip_id"]),
        source_clip_id=row["source_clip_id"],
        segment_index=int(row["segment_index"]),
        risk_level=str(planner.get("risk_level") or "unknown"),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        blob_uri=row["blob_uri"],
        start_s=float(row["start_s"]),
        end_s=float(row["end_s"]),
    )


def _candidates(index: list[dict]) -> list[RoadCandidate]:
    return [
        RoadCandidate(
            town=r["town"],
            road_id=r["road_id"],
            lane_id=r["lane_id"],
            lane_section_s=r["lane_section_s"],
            lane_section_end_s=r["lane_section_end_s"],
            driving_lanes=r["driving_lanes"],
            speed_kph=r["speed_kph"],
            lane_types=frozenset(r["lane_types"]),
            junction_forms=frozenset(r["junction_forms"]),
            in_junction=r["in_junction"],
        )
        for r in index
    ]


def _source_path(blob_uri: str) -> str:
    """A ``file://`` blob URI as the container sees it under its video mount."""
    return f"{CONTAINER_VIDEO_DIR}/{blob_uri.split('://', 1)[-1].lstrip('/')}"


def _compile_one(segment: SegmentRef, dna: dict, candidates: list[RoadCandidate], sim_dir: Path):
    """Compile the segment's scenario and its camera rig. Returns (path, selection, spec)."""
    result = map_dna(dna)
    if result.spec is None:
        return None, None, None
    spec = result.spec
    min_length = (spec.warmup_s + spec.duration_s) * spec.ego.target_speed_kph * KPH_TO_MPS
    selection = select_road(
        spec.world.road, candidates, seed=segment.clip_id, min_length_m=min_length
    )
    document = compile_scenario(spec, selection)
    sim_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = write(document, sim_dir / f"{segment.clip_id}.xosc")
    (sim_dir / f"{segment.clip_id}.cameras.json").write_text(
        json.dumps([camera.to_dict() for camera in spec.cameras], indent=2), encoding="utf-8"
    )
    return scenario_path, selection, spec


def _stage_arguments(segment: SegmentRef, spec, out_dir: str) -> list[str]:
    return [
        "--scenario",
        f"{CONTAINER_SIM_DIR}/{segment.clip_id}.xosc",
        "--cameras",
        f"{CONTAINER_SIM_DIR}/{segment.clip_id}.cameras.json",
        "--out-dir",
        out_dir,
        "--duration-s",
        str(spec.duration_s),
        "--source",
        _source_path(segment.blob_uri),
        "--source-start-s",
        str(segment.start_s),
        "--risk-level",
        segment.risk_level,
    ]


async def _upload(minio, clip_id: str, outputs: dict[str, str], render_dir: Path) -> list[str]:
    keys = []
    for view in ("ego", "chase", "compare"):
        local = render_dir / clip_id / f"{view}.mp4"
        if view not in outputs or not local.is_file():
            continue
        key = f"{SYNTHETIC_PREFIX}/{clip_id}/{view}.mp4"
        await minio.upload_file(SYNTHETIC_BUCKET, key, local, content_type="video/mp4")
        keys.append(key)
    return keys


async def _minio_client():
    from my_curator.adapters.storage.minio import MinIORepository

    endpoint = os.environ.get("MINIO_ENDPOINT")
    access = os.environ.get("MINIO_ACCESS_KEY")
    secret = os.environ.get("MINIO_SECRET_KEY")
    if not (endpoint and access and secret):
        log.warning("MinIO credentials absent — videos stay on disk")
        return None
    return await MinIORepository.create(endpoint, access, secret)


def _outcome(segment: SegmentRef, selection, spec, result: dict, keys: list[str]) -> RenderOutcome:
    rendered = result.get("status") == RENDERED
    candidate = selection.candidate if selection else None
    return RenderOutcome(
        clip_id=segment.clip_id,
        source_clip_id=segment.source_clip_id,
        segment_index=segment.segment_index,
        risk_level=segment.risk_level,
        status=RENDERED if rendered else FAILED,
        town=candidate.town if candidate else None,
        road_id=candidate.road_id if candidate else None,
        lane_id=candidate.lane_id if candidate else None,
        duration_s=spec.duration_s if spec else None,
        keys=tuple(keys),
        failure_reason=None if rendered else result.get("failure_reason", "unknown"),
    )


async def _record(repo: PGRepository, outcome: RenderOutcome) -> None:
    keys = list(outcome.keys) + ["", "", ""]
    await repo.record_sim_render(
        {
            "clip_id": outcome.clip_id,
            "source_clip_id": outcome.source_clip_id,
            "segment_index": outcome.segment_index,
            "status": outcome.status,
            "failure_reason": outcome.failure_reason,
            "town": outcome.town,
            "road_id": outcome.road_id,
            "lane_id": outcome.lane_id,
            "duration_s": outcome.duration_s,
            "ego_key": keys[0] or None,
            "chase_key": keys[1] or None,
            "compare_key": keys[2] or None,
        }
    )


async def _run(args: argparse.Namespace) -> int:
    sim_dir, render_dir = _artifact_dirs(args)
    repo = await PGRepository.create(dsn_from_env())
    try:
        rows = await repo.list_sim_segments(
            dna_version=args.dna_version, source_clip_id=args.clip, limit=args.limit
        )
        candidates = _candidates(await repo.list_sim_road_index())
        if not rows:
            log.error("no segments matched")
            return 1
        if not candidates:
            log.error("sim_road_index is empty — run my_curator.cli.build_road_index")
            return 1

        plans = _plan(rows, args, candidates, sim_dir)
        if not plans:
            log.error("nothing to render")
            return 1
        log.info("planned %d render(s) across %d town(s)", len(plans), len({p[3] for p in plans}))

        if args.dry_run:
            for segment, _, _, town in plans:
                print(
                    f"{segment.source_clip_id}[{segment.segment_index}] {segment.risk_level} → {town}"
                )
            return 0

        outcomes = await _render_all(repo, plans, args, sim_dir, render_dir)
    finally:
        await repo.close()

    report = build_render_report(outcomes)
    print(report.render_text())
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        log.info("render report written to %s", args.report)
    return 0 if report.rendered == report.attempted else 1


def _plan(rows: list[dict], args, candidates, sim_dir: Path) -> list[tuple]:
    """Choose one segment per source clip, compile it, and note the town it lands in."""
    by_source: dict[str, list[tuple[SegmentRef, dict]]] = {}
    for row in rows:
        dna = _as_dna(row)
        by_source.setdefault(row["source_clip_id"], []).append((_segment_ref(row, dna), dna))

    plans = []
    for source_clip_id in sorted(by_source):
        pairs = by_source[source_clip_id]
        chosen = select_segment([ref for ref, _ in pairs], index=args.segment)
        if chosen is None:
            log.warning("%s has no segment %s", source_clip_id, args.segment)
            continue
        dna = next(dna for ref, dna in pairs if ref.clip_id == chosen.clip_id)
        scenario, selection, spec = _compile_one(chosen, dna, candidates, sim_dir)
        if spec is None:
            log.warning("%s[%d] has no mappable DNA", source_clip_id, chosen.segment_index)
            continue
        if args.ego_interaction and classify_scene_content(spec) != EGO_INTERACTION:
            continue
        plans.append((chosen, selection, spec, selection.candidate.town))
        del scenario
        if args.max_renders and len(plans) >= args.max_renders:
            break
    return plans


async def _render_all(repo, plans, args, sim_dir: Path, render_dir: Path) -> list[RenderOutcome]:
    minio = None if args.no_upload else await _minio_client()
    single = len(plans) == 1
    by_town = group_by_town([(segment, town) for segment, _, _, town in plans])
    lookup = {segment.clip_id: (selection, spec) for segment, selection, spec, _ in plans}
    # The simulator sees these directories through bind mounts, so the paths chosen here
    # have to reach compose: otherwise the container writes its videos somewhere the host
    # never looks. Compose recreates the container when a mount moves.
    env = {
        **os.environ,
        "SIM_ARTIFACT_DIR": str(sim_dir),
        "SIM_RENDER_DIR": str(render_dir),
    }

    outcomes: list[RenderOutcome] = []
    try:
        for town, segments in by_town:
            simulator_service.boot(REPO_ROOT, town, host="127.0.0.1", port=_rpc_port(), env=env)
            for segment in segments:
                selection, spec = lookup[segment.clip_id]
                outcome = await _render_one(repo, minio, segment, selection, spec, render_dir)
                outcomes.append(outcome)
                status = "ok" if outcome.rendered else f"FAILED ({outcome.failure_reason})"
                log.info(
                    "%s[%d] %s %s → %s",
                    segment.source_clip_id,
                    segment.segment_index,
                    segment.risk_level,
                    town,
                    status,
                )
                if single and not outcome.rendered:
                    return outcomes
    finally:
        if not args.keep_simulator:
            simulator_service.shutdown(REPO_ROOT)
        if minio is not None:
            await minio.close()
    return outcomes


async def _render_one(repo, minio, segment, selection, spec, render_dir: Path) -> RenderOutcome:
    # The per-clip directory is the container's to create: it writes the frames and the
    # videos, and a host-created directory would not be writable by the simulator's user.
    try:
        result = simulator_service.run_stage(
            _stage_arguments(segment, spec, f"{CONTAINER_RENDER_DIR}/{segment.clip_id}")
        )
    except simulator_service.SimulatorError as exc:
        result = {
            "status": FAILED,
            "failure_reason": RenderFailure.STAGE_CRASHED.value,
            "detail": str(exc),
        }

    if result.get("kinematics"):
        log.info(
            "%s kinematics %s events=%s",
            segment.clip_id[:8],
            json.dumps(result["kinematics"], sort_keys=True),
            result.get("events_fired"),
        )

    keys: list[str] = []
    if result.get("status") == RENDERED and minio is not None:
        keys = await _upload(minio, segment.clip_id, result.get("outputs", {}), render_dir)

    outcome = _outcome(segment, selection, spec, result, keys)
    await _record(repo, outcome)
    return outcome


def _rpc_port() -> int:
    ports = os.environ.get("CARLA_HOST_PORTS", "2000-2002")
    return int(ports.split("-", 1)[0])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_run(_build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
