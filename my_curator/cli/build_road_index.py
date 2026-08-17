"""Build the town road index from CARLA's shipped OpenDRIVE files.

One-time (or after a CARLA upgrade): parses the built-in town networks into one row per
driving lane per lane section and swaps them into ``sim_road_index``. The table is created
if absent, so this works against a database whose volume predates the DDL.

The ``.xodr`` files live inside the CARLA image. Copy them out once, then point this at
the directory:

    docker cp my-curator-carla:/home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive /tmp/xodr
    python3 -m my_curator.cli.build_road_index --opendrive-dir /tmp/xodr
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter
from pathlib import Path

from my_curator.adapters.sim.xodr_parser import CARLA_OPENDRIVE_DIR, parse_towns
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.domain.sim.catalog import LOADABLE_TOWNS

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse CARLA town OpenDRIVE into sim_road_index")
    p.add_argument(
        "--opendrive-dir",
        default=CARLA_OPENDRIVE_DIR,
        help=f"directory holding <Town>.xodr (default: {CARLA_OPENDRIVE_DIR})",
    )
    p.add_argument(
        "--towns",
        nargs="+",
        default=list(LOADABLE_TOWNS),
        help="towns to index (default: every loadable town)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and summarize without writing to Postgres",
    )
    return p


async def _run(args: argparse.Namespace) -> None:
    candidates = parse_towns(Path(args.opendrive_dir), tuple(args.towns))
    if not candidates:
        log.error("No candidates parsed from %s — is the directory correct?", args.opendrive_dir)
        return

    per_town = Counter(c.town for c in candidates)
    for town, count in sorted(per_town.items()):
        roads = len({c.road_id for c in candidates if c.town == town})
        log.info("%-9s %5d lane candidate(s) across %4d road(s)", town, count, roads)
    log.info("Total: %d candidate(s)", len(candidates))

    if args.dry_run:
        log.info("--dry-run: nothing written")
        return

    repo = await PGRepository.create(dsn_from_env())
    try:
        written = await repo.replace_sim_road_index([c.to_dict() for c in candidates])
    finally:
        await repo.close()
    log.info("sim_road_index rebuilt with %d row(s)", written)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run(_build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
