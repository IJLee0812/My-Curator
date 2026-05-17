###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
###################################################################################################

"""CLI entrypoint for CurationConsumer (formerly ``python -m src.bus.kafka``).

Wires PGRepository, the prompt-hash startup guard, KafkaConsumer subscription,
and graceful shutdown around the pure ``CurationConsumer`` use-case.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from my_curator.adapters.bus.kafka_consumer import run_consumer_loop
from my_curator.application.consumers.curation_consumer import (
    _SCOUT_PROMPT_PATH,
    CurationConsumer,
    _compute_prompt_hash,
)
from my_curator.domain.scout.versioning import assert_prompt_registered

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    def _env(key: str) -> str | None:
        return os.environ.get(key)

    p = argparse.ArgumentParser(
        description="Curation consumer — writes Kafka events to Postgres + Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--session-id",
        default=_env("SESSION_ID"),
        required=not _env("SESSION_ID"),
        help="Session identifier (env: SESSION_ID)",
    )
    p.add_argument(
        "--dataset",
        default=_env("CURATOR_DATASET"),
        required=not _env("CURATOR_DATASET"),
        help="Dataset name (env: CURATOR_DATASET)",
    )
    p.add_argument(
        "--subset",
        default=_env("CURATOR_SUBSET"),
        required=not _env("CURATOR_SUBSET"),
        help="Dataset subset, e.g. train/val/test (env: CURATOR_SUBSET)",
    )
    p.add_argument(
        "--dataset-version",
        default=_env("CURATOR_DATASET_VERSION"),
        required=not _env("CURATOR_DATASET_VERSION"),
        help="Dataset version string (env: CURATOR_DATASET_VERSION)",
    )
    p.add_argument(
        "--source-kind",
        default="real",
        choices=["real", "synthetic"],
        help="Source kind for the sessions row (default: real)",
    )
    p.add_argument(
        "--broker",
        default=_env("KAFKA_BROKER") or "localhost:9092",
        help="Kafka bootstrap servers (env: KAFKA_BROKER, default: localhost:9092)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300000,
        help="Consumer timeout ms, 0 = run forever (default: 300000)",
    )
    p.add_argument(
        "--topic-scouted",
        default="curation.clip.scouted",
        help="Kafka topic for grounded clips (default: curation.clip.scouted)",
    )
    p.add_argument(
        "--topic-needs-review",
        default="curation.clip.needs_review",
        help="Kafka topic for review-queue clips (default: curation.clip.needs_review)",
    )
    return p


async def _run(args: argparse.Namespace, scout_prompt_hash: str) -> None:
    from my_curator.adapters.storage.pg import PGRepository, dsn_from_env

    dsn = getattr(args, "pg_dsn", None) or dsn_from_env()
    pg = await PGRepository.create(dsn)

    # Upsert session row (idempotent — ON CONFLICT DO NOTHING in PGRepository)
    await pg.insert_session(
        session_id=args.session_id,
        dataset=args.dataset,
        subset=args.subset,
        dataset_version=args.dataset_version,
        recorded_at=datetime.now(timezone.utc),
        source_kind=args.source_kind,
    )
    log.info("Session '%s' ready", args.session_id)

    consumer = CurationConsumer(
        pg,
        scout_prompt_hash=scout_prompt_hash,
        session_id=args.session_id,
        dataset=args.dataset,
        subset=args.subset,
        dataset_version=args.dataset_version,
        source_kind=args.source_kind,
        topic_scouted=args.topic_scouted,
        topic_needs_review=args.topic_needs_review,
    )

    timeout_ms: int = -1 if args.timeout == 0 else args.timeout
    try:
        await run_consumer_loop(
            consumer.handle,
            topics=[args.topic_scouted, args.topic_needs_review],
            broker=args.broker,
            group_id="curation-consumer",
            timeout_ms=timeout_ms,
        )
    finally:
        await pg.close()

    log.info("Done — processed=%d errors=%d", consumer.processed, consumer.errors)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_arg_parser().parse_args()

    if not _SCOUT_PROMPT_PATH.exists():
        log.error("Scout prompt file not found: %s", _SCOUT_PROMPT_PATH)
        sys.exit(1)

    scout_prompt_hash = _compute_prompt_hash(_SCOUT_PROMPT_PATH)
    log.info("Scout prompt hash: %s (from %s)", scout_prompt_hash, _SCOUT_PROMPT_PATH.name)

    try:
        assert_prompt_registered(scout_prompt_hash)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    asyncio.run(_run(args, scout_prompt_hash))


if __name__ == "__main__":
    main()
