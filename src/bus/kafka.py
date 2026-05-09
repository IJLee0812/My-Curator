###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
###################################################################################################

"""CurationConsumer — Kafka → Postgres storage bridge (P2-4 / P3-2 follow-up).

Subscribes to curation.clip.scouted and curation.clip.needs_review and writes
clip metadata + scenario DNA to Postgres.  Postgres is the system of record
for all ingested clips.

Milvus writes are owned exclusively by EmbedderWorker (DS pipeline path) and
the /v1/ingest endpoint (text-embedding path).  This consumer never writes
Milvus, eliminating the prior zero-vector stub race with EmbedderWorker.

CLI usage:
  python -m src.bus.kafka \\
      --session-id SESSION_001 \\
      --dataset nuscenes \\
      --subset val \\
      --dataset-version 1.0 \\
      [--broker localhost:9092] \\
      [--timeout 300000]

Environment variable fallbacks (useful in Docker):
  SESSION_ID, CURATOR_DATASET, CURATOR_SUBSET, CURATOR_DATASET_VERSION,
  KAFKA_BROKER, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, PG_DB
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from src.scouts.versioning import assert_prompt_registered, resolve_dna_version

log = logging.getLogger(__name__)

_SCOUT_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "scout_cosmos_reason2.v1.md"
PIPELINE_VERSION = "p2-6"


def _compute_prompt_hash(path: Path) -> str:
    """SHA-256 of prompt file bytes, first 16 hex chars."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


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


def _parse_dna_json(result_text: str, curation_meta: dict) -> tuple[dict, dict]:
    """Extract DNA JSON from CoT text; return (dna_json, curation_meta).

    Uses DNAValidator 3-stage extraction (last ```json``` fence →
    outermost {...} block → raw_text fallback).  curation_meta is
    returned separately for storage in the curation_meta column —
    it is never merged into dna_json.
    """
    from src.scouts.dna_validator import DNAValidator

    validator = DNAValidator()
    dna = validator.extract_json(result_text)
    if dna is None:
        dna = {"raw_text": result_text}
    return dna, curation_meta


class CurationConsumer:
    """Processes Kafka curation messages and writes to Postgres.

    Designed for injection in tests: pass pg repo directly.
    The public handle_* methods are async and await the DAL calls.
    Milvus writes are NOT performed here — EmbedderWorker (DS pipeline path)
    and /v1/ingest (text path) are the sole Milvus writers.
    """

    def __init__(
        self,
        pg,
        scout_prompt_hash: str,
        session_id: str,
        topic_scouted: str = "curation.clip.scouted",
        topic_needs_review: str = "curation.clip.needs_review",
    ) -> None:
        self._pg = pg
        self._scout_prompt_hash = scout_prompt_hash
        self._session_id = session_id
        self._topic_scouted = topic_scouted
        self._topic_needs_review = topic_needs_review
        self.processed = 0
        self.errors = 0

    async def handle(self, topic: str, data: dict) -> None:
        if topic == self._topic_scouted:
            if "dna_json" in data:
                await self._handle_scouted_ingest(data)
            else:
                await self._handle_scouted(data)
        elif topic == self._topic_needs_review:
            await self._handle_needs_review(data)
        else:
            log.warning("Unknown topic: %s — skipping", topic)
            return
        self.processed += 1

    async def _handle_scouted(self, data: dict) -> None:
        # P3-1: DS pipeline sets clip_id in message so MinIO frames key aligns with Milvus key.
        _clip_id_str = data.get("clip_id")
        clip_id = uuid.UUID(_clip_id_str) if _clip_id_str else uuid.uuid4()
        stream_id = data["stream_id"]
        start_s: float = data["segment"]["start_time"]
        end_s: float = data["segment"]["end_time"]
        blob_uri = f"stream://{stream_id}/{start_s:.2f}-{end_s:.2f}"
        frames_blob_uri = data.get("frames_blob_uri")
        dna_json, curation_meta = _parse_dna_json(data.get("result", ""), data.get("curation", {}))
        dna_json["clip_id"] = str(clip_id)

        # PG write — system of record; abort on failure.
        # Milvus is written by EmbedderWorker (DS path) consuming the same Kafka
        # topic in parallel; this consumer never touches Milvus.
        try:
            await self._pg.write_clip_with_dna(
                session_id=self._session_id,
                clip_id=clip_id,
                blob_uri=blob_uri,
                start_s=start_s,
                end_s=end_s,
                dna_version=resolve_dna_version(self._scout_prompt_hash),
                dna_json=dna_json,
                scout_prompt_hash=self._scout_prompt_hash,
                pipeline_version=PIPELINE_VERSION,
                curation_meta=curation_meta,
                frames_blob_uri=frames_blob_uri,
            )
        except Exception:
            log.exception("PG write failed for scouted clip %s", clip_id)
            self.errors += 1
            return

        # Extra review_queue row for schema-invalid clips
        if not data.get("metadata", {}).get("json_valid", True):
            try:
                await self._pg.insert_review_queue(
                    clip_id=clip_id,
                    state="rejected_schema_invalid",
                    reason="json_valid=False in publisher metadata",
                )
            except Exception:
                log.warning("review_queue INSERT failed for schema-invalid clip %s", clip_id)

        log.debug(
            "Scouted clip %s written (stream %s, %.2f–%.2f s)", clip_id, stream_id, start_s, end_s
        )

    async def _handle_scouted_ingest(self, data: dict) -> None:
        """Handle /v1/ingest format messages — pre-computed DNA, no VLM parsing (P3-2)."""
        clip_id = uuid.UUID(data["clip_id"])
        session_id = data.get("session_id") or self._session_id
        blob_uri = data["blob_uri"]
        start_s: float = data["start_s"]
        end_s: float = data["end_s"]
        dna_json: dict = data["dna_json"]
        dna_version: str = data.get("dna_version") or "0.1"
        scout_prompt_hash: str = data.get("scout_prompt_hash") or self._scout_prompt_hash
        pipeline_version: str = data.get("pipeline_version") or PIPELINE_VERSION
        curation_meta: dict = data.get("curation_meta") or {}

        # Ensure session row exists (ON CONFLICT DO NOTHING)
        try:
            await self._pg.insert_session(
                session_id=session_id,
                dataset=data.get("dataset", "ingest"),
                subset=data.get("subset", "api"),
                dataset_version=data.get("dataset_version", "0"),
                recorded_at=datetime.now(timezone.utc),
                source_kind="synthetic" if data.get("is_synthetic") else "real",
            )
        except Exception:
            log.warning("insert_session failed for ingest session %s (non-fatal)", session_id)

        try:
            await self._pg.write_clip_with_dna(
                session_id=session_id,
                clip_id=clip_id,
                blob_uri=blob_uri,
                start_s=start_s,
                end_s=end_s,
                dna_version=dna_version,
                dna_json=dna_json,
                scout_prompt_hash=scout_prompt_hash,
                pipeline_version=pipeline_version,
                curation_meta=curation_meta,
            )
        except Exception:
            log.exception("PG write failed for ingest clip %s", clip_id)
            self.errors += 1
            return

        # Milvus embedding is written by the /v1/ingest handler before publishing
        # this message; writing here would race-overwrite it with a stale vector.
        log.debug(
            "Ingest clip %s written (session %s, %.2f–%.2f s)", clip_id, session_id, start_s, end_s
        )

    async def _handle_needs_review(self, data: dict) -> None:
        clip_id = uuid.uuid4()
        stream_id = data["stream_id"]
        start_s: float = data["segment"]["start_time"]
        end_s: float = data["segment"]["end_time"]
        blob_uri = f"stream://{stream_id}/{start_s:.2f}-{end_s:.2f}"
        frames_blob_uri = data.get("frames_blob_uri")
        dna_json, curation_meta = _parse_dna_json(data.get("result", ""), data.get("curation", {}))
        dna_json["clip_id"] = str(clip_id)

        try:
            await self._pg.write_clip_with_dna(
                session_id=self._session_id,
                clip_id=clip_id,
                blob_uri=blob_uri,
                start_s=start_s,
                end_s=end_s,
                dna_version=resolve_dna_version(self._scout_prompt_hash),
                dna_json=dna_json,
                scout_prompt_hash=self._scout_prompt_hash,
                pipeline_version=PIPELINE_VERSION,
                curation_meta=curation_meta,
                frames_blob_uri=frames_blob_uri,
            )
        except Exception:
            log.exception("PG write failed for needs_review clip %s", clip_id)
            self.errors += 1
            return

        reason = curation_meta.get("reason") or "needs_review"
        try:
            await self._pg.insert_review_queue(
                clip_id=clip_id,
                state="pending",
                reason=reason,
            )
        except Exception:
            log.warning("review_queue INSERT failed for clip %s", clip_id)

        log.debug(
            "Needs-review clip %s written (stream %s, %.2f–%.2f s)",
            clip_id,
            stream_id,
            start_s,
            end_s,
        )


async def _run(args: argparse.Namespace, scout_prompt_hash: str) -> None:
    from src.storage.pg import PGRepository, dsn_from_env

    try:
        from kafka import KafkaConsumer
    except ImportError:
        log.error("kafka-python not installed — pip install kafka-python")
        sys.exit(1)

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
        topic_scouted=args.topic_scouted,
        topic_needs_review=args.topic_needs_review,
    )

    timeout_ms: int = -1 if args.timeout == 0 else args.timeout
    kafka = KafkaConsumer(
        args.topic_scouted,
        args.topic_needs_review,
        bootstrap_servers=args.broker,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="curation-consumer",
        consumer_timeout_ms=timeout_ms,
    )
    log.info(
        "Listening on [%s, %s] @ %s",
        args.topic_scouted,
        args.topic_needs_review,
        args.broker,
    )

    try:
        for message in kafka:
            try:
                await consumer.handle(message.topic, message.value)
            except Exception:
                log.exception("Unhandled error on message offset %d", message.offset)
                consumer.errors += 1
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        kafka.close()
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
