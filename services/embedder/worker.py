"""Embedder worker — Kafka consumer that embeds video clips and upserts to Milvus (P3-1).

Subscribe to ``curation.clip.scouted``, read clip_id + frames_blob_uri from
each message, download 8 JPEG frames from MinIO, run Cosmos-Embed1-336p, and
upsert the 768-dim vector into Milvus ``clip_video_embed``.

CLI usage::

    python -m services.embedder.worker \\
        --milvus-uri http://localhost:19530 \\
        [--broker localhost:9092] \\
        [--topic curation.clip.scouted] \\
        [--frames-bucket frames] \\
        [--timeout 300000]

Environment variable fallbacks:
    KAFKA_BROKER, MILVUS_URI, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from uuid import UUID

log = logging.getLogger(__name__)

_FRAMES_BUCKET = "frames"
_MIN_DURATION_S = 3.0
_DEFAULT_TOPIC = "curation.clip.scouted"


class EmbedderWorker:
    """Processes ``curation.clip.scouted`` messages: MinIO frames → Cosmos-Embed1 → Milvus.

    Designed for injection in tests: pass model/minio/milvus directly.
    """

    def __init__(self, model, minio, milvus, *, frames_bucket: str = _FRAMES_BUCKET) -> None:
        self._model = model
        self._minio = minio
        self._milvus = milvus
        self._frames_bucket = frames_bucket
        self.embedded = 0
        self.skipped = 0
        self.errors = 0

    async def handle(self, data: dict) -> None:
        """Process a single ``curation.clip.scouted`` message."""
        clip_id_str = data.get("clip_id")
        frames_blob_uri = data.get("frames_blob_uri")
        duration = data.get("segment", {}).get("duration", 0.0)

        if not clip_id_str or not frames_blob_uri:
            log.debug("No clip_id/frames_blob_uri — skipping (legacy message or partial segment)")
            self.skipped += 1
            return

        if duration < _MIN_DURATION_S:
            log.debug(
                "Segment duration %.2fs < %.1fs — skipping clip %s",
                duration,
                _MIN_DURATION_S,
                clip_id_str,
            )
            self.skipped += 1
            return

        try:
            from services.embedder.frame_loader import load_frames

            tensor = await load_frames(self._minio, self._frames_bucket, frames_blob_uri)
            embedding = self._model.embed(tensor)
            await self._milvus.upsert(UUID(clip_id_str), embedding)
            self.embedded += 1
            log.debug("Embedded clip %s (%d-dim)", clip_id_str, len(embedding))
        except Exception:
            log.exception("Embed failed for clip %s", clip_id_str)
            self.errors += 1

    async def bulk_embed(self, parquet_path: str) -> int:
        """Batch-embed all rows in a parquet file and upsert to Milvus.

        Parquet schema: ``clip_id`` (str UUID), ``frames_blob_uri`` (str),
        ``duration`` (float).

        Returns:
            Number of successfully embedded rows.
        """
        import pyarrow.parquet as pq

        from services.embedder.frame_loader import load_frames

        table = pq.read_table(parquet_path)
        records = table.to_pylist()
        batch: list[dict] = []
        embedded = 0

        for row in records:
            try:
                tensor = await load_frames(self._minio, self._frames_bucket, row["frames_blob_uri"])
                vec = self._model.embed(tensor)
                batch.append({"clip_id": UUID(row["clip_id"]), "embedding": vec})
                if len(batch) >= 32:
                    await self._milvus.batch_upsert(batch)
                    embedded += len(batch)
                    batch = []
            except Exception:
                log.exception("bulk_embed failed for clip %s", row.get("clip_id"))

        if batch:
            await self._milvus.batch_upsert(batch)
            embedded += len(batch)

        return embedded


# ── CLI entry-point ───────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    def _env(key: str) -> str | None:
        return os.environ.get(key)

    p = argparse.ArgumentParser(
        description="Embedder worker — Cosmos-Embed1 → Milvus upsert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--milvus-uri",
        default=_env("MILVUS_URI") or "http://localhost:19530",
        help="Milvus URI (env: MILVUS_URI)",
    )
    p.add_argument(
        "--broker",
        default=_env("KAFKA_BROKER") or "localhost:9092",
        help="Kafka bootstrap servers (env: KAFKA_BROKER)",
    )
    p.add_argument(
        "--topic",
        default=_DEFAULT_TOPIC,
        help=f"Kafka topic to subscribe (default: {_DEFAULT_TOPIC})",
    )
    p.add_argument(
        "--frames-bucket",
        default=_env("MINIO_FRAMES_BUCKET") or _FRAMES_BUCKET,
        help="MinIO bucket for JPEG frames (env: MINIO_FRAMES_BUCKET)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300000,
        help="Consumer timeout ms, 0 = run forever (default: 300000)",
    )
    return p


async def _run(args: argparse.Namespace) -> None:
    from src.storage.milvus import MilvusRepository
    from src.storage.minio import MinIORepository

    try:
        from kafka import KafkaConsumer
    except ImportError:
        log.error("kafka-python not installed")
        sys.exit(1)

    ep = os.environ.get("MINIO_ENDPOINT")
    ak = os.environ.get("MINIO_ACCESS_KEY")
    sk = os.environ.get("MINIO_SECRET_KEY")
    if not (ep and ak and sk):
        log.error("MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY not set")
        sys.exit(1)

    minio = await MinIORepository.create(ep, ak, sk)
    milvus = await MilvusRepository.create(args.milvus_uri)

    from services.embedder.model import CosmosEmbed1

    model = CosmosEmbed1()
    log.info("Cosmos-Embed1-336p loaded")

    worker = EmbedderWorker(model, minio, milvus, frames_bucket=args.frames_bucket)

    # kafka-python's KafkaConsumer rejects None for consumer_timeout_ms
    # (default is float('inf')); translate --timeout 0 → run-forever sentinel.
    timeout_ms: float | int = float("inf") if args.timeout == 0 else args.timeout
    kafka = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.broker,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="embedder-worker",
        consumer_timeout_ms=timeout_ms,
    )
    log.info("Listening on %s @ %s", args.topic, args.broker)

    try:
        for message in kafka:
            try:
                await worker.handle(message.value)
            except Exception:
                log.exception("Unhandled error on offset %d", message.offset)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        kafka.close()
        await minio.close()
        await milvus.close()

    log.info(
        "Done — embedded=%d skipped=%d errors=%d",
        worker.embedded,
        worker.skipped,
        worker.errors,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_arg_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
