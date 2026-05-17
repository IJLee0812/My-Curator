"""CLI entrypoint for EmbedderWorker (formerly ``python -m services.embedder.worker``)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from my_curator.adapters.bus.kafka_consumer import run_consumer_loop
from my_curator.application.workers.embedder_worker import _FRAMES_BUCKET, EmbedderWorker

log = logging.getLogger(__name__)

_DEFAULT_TOPIC = "curation.clip.scouted"


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
    from my_curator.adapters.storage.milvus import MilvusRepository
    from my_curator.adapters.storage.minio import MinIORepository

    ep = os.environ.get("MINIO_ENDPOINT")
    ak = os.environ.get("MINIO_ACCESS_KEY")
    sk = os.environ.get("MINIO_SECRET_KEY")
    if not (ep and ak and sk):
        log.error("MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY not set")
        sys.exit(1)

    minio = await MinIORepository.create(ep, ak, sk)
    milvus = await MilvusRepository.create(args.milvus_uri)

    from my_curator.adapters.embed.video_tower import CosmosEmbed1

    model = CosmosEmbed1()
    log.info("Cosmos-Embed1-336p loaded")

    worker = EmbedderWorker(model, minio, milvus, frames_bucket=args.frames_bucket)

    async def _handler(_topic: str, value: dict) -> None:
        await worker.handle(value)

    try:
        await run_consumer_loop(
            _handler,
            topics=[args.topic],
            broker=args.broker,
            group_id="embedder-worker",
            timeout_ms=args.timeout,
        )
    finally:
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
