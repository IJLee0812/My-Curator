"""CLI entrypoint for the P4-7 hybrid corpus re-embed.

Rebuilds the dual-vector hybrid Milvus collection from Postgres DNA (text
tower) + MinIO frames (video tower).  Idempotent (upsert-by-clip_id) and
resumable via a JSONL checkpoint of processed clip_ids.

Run inside the curation-api / embedder image (torch + Cosmos-Embed1 present),
on GPU 0::

    python3 -m my_curator.cli.reembed_corpus --checkpoint .reembed.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    def _env(key: str) -> str | None:
        return os.environ.get(key)

    p = argparse.ArgumentParser(
        description="P4-7 hybrid corpus re-embed — PG DNA + MinIO frames → dual-vector Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--milvus-uri", default=_env("MILVUS_URI") or "http://localhost:19530")
    p.add_argument(
        "--hybrid-collection",
        default=None,
        help="Milvus hybrid collection name (default: clip_hybrid_embed)",
    )
    p.add_argument("--frames-bucket", default=_env("MINIO_FRAMES_BUCKET") or "frames")
    p.add_argument("--session-id", default=None, help="Only re-embed this session's clips")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument(
        "--checkpoint",
        default=".reembed_checkpoint.jsonl",
        help="JSONL file of processed clip_ids for resume (default: .reembed_checkpoint.jsonl)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and re-embed everything (still idempotent)",
    )
    return p


def _load_checkpoint(path: Path, resume: bool) -> set[str]:
    if not resume or not path.exists():
        return set()
    done = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    log.info("resuming: %d clip_ids already processed (from %s)", len(done), path)
    return done


async def _run(args: argparse.Namespace) -> None:
    from my_curator.adapters.embed.text_tower import CosmosEmbed1Encoder
    from my_curator.adapters.embed.video_tower import CosmosEmbed1
    from my_curator.adapters.storage.milvus import HYBRID_COLLECTION_NAME, MilvusHybridRepository
    from my_curator.adapters.storage.minio import MinIORepository
    from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
    from my_curator.application.reembed import reembed_corpus

    ep = os.environ.get("MINIO_ENDPOINT")
    user = os.environ.get("MINIO_USER") or os.environ.get("MINIO_ACCESS_KEY")
    password = os.environ.get("MINIO_PASSWORD") or os.environ.get("MINIO_SECRET_KEY")
    if not (ep and user and password):
        log.error("MINIO_ENDPOINT / MINIO_USER / MINIO_PASSWORD not set")
        sys.exit(1)

    collection = args.hybrid_collection or HYBRID_COLLECTION_NAME
    checkpoint = Path(args.checkpoint)
    processed = _load_checkpoint(checkpoint, resume=not args.no_resume)

    pg = await PGRepository.create(dsn_from_env())
    minio = await MinIORepository.create(ep, user, password)
    hybrid = await MilvusHybridRepository.create(args.milvus_uri, collection_name=collection)

    log.info("loading Cosmos-Embed1 towers (text + video) …")
    text_encoder = await asyncio.to_thread(CosmosEmbed1Encoder)
    video_model = await asyncio.to_thread(CosmosEmbed1)

    ckpt_fh = checkpoint.open("a")

    async def _on_processed(cid: str) -> None:
        ckpt_fh.write(cid + "\n")
        ckpt_fh.flush()

    try:
        stats = await reembed_corpus(
            pg=pg,
            minio=minio,
            text_encoder=text_encoder,
            video_model=video_model,
            hybrid_repo=hybrid,
            frames_bucket=args.frames_bucket,
            session_id=args.session_id,
            processed=processed,
            on_processed=_on_processed,
            limit=args.limit,
        )
        await hybrid.flush()
        count = await hybrid.count()
    finally:
        ckpt_fh.close()
        await minio.close()
        await hybrid.close()
        await pg.close()

    log.info(
        "DONE — embedded=%d (video=%d, text_only=%d) skipped_invalid=%d "
        "skipped_resumed=%d video_errors=%d | milvus entity count=%d",
        stats.embedded,
        stats.with_video,
        stats.text_only,
        stats.skipped_invalid,
        stats.skipped_resumed,
        stats.video_errors,
        count,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run(_build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
