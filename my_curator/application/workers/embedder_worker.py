"""Embedder worker — Kafka consumer that embeds video clips and upserts to Milvus (P3-1).

Pure application-layer use-case.  Reads ``curation.clip.scouted`` messages,
downloads 8 JPEG frames from MinIO, runs Cosmos-Embed1-336p, and upserts the
768-dim vector into Milvus ``clip_video_embed``.

KafkaConsumer wiring + argparse + entrypoint live in
``my_curator.cli.run_embedder``.
"""

from __future__ import annotations

import logging
from uuid import UUID

log = logging.getLogger(__name__)

_FRAMES_BUCKET = "frames"
_MIN_DURATION_S = 3.0


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
            from my_curator.adapters.storage.frame_loader import load_frames

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

        from my_curator.adapters.storage.frame_loader import load_frames

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
