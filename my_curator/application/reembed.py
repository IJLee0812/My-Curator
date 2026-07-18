"""Corpus re-embed use-case (P4-7).

Rebuilds the hybrid Milvus collection from the durable sources of truth —
Postgres DNA (narrative-text vector, every schema-valid clip) and MinIO frames
(video vector, clips that have frames) — writing both vectors per clip in one
upsert.  This is the ONLY writer of the dual-vector collection: live
single-modality writers each hold just one vector, but a whole-row upsert into a
two-vector schema needs both, so the batch re-embed owns it.

Design points wired here:
- schema-valid filter = ``scene_description`` present AND ``DNAValidator`` passes.
- dual embed: text for all schema-valid clips; video only when frames exist.
- idempotent (upsert-by-clip_id → Milvus count invariant across re-runs).
- resumable: caller supplies the set of already-processed clip_ids and an
  ``on_processed`` callback to persist progress after each clip.

Pure application-layer orchestration: torch / GPU live behind the injected
``text_encoder`` / ``video_model`` and the lazily-imported ``load_frames``, so
this module is host-importable for unit tests.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID

from my_curator.domain.scout.dna_text import dna_to_text
from my_curator.domain.scout.dna_validator import DNAValidator

log = logging.getLogger(__name__)


class _TextEncoder(Protocol):
    def encode_text(self, text: str) -> list[float]: ...


class _VideoModel(Protocol):
    def embed(self, tensor: Any) -> list[float]: ...


@dataclass
class ReembedStats:
    total: int = 0
    embedded: int = 0
    with_video: int = 0
    text_only: int = 0
    skipped_invalid: int = 0
    skipped_resumed: int = 0
    video_errors: int = 0  # frames present but load/embed failed → stored text-only
    processed_ids: list[str] = field(default_factory=list)


def _is_schema_valid(dna: dict, validator: DNAValidator) -> bool:
    if not dna.get("scene_description"):
        return False
    ok, _ = validator.validate(dna)
    return ok


async def reembed_corpus(
    *,
    pg,
    minio,
    text_encoder: _TextEncoder,
    video_model: _VideoModel,
    hybrid_repo,
    frames_bucket: str = "frames",
    session_id: str | None = None,
    validator: DNAValidator | None = None,
    processed: set[str] | None = None,
    on_processed: Callable[[str], Awaitable[None]] | None = None,
    limit: int = 5000,
) -> ReembedStats:
    """Re-embed the v0.2 corpus into the hybrid collection.

    Args:
        pg: PGRepository (needs ``list_reembed_source``).
        minio: MinIORepository (frame downloads).
        text_encoder: Cosmos-Embed1 text tower (``encode_text``).
        video_model: Cosmos-Embed1 video tower (``embed``).
        hybrid_repo: MilvusHybridRepository (``upsert`` with text_vec/video_vec).
        processed: clip_ids (str) already done — skipped for resume.
        on_processed: awaited after each clip is upserted, for checkpointing.
    """
    from my_curator.adapters.storage.frame_loader import load_frames

    validator = validator or DNAValidator()
    processed = processed if processed is not None else set()
    stats = ReembedStats()

    rows = await pg.list_reembed_source(session_id=session_id, limit=limit)
    stats.total = len(rows)

    for row in rows:
        clip_id: UUID = row["clip_id"]
        cid = str(clip_id)
        if cid in processed:
            stats.skipped_resumed += 1
            continue

        dna: dict = row["dna_json"]
        if not _is_schema_valid(dna, validator):
            stats.skipped_invalid += 1
            continue

        text = dna_to_text(dna)
        text_vec = await asyncio.to_thread(text_encoder.encode_text, text)

        video_vec: list[float] | None = None
        frames_uri = row.get("frames_blob_uri")
        if frames_uri:
            try:
                tensor = await load_frames(minio, frames_bucket, frames_uri)
                video_vec = await asyncio.to_thread(video_model.embed, tensor)
            except Exception:
                log.exception("video embed failed for clip %s — storing text-only", cid)
                stats.video_errors += 1
                video_vec = None

        await hybrid_repo.upsert(clip_id, text_vec=text_vec, video_vec=video_vec)

        stats.embedded += 1
        if video_vec is not None:
            stats.with_video += 1
        else:
            stats.text_only += 1
        processed.add(cid)
        stats.processed_ids.append(cid)
        if on_processed is not None:
            await on_processed(cid)

        if stats.embedded % 25 == 0:
            log.info(
                "re-embed progress: %d embedded (%d video / %d text-only), %d skipped-invalid",
                stats.embedded,
                stats.with_video,
                stats.text_only,
                stats.skipped_invalid,
            )

    log.info(
        "re-embed done: total=%d embedded=%d video=%d text_only=%d "
        "skipped_invalid=%d skipped_resumed=%d video_errors=%d",
        stats.total,
        stats.embedded,
        stats.with_video,
        stats.text_only,
        stats.skipped_invalid,
        stats.skipped_resumed,
        stats.video_errors,
    )
    return stats
