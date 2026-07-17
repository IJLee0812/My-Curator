"""CurationConsumer — Kafka → Postgres storage bridge (P2-4 / P3-2 follow-up).

Pure application-layer use-case: receives parsed Kafka payloads and writes
clip metadata + scenario DNA to Postgres via PGRepository.  Postgres is the
system of record for all ingested clips.

Milvus writes are owned exclusively by EmbedderWorker (DS pipeline path) and
the /v1/ingest endpoint (text-embedding path).  This consumer never writes
Milvus, eliminating the prior zero-vector stub race with EmbedderWorker.

This module exposes only the use-case class + DNA-parsing / prompt-hash
helpers.  KafkaConsumer wiring + argparse + entrypoint live in
``my_curator.cli.run_curation_consumer``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import struct
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from my_curator.domain.scout.dna_normalizer import ensure_managed_fields
from my_curator.domain.scout.dna_validator import DNAValidator
from my_curator.domain.scout.versioning import resolve_dna_version

log = logging.getLogger(__name__)


def _moov_at_end(path: Path) -> bool:
    """Return True if the MP4 moov box comes after mdat (needs faststart)."""
    try:
        with path.open("rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    return False
                size, box_type = struct.unpack(">I4s", header)
                name = box_type.decode("ascii", errors="replace")
                if name == "moov":
                    return False
                if name == "mdat":
                    return True
                if size < 8:
                    return False
                f.seek(f.tell() - 8 + size)
    except OSError:
        return False


def _run_faststart(path: Path) -> None:
    """Apply ffmpeg -movflags +faststart in-place (blocking, run in thread)."""
    tmp = path.with_suffix(".faststart.tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c", "copy", "-movflags", "+faststart", str(tmp)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp.replace(path)
        log.info("faststart applied: %s", path.name)
    except Exception:
        tmp.unlink(missing_ok=True)
        log.warning("faststart failed for %s — skipping", path.name, exc_info=True)


_SCOUT_PROMPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "prompts" / "scout_cosmos_reason2.v2.md"
)
PIPELINE_VERSION = "p2-6"


def _compute_prompt_hash(path: Path) -> str:
    """SHA-256 of prompt file bytes, first 16 hex chars."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _parse_dna_json(result_text: str, curation_meta: dict) -> tuple[dict, dict]:
    """Extract DNA JSON from CoT text; return (dna_json, curation_meta).

    Uses DNAValidator 3-stage extraction (last ```json``` fence →
    outermost {...} block → raw_text fallback).  curation_meta is
    returned separately for storage in the curation_meta column —
    it is never merged into dna_json.
    """
    from my_curator.domain.scout.dna_normalizer import normalize_dna
    from my_curator.domain.scout.dna_validator import DNAValidator

    validator = DNAValidator()
    dna = validator.extract_json(result_text)
    if dna is None:
        dna = {"raw_text": result_text}
    else:
        dna = normalize_dna(dna)
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
        dataset: str = "curation",
        subset: str = "pipeline",
        dataset_version: str = "0",
        source_kind: str = "real",
        topic_scouted: str = "curation.clip.scouted",
        topic_needs_review: str = "curation.clip.needs_review",
    ) -> None:
        self._pg = pg
        self._scout_prompt_hash = scout_prompt_hash
        self._session_id = session_id
        self._dataset = dataset
        self._subset = subset
        self._dataset_version = dataset_version
        self._source_kind = source_kind
        self._topic_scouted = topic_scouted
        self._topic_needs_review = topic_needs_review
        self._validator = DNAValidator()  # schema-validate scouted DNA before marking pending
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
        # P3-4: use source_video_path (VIDEO_DATA_ROOT-relative) when the publisher
        # provides it; fall back to stream:// for legacy messages without path info.
        _svp = data.get("source_video_path")
        blob_uri = f"file://{_svp}" if _svp else f"stream://{stream_id}/{start_s:.2f}-{end_s:.2f}"
        if _svp and (video_root := os.environ.get("VIDEO_DATA_ROOT")):
            abs_path = Path(video_root) / _svp.lstrip("/")
            if abs_path.exists() and _moov_at_end(abs_path):
                await asyncio.to_thread(_run_faststart, abs_path)
        frames_blob_uri = data.get("frames_blob_uri")
        # P3-4: link the segment back to its original source clip identifier
        # when the publisher provides one.  Stays NULL otherwise (column is nullable).
        source_clip_id = data.get("source_clip_id")
        dna_json, curation_meta = _parse_dna_json(data.get("result", ""), data.get("curation", {}))
        resolved_version = resolve_dna_version(self._scout_prompt_hash)
        ensure_managed_fields(
            dna_json,
            dna_version=resolved_version,
            clip_id=clip_id,
            start_s=start_s,
            end_s=end_s,
            scout_prompt_hash=self._scout_prompt_hash,
            pipeline_version=PIPELINE_VERSION,
        )

        # Ensure session row exists — the startup upsert may have been wiped by a
        # DB reset while this consumer was running.  ON CONFLICT DO NOTHING is a no-op.
        try:
            await self._pg.insert_session(
                session_id=self._session_id,
                dataset=self._dataset,
                subset=self._subset,
                dataset_version=self._dataset_version,
                recorded_at=datetime.now(timezone.utc),
                source_kind=self._source_kind,
            )
        except Exception:
            log.warning("insert_session guard failed for %s (non-fatal)", self._session_id)

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
                dna_version=resolved_version,
                dna_json=dna_json,
                scout_prompt_hash=self._scout_prompt_hash,
                pipeline_version=PIPELINE_VERSION,
                curation_meta=curation_meta,
                frames_blob_uri=frames_blob_uri,
                source_clip_id=source_clip_id,
            )
        except Exception:
            log.exception("PG write failed for scouted clip %s", clip_id)
            self.errors += 1
            return

        # Schema-validate the normalized DNA before marking it clean. A degenerate/
        # truncated VLM output can parse as JSON yet miss required nested fields
        # (odd, planner_logic, …); such rows must be flagged for review, not stored
        # as pending. json_valid (publisher metadata) is only JSON-parse validity.
        schema_ok, schema_errs = self._validator.validate(dna_json)
        if not data.get("metadata", {}).get("json_valid", True):
            state, reason = "rejected_schema_invalid", "json_valid=False in publisher metadata"
        elif not schema_ok:
            state = "rejected_schema_invalid"
            reason = ("schema: " + (schema_errs[0] if schema_errs else "invalid"))[:200]
        else:
            state, reason = "pending", None
        try:
            await self._pg.insert_review_queue(clip_id=clip_id, state=state, reason=reason)
        except Exception:
            log.warning("review_queue INSERT failed for scouted clip %s (state=%s)", clip_id, state)

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
        source_clip_id = data.get("source_clip_id")

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
                source_clip_id=source_clip_id,
            )
        except Exception:
            log.exception("PG write failed for ingest clip %s", clip_id)
            self.errors += 1
            return

        # Milvus embedding is written by the /v1/ingest handler before publishing
        # this message; writing here would race-overwrite it with a stale vector.
        try:
            await self._pg.insert_review_queue(clip_id=clip_id, state="pending")
        except Exception:
            log.warning("review_queue INSERT failed for ingest clip %s", clip_id)

        log.debug(
            "Ingest clip %s written (session %s, %.2f–%.2f s)", clip_id, session_id, start_s, end_s
        )

    async def _handle_needs_review(self, data: dict) -> None:
        clip_id = uuid.uuid4()
        stream_id = data["stream_id"]
        start_s: float = data["segment"]["start_time"]
        end_s: float = data["segment"]["end_time"]
        _svp = data.get("source_video_path")
        blob_uri = f"file://{_svp}" if _svp else f"stream://{stream_id}/{start_s:.2f}-{end_s:.2f}"
        frames_blob_uri = data.get("frames_blob_uri")
        source_clip_id = data.get("source_clip_id")
        dna_json, curation_meta = _parse_dna_json(data.get("result", ""), data.get("curation", {}))
        resolved_version = resolve_dna_version(self._scout_prompt_hash)
        ensure_managed_fields(
            dna_json,
            dna_version=resolved_version,
            clip_id=clip_id,
            start_s=start_s,
            end_s=end_s,
            scout_prompt_hash=self._scout_prompt_hash,
            pipeline_version=PIPELINE_VERSION,
        )

        try:
            await self._pg.write_clip_with_dna(
                session_id=self._session_id,
                clip_id=clip_id,
                blob_uri=blob_uri,
                start_s=start_s,
                end_s=end_s,
                dna_version=resolved_version,
                dna_json=dna_json,
                scout_prompt_hash=self._scout_prompt_hash,
                pipeline_version=PIPELINE_VERSION,
                curation_meta=curation_meta,
                frames_blob_uri=frames_blob_uri,
                source_clip_id=source_clip_id,
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
