"""POST /v1/ingest — publish a scouted clip event to Kafka (P3-2).

Publishes to curation.clip.scouted only.  CurationConsumer owns the
Postgres write; this endpoint never writes to PG directly.

Computes a text embedding from DNA fields and writes it synchronously to
Milvus (with flush) BEFORE publishing to Kafka.  This guarantees the
embedding is searchable by the time CurationConsumer finishes the PG write
and the seeded_clip_id fixture returns.  CurationConsumer skips the Milvus
write for ingest-format messages to avoid a race-condition overwrite.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from my_curator.adapters.embed.text_tower import CosmosEmbed1Encoder
from my_curator.adapters.storage.milvus import MilvusRepository
from my_curator.domain.scout.dna_text import dna_to_text

from ..deps import get_embedder, get_milvus

log = logging.getLogger(__name__)
router = APIRouter()

_TOPIC = "curation.clip.scouted"


def _get_producer():
    from kafka import KafkaProducer  # kafka-python

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


# P4-7: DNA→text moved to my_curator.domain.scout.dna_text so the /v1/ingest
# text path and scripts/reembed_corpus.py share one implementation (v0.2 fields).
_dna_to_text = dna_to_text


class IngestRequest(BaseModel):
    clip_id: str
    session_id: str
    blob_uri: str
    start_s: float
    end_s: float
    dna_version: str
    dna_json: dict
    scout_prompt_hash: str
    pipeline_version: str
    frame_count: int | None = None
    is_gold: bool = False
    is_synthetic: bool = False
    judge_prompt_hash: str | None = None
    curation_meta: dict | None = None


class IngestResponse(BaseModel):
    published: bool
    topic: str
    clip_id: str


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    milvus: MilvusRepository = Depends(get_milvus),
    embedder: CosmosEmbed1Encoder = Depends(get_embedder),
) -> IngestResponse:
    # Compute and write the text embedding to Milvus synchronously before Kafka
    # publish.  CurationConsumer skips the Milvus step for ingest messages, so
    # this is the only embedding path for API-ingested clips.
    try:
        text = _dna_to_text(req.dna_json)
        vec = await asyncio.to_thread(embedder.encode_text, text)
        await milvus.upsert(_uuid.UUID(req.clip_id), vec)
        await milvus.flush()
    except Exception:
        log.warning("Milvus embed+upsert failed for ingest clip %s (non-fatal)", req.clip_id)

    payload = req.model_dump()

    def _publish() -> None:
        producer = _get_producer()
        future = producer.send(_TOPIC, value=payload)
        producer.flush(timeout=10)
        future.get(timeout=10)
        producer.close()

    try:
        await asyncio.to_thread(_publish)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kafka publish failed: {exc}") from exc

    return IngestResponse(published=True, topic=_TOPIC, clip_id=req.clip_id)
