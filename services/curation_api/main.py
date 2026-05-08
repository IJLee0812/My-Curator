"""FastAPI application entry point for curation-api (P3-2).

Startup sequence (lifespan):
  1. Connect PGRepository (asyncpg pool)
  2. Connect MilvusRepository (GPU_CAGRA collection)
  3. Connect MinIORepository (boto3 S3 client)
  4. Load CosmosEmbed1Encoder (text + video towers, bfloat16, GPU)
  5. Warm-up: encode a dummy text string to JIT-compile the text tower
  6. Mark app.state.ready = True  ← /health returns 200 only after this

Shutdown: drain PG pool and close Milvus client.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.storage.milvus import MilvusRepository
from src.storage.minio import MinIORepository
from src.storage.pg import PGRepository, dsn_from_env

from .clips import router as clips_router
from .collections import router as collections_router
from .embedder import CosmosEmbed1Encoder
from .ingest import router as ingest_router
from .search import router as search_router
from .video_search import router as video_search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False

    milvus_uri = os.environ.get("MILVUS_URI", "http://my-curator-milvus:19530")
    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "http://my-curator-minio:9000")
    minio_user = os.environ["MINIO_USER"]
    minio_password = os.environ["MINIO_PASSWORD"]

    app.state.pg = await PGRepository.create(dsn_from_env())
    app.state.milvus = await MilvusRepository.create(milvus_uri)
    app.state.minio = await MinIORepository.create(minio_endpoint, minio_user, minio_password)

    encoder = await asyncio.to_thread(CosmosEmbed1Encoder)
    await asyncio.to_thread(encoder.encode_text, "warm-up")
    app.state.embedder = encoder

    app.state.ready = True
    yield

    await app.state.pg.close()
    await app.state.milvus.close()


app = FastAPI(title="curation-api", version="0.1.0", lifespan=lifespan)

app.include_router(search_router)
app.include_router(ingest_router)
app.include_router(clips_router)
app.include_router(collections_router)
app.include_router(video_search_router)


@app.get("/health")
async def health():
    if not getattr(app.state, "ready", False):
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}
