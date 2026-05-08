"""FastAPI dependency providers for curation-api (P3-2).

All shared resources (MilvusRepository, PGRepository, MinIORepository,
CosmosEmbed1Encoder) are initialised at app startup in main.py lifespan
and stored in app.state.  Depends() callables here extract them from state.
"""

from __future__ import annotations

from fastapi import Request

from src.storage.milvus import MilvusRepository
from src.storage.minio import MinIORepository
from src.storage.pg import PGRepository

from .embedder import CosmosEmbed1Encoder


def get_milvus(request: Request) -> MilvusRepository:
    return request.app.state.milvus


def get_pg(request: Request) -> PGRepository:
    return request.app.state.pg


def get_minio(request: Request) -> MinIORepository:
    return request.app.state.minio


def get_embedder(request: Request) -> CosmosEmbed1Encoder:
    return request.app.state.embedder
