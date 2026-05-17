"""Shim — moved to my_curator.adapters.storage.milvus.  Removed in R-7."""

from my_curator.adapters.storage.milvus import (  # noqa: F401
    COLLECTION_NAME,
    DIM,
    MilvusRepository,
)

__all__ = ["COLLECTION_NAME", "DIM", "MilvusRepository"]
