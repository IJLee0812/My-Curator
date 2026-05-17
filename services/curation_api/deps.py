"""Shim — moved to my_curator.interfaces.http.curation_api.deps.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.deps import (  # noqa: F401
    get_embedder,
    get_milvus,
    get_minio,
    get_pg,
)

__all__ = ["get_embedder", "get_milvus", "get_minio", "get_pg"]
