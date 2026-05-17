"""Shim — moved to my_curator.interfaces.http.curation_api.routers.ingest.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.routers.ingest import (  # noqa: F401
    IngestRequest,
    IngestResponse,
    router,
)

__all__ = ["IngestRequest", "IngestResponse", "router"]
