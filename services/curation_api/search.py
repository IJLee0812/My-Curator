"""Shim — moved to my_curator.interfaces.http.curation_api.routers.search.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.routers.search import (  # noqa: F401
    ClipResult,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    router,
)

__all__ = ["ClipResult", "SearchFilters", "SearchRequest", "SearchResponse", "router"]
