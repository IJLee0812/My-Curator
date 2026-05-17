"""Shim — moved to my_curator.interfaces.http.curation_api.routers.stats.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.routers.stats import (  # noqa: F401
    ReviewCounts,
    StatsResponse,
    router,
)

__all__ = ["ReviewCounts", "StatsResponse", "router"]
