"""Shim — moved to my_curator.interfaces.http.curation_api.routers.clips.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.routers.clips import (  # noqa: F401
    ClipDetail,
    ClipListResponse,
    ClipSummary,
    router,
)

__all__ = ["ClipDetail", "ClipListResponse", "ClipSummary", "router"]
