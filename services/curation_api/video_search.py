"""Shim — moved to my_curator.interfaces.http.curation_api.routers.video_search.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.routers.video_search import (  # noqa: F401
    VideoSearchRequest,
    router,
)

__all__ = ["VideoSearchRequest", "router"]
