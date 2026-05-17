"""Shim — moved to my_curator.interfaces.http.curation_api.routers.collections.  Removed in R-7."""

from my_curator.interfaces.http.curation_api.routers.collections import (  # noqa: F401
    CollectionInfo,
    CollectionsResponse,
    router,
)

__all__ = ["CollectionInfo", "CollectionsResponse", "router"]
