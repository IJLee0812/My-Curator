"""Shim — moved to my_curator.adapters.storage.streaming.  Removed in R-7."""

from my_curator.adapters.storage.streaming import get_presigned_url  # noqa: F401

__all__ = ["get_presigned_url"]
