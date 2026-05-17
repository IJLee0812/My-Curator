"""Shim — moved to my_curator.adapters.storage.streaming.  Removed in R-7."""

from my_curator.adapters.storage.streaming import (  # noqa: F401
    _video_data_root,
    resolve_path,
    serve_segment,
)

__all__ = ["resolve_path", "serve_segment"]
