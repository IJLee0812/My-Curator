"""Shim — moved to my_curator.adapters.storage.frame_loader.  Removed in R-7."""

from my_curator.adapters.storage.frame_loader import (  # noqa: F401
    FRAME_SIZE,
    NUM_FRAMES,
    _download_with_retry,
    load_frames,
)

__all__ = ["FRAME_SIZE", "NUM_FRAMES", "load_frames"]
