"""Shim — moved to my_curator.domain.timestamp.  Removed in R-7."""

from my_curator.domain.timestamp import (  # noqa: F401
    _frame_aligned_times,
    get_precise_times,
    parse_timestamp_file,
)

__all__ = ["get_precise_times", "parse_timestamp_file"]
