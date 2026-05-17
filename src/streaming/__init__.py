"""Backwards-compatibility shim — moved to my_curator.adapters.storage.streaming
(byte-range / presigned URL helpers) and my_curator.domain.timestamp
(sidecar parser).  Removed in R-7.
"""

import warnings

warnings.warn(
    "src.streaming is moving to my_curator.adapters.storage.streaming "
    "(byte-range helpers) and my_curator.domain.timestamp (sidecar parser); "
    "update imports.",
    DeprecationWarning,
    stacklevel=2,
)
