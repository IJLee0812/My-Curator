"""Backwards-compatibility shim — moved to my_curator.adapters.storage.

Removed in R-7.  New code should import from ``my_curator.adapters.storage.*``.
"""

import warnings

warnings.warn(
    "src.storage is moving to my_curator.adapters.storage; update imports.",
    DeprecationWarning,
    stacklevel=2,
)
