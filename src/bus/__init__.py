"""Backwards-compatibility shim — moved to my_curator.application.consumers
(CurationConsumer) and my_curator.adapters.bus (KafkaConsumer loop).
Removed in R-7.
"""

import warnings

warnings.warn(
    "src.bus is moving to my_curator.application.consumers + my_curator.adapters.bus; "
    "update imports.",
    DeprecationWarning,
    stacklevel=2,
)
