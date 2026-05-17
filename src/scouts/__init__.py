"""Backwards-compatibility shim — moved to my_curator.domain.scout.

This module is preserved during R-1..R-6 of the my_curator/ refactor and will
be deleted in R-7.  All new code should import from
``my_curator.domain.scout.*`` (pure domain) and
``my_curator.adapters.scout.*`` (vendor adapters).
"""

import warnings

warnings.warn(
    "src.scouts is moving to my_curator.domain.scout (pure) and "
    "my_curator.adapters.scout (vendor adapters); update imports.",
    DeprecationWarning,
    stacklevel=2,
)

from my_curator.domain.scout.aggregator import BestOfNAggregator  # noqa: E402, F401
from my_curator.domain.scout.base import Scout, ScoutConfig, ScoutReport  # noqa: E402, F401

# CosmosReasonScout still lives in src.scouts.cosmos_reason during R-1; it
# moves to my_curator.adapters.scout.cosmos_reason in R-2.
from src.scouts.cosmos_reason import CosmosReasonScout  # noqa: E402, F401

__all__ = ["Scout", "ScoutConfig", "ScoutReport", "CosmosReasonScout", "BestOfNAggregator"]
