"""Pure-domain scout primitives: Protocol + dataclasses + scoring + validation.

No GStreamer / CUDA / vLLM / Kafka imports — host-testable.  Vendor adapters
(e.g. CosmosReasonScout) live under my_curator.adapters.scout.
"""

from my_curator.domain.scout.aggregator import BestOfNAggregator
from my_curator.domain.scout.base import Scout, ScoutConfig, ScoutReport
from my_curator.domain.scout.dna_validator import DNAValidator
from my_curator.domain.scout.versioning import (
    PROMPT_VERSION_MAP,
    assert_prompt_registered,
    resolve_dna_version,
)

__all__ = [
    "BestOfNAggregator",
    "DNAValidator",
    "PROMPT_VERSION_MAP",
    "Scout",
    "ScoutConfig",
    "ScoutReport",
    "assert_prompt_registered",
    "resolve_dna_version",
]
