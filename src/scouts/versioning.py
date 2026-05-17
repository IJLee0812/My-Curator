"""Shim — moved to my_curator.domain.scout.versioning.  Removed in R-7."""

from my_curator.domain.scout.versioning import (  # noqa: F401
    PROMPT_VERSION_MAP,
    assert_prompt_registered,
    resolve_dna_version,
)

__all__ = ["PROMPT_VERSION_MAP", "assert_prompt_registered", "resolve_dna_version"]
