"""Shim — moved to my_curator.domain.scout.dna_validator.  Removed in R-7."""

from my_curator.domain.scout.dna_validator import (  # noqa: F401
    DNAValidator,
    _extract_last_object,
)

__all__ = ["DNAValidator"]
