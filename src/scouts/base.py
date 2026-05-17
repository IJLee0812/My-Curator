"""Shim — moved to my_curator.domain.scout.base.  Removed in R-7."""

from my_curator.domain.scout.base import Scout, ScoutConfig, ScoutReport  # noqa: F401

__all__ = ["Scout", "ScoutConfig", "ScoutReport"]
