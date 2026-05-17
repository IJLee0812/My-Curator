"""Shim — moved to my_curator.adapters.storage.pg.  Removed in R-7."""

from my_curator.adapters.storage.pg import (  # noqa: F401
    PGRepository,
    _setup_connection,
    dsn_from_env,
)

__all__ = ["PGRepository", "dsn_from_env"]
