"""Shim — moved to my_curator.adapters.storage.minio.  Removed in R-7."""

from my_curator.adapters.storage.minio import MinIORepository  # noqa: F401

__all__ = ["MinIORepository"]
