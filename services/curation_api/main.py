"""Shim — moved to my_curator.interfaces.http.curation_api.app.  Removed in R-7.

Re-exports the FastAPI ``app`` instance so the legacy CMD path
``services.curation_api.main:app`` continues to resolve through the
compose container until the Dockerfile CMD switches to
``my_curator.interfaces.http.curation_api.app:app`` (R-6 Dockerfile rewrite).
"""

from my_curator.interfaces.http.curation_api.app import app  # noqa: F401

__all__ = ["app"]
