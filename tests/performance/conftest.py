"""Performance test fixtures — skip entire suite if curation-api is unreachable."""

import httpx
import pytest

API_BASE = "http://localhost:8001"


def pytest_collection_modifyitems(config, items):
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        skip = pytest.mark.skip(reason=f"curation-api not reachable at {API_BASE}")
        for item in items:
            if "performance" in (m.name for m in item.iter_markers()):
                item.add_marker(skip)
