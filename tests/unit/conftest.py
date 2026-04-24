"""Unit-test-specific fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset config_loader singleton before and after each unit test.

    Scoped to tests/unit/ only — the singleton is irrelevant for integration
    and e2e tests which do not import config_loader directly.
    """
    try:
        import config_loader

        config_loader._config_instance = None
        yield
        config_loader._config_instance = None
    except ImportError:
        yield
