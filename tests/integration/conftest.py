"""Integration test configuration.

Mocks GStreamer / GPU stack at import time so that app modules can be
imported without a DeepStream Docker environment.  When real GStreamer is
present (inside Docker) the mocks are skipped and the real library is used.
"""

import sys
from unittest.mock import MagicMock


def _mock_gstreamer_if_absent() -> None:
    try:
        import gi  # noqa: F401

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401

        return  # Real GStreamer available — nothing to mock
    except Exception:
        pass

    gst_mock = MagicMock()
    gst_mock.Rank.NONE = 0
    gst_mock.PadProbeReturn.OK = 1

    gi_mock = MagicMock()
    gi_mock.require_version = MagicMock()

    sys.modules.setdefault("gi", gi_mock)
    sys.modules.setdefault("gi.repository", MagicMock())
    sys.modules.setdefault("gi.repository.Gst", gst_mock)
    sys.modules.setdefault("gi.repository.GLib", MagicMock())
    sys.modules.setdefault("gi.repository.GObject", MagicMock())
    sys.modules.setdefault("gstnvvllmvlm", MagicMock())


_mock_gstreamer_if_absent()
