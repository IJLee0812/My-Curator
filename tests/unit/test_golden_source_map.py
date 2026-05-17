"""GT-8: golden multi-stream source_map equivalence.

Verifies the per-stream (source_clip_id, source_video_path) propagation logic
inside ``VLMKafkaSignalPublisher``: two URIs (file://A, file://B) plus the
``--source-clip-id`` single-source override.  Published messages must surface
``source_clip_id`` and ``source_video_path`` from the per-stream entry.

References:
  docs/refactoring_plan.md  §3.1 GT-8.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_publisher(source_map):
    from my_curator.application.pipeline.publisher import VLMKafkaSignalPublisher

    return VLMKafkaSignalPublisher(
        {},
        "default-topic",
        dry_run=True,
        source_map=source_map,
    )


def _invoke(pub, stream_id: int, text: str = "scene"):
    element = MagicMock()
    element.get_llm.return_value = None
    pub.on_vlm_result(element, stream_id, 0.0, 5.0, text)


@pytest.mark.unit
class TestGoldenSourceMap:
    def test_two_streams_each_carry_own_source_fields(self):
        smap = {
            0: ("clipA", "session1/clipA/video/clipA.mp4"),
            1: ("clipB", "session2/clipB/video/clipB.mp4"),
        }
        pub = _make_publisher(smap)
        _invoke(pub, stream_id=0)
        _invoke(pub, stream_id=1)
        msg_a, msg_b = pub._collected_results
        assert msg_a["source_clip_id"] == "clipA"
        assert msg_a["source_video_path"] == "session1/clipA/video/clipA.mp4"
        assert msg_b["source_clip_id"] == "clipB"
        assert msg_b["source_video_path"] == "session2/clipB/video/clipB.mp4"

    def test_override_replaces_single_source_stem(self):
        # Simulates --source-clip-id override path (single source, stem replaced).
        smap = {0: ("override-id", "session/override-id/video/override-id.mp4")}
        pub = _make_publisher(smap)
        _invoke(pub, stream_id=0)
        msg = pub._collected_results[0]
        assert msg["source_clip_id"] == "override-id"

    def test_unmapped_stream_omits_source_fields(self):
        smap = {1: ("clipB", "path/clipB.mp4")}
        pub = _make_publisher(smap)
        _invoke(pub, stream_id=0)
        msg = pub._collected_results[0]
        assert "source_clip_id" not in msg
        assert "source_video_path" not in msg

    def test_partial_source_map_path_only(self):
        # source_clip_id present, source_video_path absent → only clip_id emitted.
        smap = {0: ("clipA", None)}
        pub = _make_publisher(smap)
        _invoke(pub, stream_id=0)
        msg = pub._collected_results[0]
        assert msg["source_clip_id"] == "clipA"
        assert "source_video_path" not in msg
