"""Integration tests for P4-5: VLM segment overlap (0 → 1 s) + selection_fps (1 → 2).

Verifies (no GPU required):
  - compute_step_ns / compute_sample_interval_ns math (overlap=1; fps incl. shipped 2)
  - Segment start times follow the [0, 4, 8, ...] s pattern
  - configs/config_driving_scene.yaml ships overlap_sec=1 and selection_fps=2
  - source_clip_id propagates consistently across overlapping segments
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from my_curator.adapters.gst.utils import (
    clamp_segment_end_ns,
    compute_sample_interval_ns,
    compute_step_ns,
)
from my_curator.application.consumers.curation_consumer import CurationConsumer

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config_driving_scene.yaml"


@pytest.mark.integration
class TestOverlapSegmentMath:
    def test_step_ns_overlap_1(self):
        """overlap_sec=1, length=5 s → step = 4 s."""
        assert compute_step_ns(5, 1) == 4_000_000_000

    def test_step_ns_no_overlap_baseline(self):
        """overlap_sec=0 (pre-P4-5) → step = 5 s."""
        assert compute_step_ns(5, 0) == 5_000_000_000

    def test_step_clamps_when_overlap_ge_length(self):
        """overlap_sec >= length_sec → step clamped to 1 s minimum."""
        assert compute_step_ns(5, 5) == 1_000_000_000
        assert compute_step_ns(5, 6) == 1_000_000_000

    def test_segment_starts_are_4s_apart(self):
        """20 s video, overlap=1 → starts at [0, 4, 8, 12, 16] s."""
        step_ns = compute_step_ns(5, 1)
        starts_s = [t / 1_000_000_000 for t in range(0, 20 * 1_000_000_000, step_ns)]
        assert starts_s == [0.0, 4.0, 8.0, 12.0, 16.0]

    def test_overlap_segment_count_vs_baseline(self):
        """60 s source: overlap=1 yields 15 segments vs 12 without (+25%)."""
        total_s = 60
        step_base = compute_step_ns(5, 0) / 1_000_000_000
        step_overlap = compute_step_ns(5, 1) / 1_000_000_000
        assert (total_s / step_overlap) / (total_s / step_base) == pytest.approx(1.25)

    @pytest.mark.parametrize(
        "fps, expected_ns",
        [(4, 250_000_000), (2, 500_000_000), (1, 1_000_000_000), (0, None)],
    )
    def test_sample_interval_ns(self, fps, expected_ns):
        """selection_fps → sample interval (1 s / fps); fps=2 shipped, fps=0 disables (None)."""
        assert compute_sample_interval_ns(fps) == expected_ns

    @pytest.mark.parametrize("fps, expected_frames", [(4, 20), (2, 10), (1, 5)])
    def test_frames_per_5s_segment(self, fps, expected_frames):
        """5 s segment @ selection_fps → frame count; fps=2 (shipped) → 10 frames."""
        interval_ns = compute_sample_interval_ns(fps)
        assert (5 * 1_000_000_000) // interval_ns == expected_frames


@pytest.mark.integration
class TestClampSegmentEnd:
    """EOS flush clamp: trailing segment end must not exceed the last real frame."""

    _P = 33_333_333  # ~30 fps frame period

    def test_trailing_segment_clamped_to_last_frame(self):
        # 10 s clip @30 fps: last frame PTS ≈ 9.9667 s; segment [8.1 s, 13.1 s]
        start, end = 8_100_000_000, 13_100_000_000
        last_pts = 9_966_666_567
        clamped = clamp_segment_end_ns(start, end, last_pts, self._P)
        assert clamped == last_pts + self._P
        assert clamped / 1e9 == pytest.approx(10.0, abs=0.05)

    def test_inner_segment_unchanged(self):
        # Segment ends before the last frame → no clamp.
        start, end = 0, 5_000_000_000
        assert clamp_segment_end_ns(start, end, 9_966_666_567, self._P) == end

    def test_no_frames_observed_is_noop(self):
        assert clamp_segment_end_ns(0, 5_000_000_000, None, 0) == 5_000_000_000

    def test_never_clamps_below_start(self):
        # Pathological: last frame before segment start → clamp floors at start.
        start, end = 8_100_000_000, 13_100_000_000
        assert clamp_segment_end_ns(start, end, 1_000_000_000, self._P) == start

    def test_zero_period_falls_back_to_last_pts(self):
        # Single-frame stream (period unknown) → clamp to the frame PTS itself.
        start, end = 0, 5_000_000_000
        assert clamp_segment_end_ns(start, end, 2_000_000_000, 0) == 2_000_000_000


@pytest.mark.integration
@pytest.mark.parametrize("key, expected", [("overlap_sec", 1), ("selection_fps", 2)])
def test_config_ships_p4_5_values(key, expected):
    """configs/config_driving_scene.yaml must ship overlap_sec=1 and selection_fps=2 (P4-5)."""
    import yaml

    cfg = yaml.safe_load(_CONFIG_PATH.read_text())
    actual = cfg["segment"][key]
    assert actual == expected, (
        f"segment.{key}={actual} in config — expected {expected} after P4-5. "
        "Update configs/config_driving_scene.yaml."
    )


@pytest.mark.integration
class TestSourceClipIdAcrossOverlapSegments:
    """source_clip_id is consistent for all overlapping segments from the same source."""

    @pytest.fixture
    def consumer(self):
        """A CurationConsumer wired to a mock PG; yields (consumer, pg_mock)."""
        pg = AsyncMock()
        return CurationConsumer(pg, "abcd1234abcd1234", session_id="test-overlap-p45"), pg

    @staticmethod
    def _msg(start: float, source_clip_id: str | None = None) -> dict:
        msg = {
            "stream_id": 0,
            "segment": {"start_time": start, "end_time": start + 5.0, "duration": 5.0},
            "result": "{}",
            "curation": {},
            "metadata": {"json_valid": True},
        }
        if source_clip_id is not None:
            msg["source_clip_id"] = source_clip_id
        return msg

    async def test_overlap_segments_share_source_clip_id(self, consumer):
        """Two overlapping segments ([0,5),[4,9)) from one source share source_clip_id."""
        csm, pg = consumer
        await csm.handle("curation.clip.scouted", self._msg(0.0, "src_clip_042"))
        await csm.handle("curation.clip.scouted", self._msg(4.0, "src_clip_042"))

        assert pg.write_clip_with_dna.call_count == 2
        for call in pg.write_clip_with_dna.call_args_list:
            assert call.kwargs["source_clip_id"] == "src_clip_042"

    async def test_segments_from_different_sources_keep_own_id(self, consumer):
        """Segments from different source clips carry their own source_clip_id."""
        csm, pg = consumer
        await csm.handle("curation.clip.scouted", self._msg(0.0, "clip_A"))
        await csm.handle("curation.clip.scouted", self._msg(4.0, "clip_B"))

        calls = pg.write_clip_with_dna.call_args_list
        assert calls[0].kwargs["source_clip_id"] == "clip_A"
        assert calls[1].kwargs["source_clip_id"] == "clip_B"

    async def test_overlap_segment_start_times_reflect_4s_step(self, consumer):
        """The start_s values written to PG follow the 0, 4, 8 pattern (step = 4 s)."""
        csm, pg = consumer
        for start in (0.0, 4.0, 8.0):
            await csm.handle("curation.clip.scouted", self._msg(start, "src_clip_007"))

        starts = [c.kwargs["start_s"] for c in pg.write_clip_with_dna.call_args_list]
        assert starts == [0.0, 4.0, 8.0]

    async def test_source_clip_id_none_when_absent(self, consumer):
        """A segment without source_clip_id in the message passes None to PG."""
        csm, pg = consumer
        await csm.handle("curation.clip.scouted", self._msg(0.0))
        assert pg.write_clip_with_dna.call_args.kwargs["source_clip_id"] is None
