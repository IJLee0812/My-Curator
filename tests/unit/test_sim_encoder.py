"""Encoding: the pipelines that get built, and the frame arithmetic behind the cut.

Pipeline construction is checked as text because a wrong element name only surfaces as a
GStreamer error at render time, and the segment cut is checked on real bytes because an
off-by-one frame there desynchronizes the comparison view from its source.
"""

from __future__ import annotations

import pytest

from my_curator.adapters.sim import encoder

pytestmark = pytest.mark.unit


def command(parts: list[str]) -> str:
    return " ".join(parts)


class TestViewPipeline:
    def test_it_reads_raw_frames_at_the_recorded_geometry(self, tmp_path):
        text = command(
            encoder.view_pipeline(
                tmp_path / "ego.bgra",
                tmp_path / "ego.mp4",
                width=1280,
                height=720,
                fps=10,
                overlay="critical",
            )
        )
        assert "rawvideoparse width=1280 height=720 format=bgra framerate=10/1" in text

    def test_it_encodes_h264_into_mp4(self, tmp_path):
        text = command(
            encoder.view_pipeline(
                tmp_path / "a.bgra", tmp_path / "a.mp4", width=64, height=64, fps=10, overlay=""
            )
        )
        assert "x264enc" in text
        assert "h264parse ! qtmux ! filesink" in text

    def test_the_overlay_and_the_clock_are_both_burned_in(self, tmp_path):
        text = command(
            encoder.view_pipeline(
                tmp_path / "a.bgra",
                tmp_path / "a.mp4",
                width=64,
                height=64,
                fps=10,
                overlay="critical | 0f03cdc9",
            )
        )
        assert "text=critical | 0f03cdc9" in text
        assert "timeoverlay" in text

    def test_the_keyframe_interval_follows_the_frame_rate(self, tmp_path):
        text = command(
            encoder.view_pipeline(
                tmp_path / "a.bgra", tmp_path / "a.mp4", width=64, height=64, fps=10, overlay=""
            )
        )
        assert "key-int-max=10" in text


class TestComparePipeline:
    def test_three_panes_are_placed_side_by_side(self, tmp_path):
        text = command(
            encoder.compare_pipeline(
                tmp_path / "o.bgra",
                tmp_path / "e.bgra",
                tmp_path / "c.bgra",
                tmp_path / "out.mp4",
                width=1280,
                height=720,
                fps=10,
            )
        )
        assert "sink_0::xpos=0" in text
        assert f"sink_1::xpos={encoder.PANE_WIDTH}" in text
        assert f"sink_2::xpos={encoder.PANE_WIDTH * 2}" in text
        assert f"width={encoder.PANE_WIDTH * 3},height={encoder.PANE_HEIGHT}" in text

    def test_the_simulator_views_are_scaled_down_and_the_original_is_not(self, tmp_path):
        text = command(
            encoder.compare_pipeline(
                tmp_path / "o.bgra",
                tmp_path / "e.bgra",
                tmp_path / "c.bgra",
                tmp_path / "out.mp4",
                width=1280,
                height=720,
                fps=10,
            )
        )
        assert text.count("videoscale") == 2

    def test_each_pane_is_labelled(self, tmp_path):
        text = command(
            encoder.compare_pipeline(
                tmp_path / "o.bgra",
                tmp_path / "e.bgra",
                tmp_path / "c.bgra",
                tmp_path / "out.mp4",
                width=640,
                height=360,
                fps=10,
            )
        )
        for label in ("text=original", "text=synthetic ego", "text=synthetic chase"):
            assert label in text


class TestExtractPipeline:
    def test_the_source_is_decoded_to_pane_sized_raw_frames(self, tmp_path):
        text = command(encoder.extract_pipeline(tmp_path / "src.mp4", tmp_path / "o.bgra", 10))
        assert "decodebin" in text
        assert (
            f"video/x-raw,format=BGRA,width={encoder.PANE_WIDTH},"
            f"height={encoder.PANE_HEIGHT},framerate=10/1" in text
        )


class TestSliceRaw:
    @staticmethod
    def _clip(path, frames: int) -> None:
        size = encoder.frame_bytes(encoder.PANE_WIDTH, encoder.PANE_HEIGHT)
        path.write_bytes(b"".join(bytes([index % 256]) * size for index in range(frames)))

    def test_it_copies_exactly_the_segment_frames(self, tmp_path):
        source, out = tmp_path / "full.bgra", tmp_path / "cut.bgra"
        self._clip(source, 100)
        assert encoder.slice_raw(source, out, start_s=4.0, duration_s=5.0, fps=10) == 50
        size = encoder.frame_bytes(encoder.PANE_WIDTH, encoder.PANE_HEIGHT)
        assert out.stat().st_size == 50 * size

    def test_it_starts_at_the_right_frame(self, tmp_path):
        source, out = tmp_path / "full.bgra", tmp_path / "cut.bgra"
        self._clip(source, 100)
        encoder.slice_raw(source, out, start_s=4.0, duration_s=1.0, fps=10)
        assert out.read_bytes()[0] == 40

    def test_a_short_trailing_segment_is_clamped_to_what_exists(self, tmp_path):
        source, out = tmp_path / "full.bgra", tmp_path / "cut.bgra"
        self._clip(source, 99)
        assert encoder.slice_raw(source, out, start_s=8.0, duration_s=5.0, fps=10) == 19

    def test_a_segment_past_the_end_is_an_error(self, tmp_path):
        source, out = tmp_path / "full.bgra", tmp_path / "cut.bgra"
        self._clip(source, 10)
        with pytest.raises(encoder.EncodingError, match="outside the decoded clip"):
            encoder.slice_raw(source, out, start_s=30.0, duration_s=5.0, fps=10)


class TestRunPipeline:
    def test_a_failing_pipeline_is_reported_not_swallowed(self):
        with pytest.raises(encoder.EncodingError, match="probe"):
            encoder.run_pipeline(["false"], what="probe")
