"""Unit tests for scripts/download_model.py.

Covers:
- _filename_for(): YOLO26 and YOLOE filename construction, invalid model error
- download_model(): skip-if-exists, makedirs, urlretrieve call, URL construction,
  cleanup on failure
- _progress_hook(): progress bar rendering (smoke test)

All network I/O is mocked via unittest.mock — no real HTTP requests are made.
"""

import os
import sys

import pytest

# scripts/ is not on sys.path by default; add it so we can import directly.
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

import download_model as dm  # noqa: E402

# ---------------------------------------------------------------------------
# Tests: _filename_for()
# ---------------------------------------------------------------------------


class TestFilenameFor:
    def test_yolo26_small(self):
        assert dm._filename_for("yolo26", "s") == "yolo26s.pt"

    def test_yolo26_medium(self):
        assert dm._filename_for("yolo26", "m") == "yolo26m.pt"

    def test_yolo26_large(self):
        assert dm._filename_for("yolo26", "l") == "yolo26l.pt"

    def test_yoloe_small(self):
        assert dm._filename_for("yoloe", "s") == "yoloe-26s-seg.pt"

    def test_yoloe_medium(self):
        assert dm._filename_for("yoloe", "m") == "yoloe-26m-seg.pt"

    def test_yoloe_large(self):
        assert dm._filename_for("yoloe", "l") == "yoloe-26l-seg.pt"

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            dm._filename_for("yolov8", "m")

    def test_yoloe_filename_contains_seg_suffix(self):
        """YOLOE always uses the seg checkpoint filename."""
        name = dm._filename_for("yoloe", "m")
        assert name.endswith("-seg.pt")

    def test_yolo26_filename_does_not_contain_seg(self):
        name = dm._filename_for("yolo26", "m")
        assert "seg" not in name


# ---------------------------------------------------------------------------
# Tests: download_model() URL construction + file I/O (urllib mocked)
# ---------------------------------------------------------------------------


class TestDownloadModel:
    def test_skips_download_when_file_already_exists(self, tmp_path, capsys):
        existing = tmp_path / "yolo26m.pt"
        existing.write_text("weights")

        from unittest.mock import patch

        with patch("download_model.urllib.request.urlretrieve") as mock_retrieve:
            result = dm.download_model("yolo26", "m", str(tmp_path))

        mock_retrieve.assert_not_called()
        assert result == str(existing)
        assert "Already exists" in capsys.readouterr().out

    def test_calls_urlretrieve_with_correct_url(self, tmp_path):
        from unittest.mock import patch

        with (
            patch("download_model.urllib.request.urlretrieve") as mock_retrieve,
            patch("download_model.os.path.getsize", return_value=12_000_000),
        ):
            dm.download_model("yolo26", "m", str(tmp_path))

        called_url = mock_retrieve.call_args[0][0]
        assert called_url == f"{dm.BASE_URL}/yolo26m.pt"

    def test_calls_urlretrieve_with_correct_yoloe_url(self, tmp_path):
        from unittest.mock import patch

        with (
            patch("download_model.urllib.request.urlretrieve") as mock_retrieve,
            patch("download_model.os.path.getsize", return_value=15_000_000),
        ):
            dm.download_model("yoloe", "m", str(tmp_path))

        called_url = mock_retrieve.call_args[0][0]
        assert called_url == f"{dm.BASE_URL}/yoloe-26m-seg.pt"

    def test_output_path_inside_output_dir(self, tmp_path):
        from unittest.mock import patch

        with (
            patch("download_model.urllib.request.urlretrieve"),
            patch("download_model.os.path.getsize", return_value=10_000_000),
        ):
            result = dm.download_model("yolo26", "m", str(tmp_path))

        assert result == str(tmp_path / "yolo26m.pt")

    def test_creates_output_dir_when_missing(self, tmp_path):
        new_dir = tmp_path / "models" / "subdir"
        assert not new_dir.exists()

        from unittest.mock import patch

        with (
            patch("download_model.urllib.request.urlretrieve"),
            patch("download_model.os.path.getsize", return_value=1_000_000),
        ):
            dm.download_model("yolo26", "m", str(new_dir))

        assert new_dir.exists()

    def test_cleanup_partial_file_on_failure(self, tmp_path):
        """If urlretrieve raises, the partial output file is removed."""
        from unittest.mock import patch

        def _fail(url, dest, hook):
            # Create a partial file to simulate interrupted download
            open(dest, "w").close()
            raise OSError("network error")

        with (
            patch("download_model.urllib.request.urlretrieve", side_effect=_fail),
            patch("download_model.sys.exit"),
        ):
            dm.download_model("yolo26", "m", str(tmp_path))

        partial = tmp_path / "yolo26m.pt"
        assert not partial.exists()

    def test_progress_hook_passed_to_urlretrieve(self, tmp_path):
        """urlretrieve is called with _progress_hook as the reporthook."""
        from unittest.mock import patch

        with (
            patch("download_model.urllib.request.urlretrieve") as mock_retrieve,
            patch("download_model.os.path.getsize", return_value=5_000_000),
        ):
            dm.download_model("yoloe", "s", str(tmp_path))

        _, _, hook = mock_retrieve.call_args[0]
        assert hook is dm._progress_hook

    def test_returns_output_path_on_success(self, tmp_path):
        from unittest.mock import patch

        with (
            patch("download_model.urllib.request.urlretrieve"),
            patch("download_model.os.path.getsize", return_value=9_000_000),
        ):
            result = dm.download_model("yoloe", "l", str(tmp_path))

        assert result == str(tmp_path / "yoloe-26l-seg.pt")


# ---------------------------------------------------------------------------
# Tests: _progress_hook() — smoke tests (output is visual, not asserted deeply)
# ---------------------------------------------------------------------------


class TestProgressHook:
    def test_does_not_raise_with_known_size(self, capsys):
        dm._progress_hook(1, 1024, 102400)
        out = capsys.readouterr().out
        assert "%" in out

    def test_does_not_raise_with_zero_total_size(self, capsys):
        """total_size=0 should not cause a ZeroDivisionError."""
        dm._progress_hook(1, 1024, 0)  # must not raise

    def test_caps_percentage_at_100(self, capsys):
        """block_num * block_size > total_size should not exceed 100%."""
        dm._progress_hook(200, 1024, 1024)
        out = capsys.readouterr().out
        assert "100.0%" in out
