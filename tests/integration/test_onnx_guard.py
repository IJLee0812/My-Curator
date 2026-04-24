"""Integration tests for ONNX guard logic (check_onnx_exists).

check_onnx_exists() is extracted from main() in
vllm_ds_app_kafka_publish.py and lives in plugin/vlm_utils.py.
It parses an nvinfer config file for the onnx-file key and returns
the path if the file is missing, None otherwise.
"""

import os

from vlm_utils import check_onnx_exists


def _write_config(tmp_path, content: str) -> str:
    p = tmp_path / "config_infer.txt"
    p.write_text(content)
    return str(p)


class TestCheckOnnxExists:
    def test_returns_none_when_onnx_exists(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("fake onnx")
        cfg = _write_config(tmp_path, f"onnx-file={onnx}\n")
        assert check_onnx_exists(cfg) is None

    def test_returns_path_when_onnx_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent.onnx")
        cfg = _write_config(tmp_path, f"onnx-file={missing}\n")
        result = check_onnx_exists(cfg)
        assert result == missing

    def test_returns_none_when_no_onnx_line(self, tmp_path):
        cfg = _write_config(tmp_path, "batch-size=1\nmodel-engine-file=yolo.engine\n")
        assert check_onnx_exists(cfg) is None

    def test_returns_none_for_nonexistent_config(self, tmp_path):
        missing_cfg = str(tmp_path / "no_such_config.txt")
        assert check_onnx_exists(missing_cfg) is None

    def test_ignores_lines_before_onnx_line(self, tmp_path):
        onnx = tmp_path / "real.onnx"
        onnx.write_text("data")
        cfg = _write_config(
            tmp_path,
            f"batch-size=1\nother-key=value\nonnx-file={onnx}\n",
        )
        assert check_onnx_exists(cfg) is None

    def test_handles_whitespace_around_equals(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("data")
        cfg = _write_config(tmp_path, f"onnx-file = {onnx}\n")
        # regex matches onnx-file\s*=\s*, so spaces around = are matched
        assert check_onnx_exists(cfg) is None

    def test_returns_missing_path_string(self, tmp_path):
        path = "/nonexistent/deeply/nested/model.onnx"
        cfg = _write_config(tmp_path, f"onnx-file={path}\n")
        assert check_onnx_exists(cfg) == path
