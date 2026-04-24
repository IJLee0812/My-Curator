"""Integration tests for the engine auto-detect logic (move_built_engine).

move_built_engine() is extracted from VLMKafkaApp.run() in
vllm_ds_app_kafka_publish.py and lives in plugin/vlm_utils.py.
These tests exercise it in isolation using tmp_path fixtures.
"""

import os
from pathlib import Path

from vlm_utils import move_built_engine


class TestMoveBuiltEngine:
    def test_returns_none_when_dest_is_none(self):
        assert move_built_engine(None) is None

    def test_returns_none_when_dest_already_exists(self, tmp_path):
        dest = tmp_path / "models" / "yolo26m.engine"
        dest.parent.mkdir(parents=True)
        dest.touch()
        assert move_built_engine(str(dest), cwd=str(tmp_path)) is None

    def test_returns_none_when_no_built_engine_in_cwd(self, tmp_path):
        dest = str(tmp_path / "models" / "yolo26m.engine")
        result = move_built_engine(dest, cwd=str(tmp_path))
        assert result is None

    def test_moves_engine_to_dest(self, tmp_path):
        src = tmp_path / "model_b1_gpu0_fp16.engine"
        src.write_text("fake engine")
        dest = str(tmp_path / "models" / "yolo26m.engine")

        result = move_built_engine(dest, cwd=str(tmp_path))

        assert result == dest
        assert os.path.isfile(dest)
        assert not src.exists()

    def test_creates_dest_directory(self, tmp_path):
        src = tmp_path / "model_b1_gpu0_fp16.engine"
        src.write_text("fake engine")
        dest = str(tmp_path / "deep" / "nested" / "yolo26m.engine")

        move_built_engine(dest, cwd=str(tmp_path))

        assert os.path.isfile(dest)

    def test_moves_onnx_pattern_engine(self, tmp_path):
        src = tmp_path / "yolo26m.onnx_b1_gpu0_fp16.engine"
        src.write_text("fake onnx engine")
        dest = str(tmp_path / "models" / "yolo26m.engine")

        result = move_built_engine(dest, cwd=str(tmp_path))

        assert result == dest
        assert os.path.isfile(dest)

    def test_returns_dest_path_on_success(self, tmp_path):
        src = tmp_path / "model_b1_gpu0_fp16.engine"
        src.write_text("engine content")
        dest = str(tmp_path / "out.engine")

        result = move_built_engine(dest, cwd=str(tmp_path))

        assert result == dest
