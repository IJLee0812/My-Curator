"""Tests for plugin/config_loader.py — YAML loading, defaults, singleton."""

import pytest
from config_loader import Config, get_config, reload_config


class TestConfigDefaults:
    """All defaults when config is empty or missing."""

    def test_empty_dict_gives_defaults(self, empty_config_yaml):
        cfg = Config(empty_config_yaml)
        assert cfg.model_path == "nvidia/Cosmos-Reason2-8B"
        assert cfg.max_model_len == 20480
        assert cfg.trust_remote_code is True
        assert cfg.gpu_memory_utilization == 0.4
        assert cfg.enforce_eager is False
        assert cfg.gpu_id == 0
        assert cfg.video_mode == 1
        assert cfg.tensor_format == "pytorch"

    def test_segment_defaults(self, empty_config_yaml):
        cfg = Config(empty_config_yaml)
        assert cfg.segment_length_sec == 10
        assert cfg.overlap_sec == 0
        assert cfg.subsample_interval == 1
        assert cfg.selection_fps == 1

    def test_inference_defaults(self, empty_config_yaml):
        cfg = Config(empty_config_yaml)
        assert cfg.user_prompt == "Describe the scene in detail."
        assert cfg.system_prompt is None
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.7
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.repetition_penalty is None
        assert cfg.stream_prompts == {}

    def test_pipeline_defaults(self, empty_config_yaml):
        cfg = Config(empty_config_yaml)
        assert cfg.queue_maxsize == 20
        assert cfg.max_wait_timeout == 300

    def test_video_defaults(self, empty_config_yaml):
        cfg = Config(empty_config_yaml)
        assert cfg.default_fps == (30, 1)

    def test_detection_hints_defaults(self, empty_config_yaml):
        cfg = Config(empty_config_yaml)
        assert cfg.detection_hints_enabled is False
        assert cfg.detection_hints_min_confidence == 0.3


class TestConfigCustomValues:
    """Custom values loaded from YAML."""

    def test_model_properties(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert cfg.model_path == "/workspace/models/hub/test-model"
        assert cfg.max_model_len == 4096
        assert cfg.gpu_memory_utilization == 0.5
        assert cfg.gpu_id == 1

    def test_segment_properties(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert cfg.segment_length_sec == 5
        assert cfg.overlap_sec == 1
        assert cfg.selection_fps == 2

    def test_inference_properties(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.85
        assert cfg.top_k == 40
        assert cfg.repetition_penalty == 1.2
        assert cfg.system_prompt == "You are a test assistant."

    def test_user_prompt(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert "{num_frames}" in cfg.user_prompt
        assert "{stream_id}" in cfg.user_prompt

    def test_stream_prompts(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        prompts = cfg.stream_prompts
        assert 0 in prompts
        assert prompts[0]["user_prompt"] == "Stream 0 override prompt."
        assert prompts[0]["temperature"] == 0.1

    def test_pipeline_properties(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert cfg.queue_maxsize == 4
        assert cfg.max_wait_timeout == 60

    def test_video_fps(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert cfg.default_fps == (25, 1)

    def test_detection_hints_custom(self, sample_config_yaml):
        cfg = Config(sample_config_yaml)
        assert cfg.detection_hints_enabled is True
        assert cfg.detection_hints_min_confidence == 0.5

    def test_no_hints_section_gives_defaults(self, config_no_hints):
        cfg = Config(config_no_hints)
        assert cfg.detection_hints_enabled is False
        assert cfg.detection_hints_min_confidence == 0.3


class TestConfigSingleton:
    """Singleton pattern via get_config / reload_config."""

    def test_get_config_returns_same_instance(self, sample_config_yaml):
        c1 = get_config(sample_config_yaml)
        c2 = get_config()  # should return cached
        assert c1 is c2

    def test_reload_config_creates_new_instance(self, sample_config_yaml, empty_config_yaml):
        c1 = get_config(sample_config_yaml)
        c2 = reload_config(empty_config_yaml)
        assert c1 is not c2
        assert c2.model_path == "nvidia/Cosmos-Reason2-8B"  # default


class TestConfigEdgeCases:
    """Edge cases: nonexistent file, None path."""

    def test_nonexistent_file_uses_defaults(self):
        cfg = Config("/nonexistent/path/config.yaml")
        assert cfg.model_path == "nvidia/Cosmos-Reason2-8B"

    def test_none_path_uses_defaults(self):
        cfg = Config(None)
        # Should not raise; uses defaults
        assert cfg.max_tokens == 2048

    def test_top_k_int_conversion(self, tmp_path):
        """top_k YAML value is always converted to int."""
        content = "inference:\n  top_k: 50.0\n"
        p = tmp_path / "tk.yaml"
        p.write_text(content)
        cfg = Config(str(p))
        assert cfg.top_k == 50
        assert isinstance(cfg.top_k, int)
