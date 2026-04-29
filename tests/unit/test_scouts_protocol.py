"""Protocol contract tests for the Scout abstraction layer.

All tests run with a mock engine — no GPU or vllm required.
"""

import pytest

from src.scouts.base import Scout, ScoutConfig, ScoutReport
from src.scouts.cosmos_reason import CosmosReasonScout

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config(**overrides) -> ScoutConfig:
    base = dict(
        temperatures=[0.3, 0.5, 0.7],
        seeds={0.3: 42, 0.5: 43, 0.7: 44},
        n=3,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        engine_backend="gstnvvllmvlm_api",
    )
    base.update(overrides)
    return ScoutConfig(**base)


# ---------------------------------------------------------------------------
# Scout Protocol structural checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoutProtocol:
    def test_cosmos_reason_scout_satisfies_scout_protocol(self):
        assert isinstance(CosmosReasonScout(), Scout)

    def test_arbitrary_class_with_sample_satisfies_protocol(self):
        class _Dummy:
            def sample(self, inputs, prompt_config, config, t0_result=None):
                return []

        assert isinstance(_Dummy(), Scout)

    def test_class_without_sample_does_not_satisfy_protocol(self):
        class _Bad:
            pass

        assert not isinstance(_Bad(), Scout)


# ---------------------------------------------------------------------------
# ScoutReport dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoutReport:
    def test_fields_set_correctly(self):
        r = ScoutReport(text='{"weather":"clear"}', temperature=0.3, seed=42, latency_ms=120.5)
        assert r.text == '{"weather":"clear"}'
        assert r.temperature == 0.3
        assert r.seed == 42
        assert r.latency_ms == 120.5
        assert r.partial_sampling is False

    def test_partial_sampling_flag_settable(self):
        r = ScoutReport(text="x", temperature=0.5, seed=43, latency_ms=0.0)
        r.partial_sampling = True
        assert r.partial_sampling is True


# ---------------------------------------------------------------------------
# ScoutConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoutConfig:
    def test_seed_for_known_temperature(self):
        cfg = _config()
        assert cfg.seed_for(0.3) == 42
        assert cfg.seed_for(0.5) == 43
        assert cfg.seed_for(0.7) == 44

    def test_seed_for_unknown_temperature_returns_default(self):
        cfg = _config()
        assert cfg.seed_for(0.99) == 42

    def test_n1_fallback_config(self):
        cfg = _config(n=1)
        assert cfg.n == 1

    def test_from_yaml_loads_correctly(self, tmp_path):
        yaml_content = """\
temperatures: [0.3, 0.5, 0.7]
seeds:
  0.3: 42
  0.5: 43
  0.7: 44
n: 3
max_tokens: 512
top_p: 0.95
top_k: 40
repetition_penalty: 1.05
engine:
  backend: gstnvvllmvlm_api
  base_url: ""
  model: cosmos-reason2-8b
  timeout_s: 20
"""
        cfg_file = tmp_path / "scout.yaml"
        cfg_file.write_text(yaml_content)
        cfg = ScoutConfig.from_yaml(str(cfg_file))

        assert cfg.temperatures == [0.3, 0.5, 0.7]
        assert cfg.n == 3
        assert cfg.max_tokens == 512
        assert cfg.top_p == 0.95
        assert cfg.top_k == 40
        assert cfg.repetition_penalty == 1.05
        assert cfg.engine_backend == "gstnvvllmvlm_api"
        assert cfg.engine_timeout_s == 20.0
        assert cfg.seed_for(0.3) == 42
        assert cfg.seed_for(0.5) == 43
        assert cfg.seed_for(0.7) == 44

    def test_from_yaml_missing_engine_section_uses_defaults(self, tmp_path):
        yaml_content = """\
temperatures: [0.3]
seeds:
  0.3: 42
n: 1
max_tokens: 256
top_p: 0.9
top_k: 50
repetition_penalty: 1.1
"""
        cfg_file = tmp_path / "scout_min.yaml"
        cfg_file.write_text(yaml_content)
        cfg = ScoutConfig.from_yaml(str(cfg_file))
        assert cfg.engine_backend == "gstnvvllmvlm_api"
        assert cfg.engine_timeout_s == 30.0

    def test_temperatures_sliced_to_n(self):
        cfg = _config(n=2)
        active = cfg.temperatures[: cfg.n]
        assert active == [0.3, 0.5]

    def test_from_yaml_loads_real_scout_yaml(self):
        import os

        yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "scout.yaml")
        cfg = ScoutConfig.from_yaml(yaml_path)
        assert cfg.n == 3
        assert cfg.temperatures == [0.3, 0.5, 0.7]
        assert cfg.engine_backend == "gstnvvllmvlm_api"
