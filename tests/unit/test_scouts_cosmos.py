"""Unit tests for CosmosReasonScout adapter.

All tests use mock LLM and mock SamplingParams — no GPU or vllm required.
"""

from unittest.mock import MagicMock

import pytest

from src.scouts.base import ScoutConfig, ScoutReport
from src.scouts.cosmos_reason import CosmosReasonScout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(n: int = 3) -> ScoutConfig:
    return ScoutConfig(
        temperatures=[0.3, 0.5, 0.7],
        seeds={0.3: 42, 0.5: 43, 0.7: 44},
        n=n,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        engine_backend="gstnvvllmvlm_api",
    )


def _mock_output(text: str) -> MagicMock:
    out = MagicMock()
    out.outputs = [MagicMock(text=text)]
    return out


def _make_scout(n_outputs: list[str] | None = None) -> tuple[CosmosReasonScout, MagicMock]:
    """Return (scout, mock_llm) with llm.generate pre-configured."""
    mock_llm = MagicMock()
    mock_sp = MagicMock(return_value=MagicMock())
    if n_outputs is not None:
        mock_llm.generate.return_value = [_mock_output(t) for t in n_outputs]
    scout = CosmosReasonScout(llm=mock_llm, _sampling_params_cls=mock_sp)
    return scout, mock_llm


# ---------------------------------------------------------------------------
# N=3 normal path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestN3NormalPath:
    def test_returns_three_reports_when_t0_result_provided(self):
        scout, mock_llm = _make_scout(["result_t05", "result_t07"])
        reports = scout.sample({}, {}, _config(n=3), t0_result="result_t03")

        assert len(reports) == 3
        assert not any(r.partial_sampling for r in reports)

    def test_t0_report_wraps_provided_text(self):
        scout, _ = _make_scout(["r1", "r2"])
        reports = scout.sample({}, {}, _config(n=3), t0_result="from_nvvllmvlm")
        assert reports[0].text == "from_nvvllmvlm"
        assert reports[0].temperature == 0.3
        assert reports[0].seed == 42
        assert reports[0].latency_ms == 0.0

    def test_temperature_order_preserved(self):
        scout, _ = _make_scout(["t05_text", "t07_text"])
        reports = scout.sample({}, {}, _config(n=3), t0_result="t03_text")
        assert [r.temperature for r in reports] == [0.3, 0.5, 0.7]

    def test_seeds_match_config(self):
        scout, _ = _make_scout(["x", "y"])
        reports = scout.sample({}, {}, _config(n=3), t0_result="z")
        assert reports[0].seed == 42
        assert reports[1].seed == 43
        assert reports[2].seed == 44

    def test_batch_called_once_for_extra_temperatures(self):
        scout, mock_llm = _make_scout(["r1", "r2"])
        scout.sample({}, {}, _config(n=3), t0_result="r0")
        mock_llm.generate.assert_called_once()

    def test_batch_receives_two_copies_of_inputs(self):
        scout, mock_llm = _make_scout(["r1", "r2"])
        inputs = {"prompt": "test_prompt"}
        scout.sample(inputs, {}, _config(n=3), t0_result="r0")

        prompts_arg = mock_llm.generate.call_args[0][0]
        assert len(prompts_arg) == 2
        assert all(p is inputs for p in prompts_arg)

    def test_sampling_params_created_for_t05_and_t07(self):
        mock_llm = MagicMock()
        mock_sp = MagicMock(return_value=MagicMock())
        mock_llm.generate.return_value = [_mock_output("r1"), _mock_output("r2")]
        scout = CosmosReasonScout(llm=mock_llm, _sampling_params_cls=mock_sp)

        scout.sample({}, {}, _config(n=3), t0_result="r0")

        assert mock_sp.call_count == 2
        call_kwargs = [c.kwargs for c in mock_sp.call_args_list]
        temps = {kw["temperature"] for kw in call_kwargs}
        seeds = {kw["seed"] for kw in call_kwargs}
        assert temps == {0.5, 0.7}
        assert seeds == {43, 44}


# ---------------------------------------------------------------------------
# N=1 fallback mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestN1FallbackMode:
    def test_returns_one_report(self):
        scout, mock_llm = _make_scout()
        reports = scout.sample({}, {}, _config(n=1), t0_result="only_one")
        assert len(reports) == 1

    def test_no_llm_generate_called(self):
        scout, mock_llm = _make_scout()
        scout.sample({}, {}, _config(n=1), t0_result="x")
        mock_llm.generate.assert_not_called()

    def test_partial_sampling_not_set_when_n1(self):
        scout, _ = _make_scout()
        reports = scout.sample({}, {}, _config(n=1), t0_result="x")
        assert not reports[0].partial_sampling


# ---------------------------------------------------------------------------
# Partial failure handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialFailure:
    def test_batch_exception_sets_partial_sampling_on_all(self):
        mock_llm = MagicMock()
        mock_sp = MagicMock(return_value=MagicMock())
        mock_llm.generate.side_effect = RuntimeError("GPU OOM")
        scout = CosmosReasonScout(llm=mock_llm, _sampling_params_cls=mock_sp)

        reports = scout.sample({}, {}, _config(n=3), t0_result="safe_t03")

        assert len(reports) == 1
        assert reports[0].partial_sampling is True

    def test_no_llm_with_t0_result_returns_partial(self):
        scout = CosmosReasonScout(llm=None)
        reports = scout.sample({}, {}, _config(n=3), t0_result="only_t03")
        assert len(reports) == 1
        assert reports[0].partial_sampling is True

    def test_no_llm_no_t0_result_returns_empty(self):
        scout = CosmosReasonScout(llm=None)
        reports = scout.sample({}, {}, _config(n=3), t0_result=None)
        assert reports == []

    def test_partial_flag_applied_to_t0_report_too(self):
        mock_llm = MagicMock()
        mock_sp = MagicMock(return_value=MagicMock())
        mock_llm.generate.side_effect = RuntimeError("fail")
        scout = CosmosReasonScout(llm=mock_llm, _sampling_params_cls=mock_sp)

        reports = scout.sample({}, {}, _config(n=3), t0_result="r0")
        assert reports[0].partial_sampling is True


# ---------------------------------------------------------------------------
# Standalone mode (no t0_result pre-computed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStandaloneMode:
    def test_generates_t0_via_generate_one_when_no_t0_result(self):
        mock_llm = MagicMock()
        mock_sp = MagicMock(return_value=MagicMock())
        # First call: _generate_one for T=0.3; second call: _generate_batch for T=0.5+T=0.7
        mock_llm.generate.side_effect = [
            [_mock_output("standalone_t03")],
            [_mock_output("standalone_t05"), _mock_output("standalone_t07")],
        ]
        scout = CosmosReasonScout(llm=mock_llm, _sampling_params_cls=mock_sp)

        reports = scout.sample({}, {}, _config(n=3), t0_result=None)

        assert len(reports) == 3
        assert reports[0].text == "standalone_t03"
        assert reports[0].temperature == 0.3
        assert mock_llm.generate.call_count == 2

    def test_n1_standalone_calls_generate_once(self):
        mock_llm = MagicMock()
        mock_sp = MagicMock(return_value=MagicMock())
        mock_llm.generate.return_value = [_mock_output("only")]
        scout = CosmosReasonScout(llm=mock_llm, _sampling_params_cls=mock_sp)

        reports = scout.sample({}, {}, _config(n=1), t0_result=None)

        assert len(reports) == 1
        mock_llm.generate.assert_called_once()
