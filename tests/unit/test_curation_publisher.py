"""Unit tests for VLMKafkaSignalPublisher P2-4 curation wiring.

Mocks element, Scout, and Aggregator — no GPU, GStreamer, or Kafka required.
Covers: legacy path, Scout lazy-init, topic routing, N=1 fallback,
        message format, and per-segment resource release.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.scouts.aggregator import BestOfNAggregator
from src.scouts.base import ScoutConfig, ScoutReport

# Minimal valid DNA v0.1 CoT output. Contains "vehicle_car" so the aggregator
# score > 0 when inventory has {"car": N} (substring match).
_VALID_DNA_COT_OUTPUT = """\
Reasoning: clear day, primary road, one vehicle_car tailing at mid range, ego cruising.

```json
{
  "dna_version": "0.1.0",
  "clip_id": "00000000-0000-0000-0000-000000000000",
  "timestamp_range": {"start_s": 0, "end_s": 5},
  "odd": {"weather": "clear", "lighting": "day", "sensor_fidelity": ["clean"]},
  "topology": {"road_type": "primary", "lane_event": "normal", "intersection_type": "none"},
  "actor_dynamics": [
    {"actor_class": "vehicle_car", "state": "tailing", "distance_bucket": "mid",
     "confidence": 0.9, "grounded_by_yolo26": false}
  ],
  "planner_logic": {"ego_maneuver": "cruise", "risk_level": "nominal"},
  "confidence": {"overall": 0.9, "scout_agreement": 1.0, "hallucination_flags": []},
  "provenance": {
    "scout_models": ["cosmos-reason2-8b"], "scout_prompt_hash": "abcd1234",
    "pipeline_version": "p2-6", "is_synthetic": false,
    "reference_standards": ["ASAM OSI v3.x"]
  }
}
```"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scout_config(n: int = 3) -> ScoutConfig:
    return ScoutConfig(
        temperatures=[0.3, 0.5, 0.7],
        seeds={0.3: 42, 0.5: 43, 0.7: 44},
        n=n,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        engine_backend="gstnvvllmvlm_api",
        kafka_topic_scouted="curation.clip.scouted",
        kafka_topic_needs_review="curation.clip.needs_review",
    )


def _make_report(
    text: str = "car detected on road",
    temperature: float = 0.7,
    partial: bool = False,
) -> ScoutReport:
    return ScoutReport(
        text=text,
        temperature=temperature,
        seed=44,
        latency_ms=312.4,
        partial_sampling=partial,
    )


def _make_ctx(last_inventory=None, last_inputs=None):
    ctx = MagicMock()
    ctx.last_inventory = last_inventory if last_inventory is not None else {}
    ctx.last_inputs = (
        last_inputs if last_inputs is not None else {"prompt": "t", "multi_modal_data": {}}
    )
    return ctx


def _make_element(stream_id: int = 0, last_inventory=None, last_inputs=None, llm=None):
    ctx = _make_ctx(last_inventory=last_inventory, last_inputs=last_inputs)
    element = MagicMock()
    element.get_llm.return_value = llm if llm is not None else MagicMock()
    element.stream_contexts = {stream_id: ctx}
    return element


def _make_publisher(aggregator=None, scout_config=None, source_map=None):
    from vllm_ds_app_kafka_publish import VLMKafkaSignalPublisher

    return VLMKafkaSignalPublisher(
        {},
        "default-topic",
        dry_run=True,
        aggregator=aggregator,
        scout_config=scout_config,
        source_map=source_map,
    )


def _invoke(
    publisher,
    element=None,
    stream_id: int = 0,
    start: float = 0.0,
    end: float = 5.0,
    text: str = "scene description",
):
    publisher.on_vlm_result(element, stream_id, start, end, text)


# ---------------------------------------------------------------------------
# Legacy path (no aggregator) — backward-compatible behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyPath:
    def test_result_collected(self):
        pub = _make_publisher()
        _invoke(pub)
        assert len(pub._collected_results) == 1

    def test_no_curation_section_in_message(self):
        pub = _make_publisher()
        _invoke(pub)
        assert "curation" not in pub._collected_results[0]

    def test_uses_default_topic(self, capsys):
        pub = _make_publisher()
        _invoke(pub)
        assert "default-topic" in capsys.readouterr().out

    def test_result_text_stored_verbatim(self):
        pub = _make_publisher()
        _invoke(pub, text="raw vlm output")
        assert pub._collected_results[0]["result"] == "raw vlm output"


# ---------------------------------------------------------------------------
# Publisher init with aggregator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublisherInitWithAggregator:
    def test_aggregator_stored(self):
        agg = BestOfNAggregator()
        pub = _make_publisher(aggregator=agg, scout_config=_scout_config())
        assert pub._aggregator is agg

    def test_scout_config_stored(self):
        cfg = _scout_config()
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=cfg)
        assert pub._scout_config is cfg

    def test_scout_initially_none(self):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        assert pub._scout is None

    def test_partial_count_initially_zero(self):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        assert pub._partial_count == 0


# ---------------------------------------------------------------------------
# Scout lazy initialisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoutLazyInit:
    def test_scout_created_on_first_call(self):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        best = _make_report()
        with patch("src.scouts.cosmos_reason.CosmosReasonScout") as MockScout:
            mock_instance = MagicMock()
            mock_instance.sample.return_value = [best]
            MockScout.return_value = mock_instance
            _invoke(pub, element=_make_element())
        MockScout.assert_called_once()

    def test_scout_not_recreated_on_subsequent_calls(self):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        best = _make_report()
        with patch("src.scouts.cosmos_reason.CosmosReasonScout") as MockScout:
            mock_instance = MagicMock()
            mock_instance.sample.return_value = [best]
            MockScout.return_value = mock_instance
            _invoke(pub, element=_make_element())
            _invoke(pub, element=_make_element())
        assert MockScout.call_count == 1

    def test_scout_stays_none_when_llm_is_none(self):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        element = _make_element(llm=None)
        element.get_llm.return_value = None
        _invoke(pub, element=element)
        assert pub._scout is None


# ---------------------------------------------------------------------------
# Kafka topic routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKafkaTopicRouting:
    def _pub_with_mock_scout(self, reports):
        """Return publisher with pre-initialised mock Scout returning given reports."""
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        mock_scout = MagicMock()
        mock_scout.sample.return_value = reports
        pub._scout = mock_scout
        return pub

    def test_success_routes_to_scouted_topic(self, capsys):
        # Valid DNA JSON with vehicle_car → score > 0 for {"car": 2}; not partial → scouted
        best = _make_report(text=_VALID_DNA_COT_OUTPUT, partial=False)
        pub = self._pub_with_mock_scout([best])
        _invoke(pub, element=_make_element(last_inventory={"car": 2}))
        assert "curation.clip.scouted" in capsys.readouterr().out

    def test_partial_routes_to_needs_review_topic(self, capsys):
        partial = _make_report(partial=True)
        pub = self._pub_with_mock_scout([partial])
        _invoke(pub, element=_make_element())
        assert "curation.clip.needs_review" in capsys.readouterr().out

    def test_zero_grounding_routes_to_needs_review(self, capsys):
        # Report text has no "airplane" → score=0, inventory non-empty → zero_grounding
        best = _make_report(text="highway scene with traffic", partial=False)
        pub = self._pub_with_mock_scout([best])
        _invoke(pub, element=_make_element(last_inventory={"airplane": 1}))
        assert "curation.clip.needs_review" in capsys.readouterr().out

    def test_zero_grounding_reason_field(self):
        best = _make_report(text="highway scene with traffic", partial=False)
        pub = self._pub_with_mock_scout([best])
        _invoke(pub, element=_make_element(last_inventory={"airplane": 1}))
        assert pub._collected_results[0]["curation"]["reason"] == "zero_grounding"

    def test_partial_reason_field(self):
        partial = _make_report(partial=True)
        pub = self._pub_with_mock_scout([partial])
        _invoke(pub, element=_make_element())
        assert pub._collected_results[0]["curation"]["reason"] == "partial_batch"

    def test_success_reason_is_none(self):
        best = _make_report(text=_VALID_DNA_COT_OUTPUT, partial=False)
        pub = self._pub_with_mock_scout([best])
        _invoke(pub, element=_make_element(last_inventory={"car": 1}))
        assert pub._collected_results[0]["curation"]["reason"] is None

    def test_empty_inventory_never_triggers_zero_grounding(self):
        best = _make_report(text=_VALID_DNA_COT_OUTPUT, partial=False)
        pub = self._pub_with_mock_scout([best])
        _invoke(pub, element=_make_element(last_inventory={}))
        msg = pub._collected_results[0]
        assert msg["curation"]["needs_review"] is False
        assert msg["curation"]["reason"] is None


# ---------------------------------------------------------------------------
# N=1 consecutive-failure fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestN1Fallback:
    def _pub_always_partial(self):
        cfg = _scout_config()
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=cfg)
        mock_scout = MagicMock()
        mock_scout.sample.return_value = [_make_report(partial=True)]
        pub._scout = mock_scout
        return pub, cfg

    def test_n1_activated_after_3_consecutive_partials(self):
        pub, cfg = self._pub_always_partial()
        for _ in range(3):
            _invoke(pub, element=_make_element())
        assert cfg.n == 1

    def test_n1_not_activated_after_only_2_partials(self):
        pub, cfg = self._pub_always_partial()
        for _ in range(2):
            _invoke(pub, element=_make_element())
        assert cfg.n == 3  # unchanged

    def test_partial_count_increments_each_consecutive_failure(self):
        pub, _ = self._pub_always_partial()
        _invoke(pub, element=_make_element())
        assert pub._partial_count == 1
        _invoke(pub, element=_make_element())
        assert pub._partial_count == 2

    def test_partial_count_resets_on_success(self):
        cfg = _scout_config()
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=cfg)
        mock_scout = MagicMock()
        pub._scout = mock_scout

        # 2 failures
        mock_scout.sample.return_value = [_make_report(partial=True)]
        _invoke(pub, element=_make_element())
        _invoke(pub, element=_make_element())
        assert pub._partial_count == 2

        # 1 success (text mentions "car", inventory has "car" → score > 0)
        mock_scout.sample.return_value = [_make_report(text="car on road", partial=False)]
        _invoke(pub, element=_make_element(last_inventory={"car": 1}))
        assert pub._partial_count == 0

    def test_n1_not_activated_when_failures_not_consecutive(self):
        cfg = _scout_config()
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=cfg)
        mock_scout = MagicMock()
        pub._scout = mock_scout

        mock_scout.sample.return_value = [_make_report(partial=True)]
        _invoke(pub, element=_make_element())  # count=1
        _invoke(pub, element=_make_element())  # count=2

        mock_scout.sample.return_value = [_make_report(text="car", partial=False)]
        _invoke(pub, element=_make_element(last_inventory={"car": 1}))  # reset count=0

        mock_scout.sample.return_value = [_make_report(partial=True)]
        _invoke(pub, element=_make_element())  # count=1

        assert cfg.n == 3  # still 3 — never hit 3 consecutive

    def test_sample_called_with_n1_after_fallback(self):
        """After fallback activates, the next scout.sample() must observe n=1.

        Verifies the effect of cfg.n mutation on subsequent invocations —
        i.e. AC11 is not just a counter assertion but actually propagates to
        the Scout. ScoutConfig is shared by reference, so mutation in the
        publisher is visible to sample() at the next call site.
        """
        pub, cfg = self._pub_always_partial()
        for _ in range(3):
            _invoke(pub, element=_make_element())
        assert cfg.n == 1  # fallback engaged

        pub._scout.sample.reset_mock()
        _invoke(pub, element=_make_element())  # 4th invocation
        assert pub._scout.sample.call_count == 1
        passed_cfg = pub._scout.sample.call_args.args[2]
        assert passed_cfg is cfg
        assert passed_cfg.n == 1


# ---------------------------------------------------------------------------
# Kafka message format
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCurationMessageFormat:
    def _get_msg(self, text="car spotted", partial=False, inventory=None):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        mock_scout = MagicMock()
        mock_scout.sample.return_value = [_make_report(text=text, partial=partial)]
        pub._scout = mock_scout
        inv = inventory if inventory is not None else {}
        _invoke(pub, element=_make_element(last_inventory=inv))
        return pub._collected_results[0]

    def test_curation_section_present(self):
        assert "curation" in self._get_msg()

    def test_result_is_selected_report_text(self):
        msg = self._get_msg(text="selected best report")
        assert msg["result"] == "selected best report"

    def test_curation_temperature(self):
        assert "temperature" in self._get_msg()["curation"]

    def test_curation_seed(self):
        assert "seed" in self._get_msg()["curation"]

    def test_curation_latency_ms(self):
        assert "latency_ms" in self._get_msg()["curation"]

    def test_curation_n_samples(self):
        assert self._get_msg()["curation"]["n_samples"] == 1

    def test_curation_needs_review_false_on_success(self):
        msg = self._get_msg(text=_VALID_DNA_COT_OUTPUT, inventory={"car": 1})
        assert msg["curation"]["needs_review"] is False

    def test_curation_needs_review_true_on_partial(self):
        msg = self._get_msg(partial=True)
        assert msg["curation"]["needs_review"] is True

    def test_metadata_json_valid_present(self):
        assert "json_valid" in self._get_msg()["metadata"]

    def test_metadata_source(self):
        assert self._get_msg()["metadata"]["source"] == "vllm-ds-plugin"


# ---------------------------------------------------------------------------
# Per-segment resource release
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResourceRelease:
    def _run_curation(self, inventory=None, inputs=None):
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        mock_scout = MagicMock()
        mock_scout.sample.return_value = [_make_report()]
        pub._scout = mock_scout
        element = _make_element(
            last_inventory=inventory if inventory is not None else {},
            last_inputs=inputs or {"prompt": "t"},
        )
        ctx = element.stream_contexts[0]
        _invoke(pub, element=element)
        return ctx

    def test_last_inputs_set_to_none_after_scout(self):
        ctx = self._run_curation()
        assert ctx.last_inputs is None

    def test_last_inventory_cleared_after_scout(self):
        ctx = self._run_curation(inventory={"car": 2})
        assert ctx.last_inventory == {}


# ---------------------------------------------------------------------------
# P3-4: source_map per-stream source field propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSourceMapPropagation:
    def test_no_source_map_omits_source_fields(self):
        """Legacy path emits no source_clip_id / source_video_path when source_map absent."""
        pub = _make_publisher()
        _invoke(pub)
        msg = pub._collected_results[0]
        assert "source_clip_id" not in msg
        assert "source_video_path" not in msg

    def test_source_map_includes_source_clip_id(self):
        """source_clip_id appears when stream_id is in source_map."""
        pub = _make_publisher(source_map={0: ("00042", None)})
        _invoke(pub)
        msg = pub._collected_results[0]
        assert msg["source_clip_id"] == "00042"

    def test_source_map_includes_source_video_path(self):
        """source_video_path appears when stream_id is in source_map with a path."""
        pub = _make_publisher(source_map={0: ("00042", "session/00042/video/00042.mp4")})
        _invoke(pub)
        msg = pub._collected_results[0]
        assert msg["source_video_path"] == "session/00042/video/00042.mp4"

    def test_source_map_missing_stream_id_omits_fields(self):
        """Fields are absent when stream_id 0 is not in source_map (mapped for stream 1 only)."""
        pub = _make_publisher(source_map={1: ("00099", "/some/path.mp4")})
        _invoke(pub, stream_id=0)
        msg = pub._collected_results[0]
        assert "source_clip_id" not in msg
        assert "source_video_path" not in msg

    def test_curation_path_includes_source_clip_id(self):
        """Curation path also propagates source_clip_id from source_map."""
        pub = _make_publisher(
            aggregator=BestOfNAggregator(),
            scout_config=_scout_config(),
            source_map={0: ("00077", None)},
        )
        mock_scout = MagicMock()
        mock_scout.sample.return_value = [_make_report(text=_VALID_DNA_COT_OUTPUT)]
        pub._scout = mock_scout
        _invoke(pub, element=_make_element(last_inventory={"car": 1}))
        msg = pub._collected_results[0]
        assert msg["source_clip_id"] == "00077"

    def test_curation_path_omits_source_fields_when_no_map(self):
        """Curation path omits source fields when source_map absent."""
        pub = _make_publisher(aggregator=BestOfNAggregator(), scout_config=_scout_config())
        mock_scout = MagicMock()
        mock_scout.sample.return_value = [_make_report(text=_VALID_DNA_COT_OUTPUT)]
        pub._scout = mock_scout
        _invoke(pub, element=_make_element(last_inventory={"car": 1}))
        msg = pub._collected_results[0]
        assert "source_clip_id" not in msg
        assert "source_video_path" not in msg
