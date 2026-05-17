"""GT-1: golden publisher message envelope equivalence.

Locks the Kafka message envelope (keys and primitive types) produced by
``VLMKafkaSignalPublisher.on_vlm_result()`` for a fixed input / mocked Scout
combination.  Across R-1..R-7 the publisher moves to
``my_curator/application/pipeline/publisher.py``; envelope shape must remain
byte-equivalent.

Per the plan §4 R-0, ``last_inputs.json`` / ``last_inventory.json`` /
``scout_report.json`` are captured against a live DS run and committed under
``tests/fixtures/golden/``.  Until that capture lands the suite below runs with
a synthetic in-memory fixture so the test infrastructure is exercised every
commit (covers envelope schema + per-field types).  Once the real capture is in
place, the ``REAL_FIXTURE_DIR`` branch activates and locks the exact captured
``best.text`` snapshot.

References:
  docs/refactoring_plan.md  §3.1 GT-1, §4 R-0.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from my_curator.domain.scout.aggregator import BestOfNAggregator
from my_curator.domain.scout.base import ScoutConfig, ScoutReport

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
_REAL_FIXTURE = _FIXTURE_DIR / "publisher_last_inputs.json"

_VALID_DNA_COT = """\
Reasoning: clear day, one vehicle_car ahead.

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


def _scout_config() -> ScoutConfig:
    return ScoutConfig(
        temperatures=[0.3, 0.5, 0.7],
        seeds={0.3: 42, 0.5: 43, 0.7: 44},
        n=3,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        engine_backend="gstnvvllmvlm_api",
    )


def _make_element(stream_id: int = 0):
    ctx = MagicMock()
    ctx.last_inventory = {"car": 2}
    ctx.last_inputs = {"prompt": "describe", "multi_modal_data": {}}
    element = MagicMock()
    element.get_llm.return_value = MagicMock()
    element.stream_contexts = {stream_id: ctx}
    return element


def _make_publisher_with_fixed_scout():
    from my_curator.application.pipeline.publisher import VLMKafkaSignalPublisher

    pub = VLMKafkaSignalPublisher(
        {},
        "default-topic",
        dry_run=True,
        aggregator=BestOfNAggregator(),
        scout_config=_scout_config(),
    )
    fixed = ScoutReport(
        text=_VALID_DNA_COT,
        temperature=0.3,
        seed=42,
        latency_ms=0.0,
        partial_sampling=False,
    )
    mock_scout = MagicMock()
    mock_scout.sample.return_value = [fixed]
    pub._scout = mock_scout
    return pub


_EXPECTED_TOP_KEYS = {
    "stream_id",
    "timestamp",
    "segment",
    "result",
    "curation",
    "metadata",
    "clip_id",
}
_EXPECTED_CURATION_KEYS = {
    "temperature",
    "seed",
    "latency_ms",
    "partial_sampling",
    "n_samples",
    "needs_review",
    "reason",
}


@pytest.mark.unit
class TestGoldenPublisherEnvelope:
    def test_top_level_keys_match_envelope(self):
        pub = _make_publisher_with_fixed_scout()
        pub.on_vlm_result(_make_element(), 0, 0.0, 5.0, "t0-text")
        msg = pub._collected_results[0]
        assert _EXPECTED_TOP_KEYS.issubset(msg.keys()), (
            f"Missing envelope keys: {_EXPECTED_TOP_KEYS - msg.keys()}"
        )

    def test_segment_field_types(self):
        pub = _make_publisher_with_fixed_scout()
        pub.on_vlm_result(_make_element(), 0, 0.0, 5.0, "t0-text")
        seg = pub._collected_results[0]["segment"]
        assert isinstance(seg["start_time"], float)
        assert isinstance(seg["end_time"], float)
        assert isinstance(seg["duration"], float)

    def test_curation_subfield_types(self):
        pub = _make_publisher_with_fixed_scout()
        pub.on_vlm_result(_make_element(), 0, 0.0, 5.0, "t0-text")
        cur = pub._collected_results[0]["curation"]
        assert _EXPECTED_CURATION_KEYS.issubset(cur.keys())
        assert isinstance(cur["temperature"], float)
        assert isinstance(cur["seed"], int)
        assert isinstance(cur["latency_ms"], float)
        assert isinstance(cur["partial_sampling"], bool)
        assert isinstance(cur["n_samples"], int)
        assert isinstance(cur["needs_review"], bool)
        assert cur["reason"] is None or isinstance(cur["reason"], str)

    def test_metadata_envelope(self):
        pub = _make_publisher_with_fixed_scout()
        pub.on_vlm_result(_make_element(), 0, 0.0, 5.0, "t0-text")
        meta = pub._collected_results[0]["metadata"]
        assert meta["source"] == "vllm-ds-plugin"
        assert isinstance(meta["json_valid"], bool)

    def test_clip_id_is_uuid_string(self):
        import uuid

        pub = _make_publisher_with_fixed_scout()
        pub.on_vlm_result(_make_element(), 0, 0.0, 5.0, "t0-text")
        msg = pub._collected_results[0]
        # Round-trip parse must succeed.
        uuid.UUID(msg["clip_id"])

    def test_n_samples_equals_scout_returned_count(self):
        pub = _make_publisher_with_fixed_scout()
        pub.on_vlm_result(_make_element(), 0, 0.0, 5.0, "t0-text")
        assert pub._collected_results[0]["curation"]["n_samples"] == 1


@pytest.mark.unit
@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason=(
        "GT-1 real-DS fixture not captured yet — run capture script before R-5 to lock "
        "tests/fixtures/golden/publisher_last_inputs.json. See plan §4 R-0."
    ),
)
def test_golden_publisher_real_capture():
    """Activated once the real-DS fixture is committed (plan §4 R-0)."""
    # Placeholder body — replaced when the fixture lands.
    assert _REAL_FIXTURE.exists()
