"""Integration tests for VLMKafkaSignalPublisher (dry-run mode).

GStreamer is mocked by tests/integration/conftest.py so these tests run
on the host as well as inside Docker.  Kafka is not required — dry_run=True
routes output to console.
"""

import json

import pytest


@pytest.fixture
def publisher():
    from vllm_ds_app_kafka_publish import VLMKafkaSignalPublisher

    return VLMKafkaSignalPublisher({}, "test-topic", dry_run=True)


class TestVLMKafkaSignalPublisherInit:
    def test_dry_run_no_producer(self, publisher):
        assert publisher.producer is None

    def test_dry_run_flag_set(self, publisher):
        assert publisher.dry_run is True

    def test_initial_counters_zero(self, publisher):
        assert publisher.messages_sent == 0
        assert publisher.messages_failed == 0

    def test_collected_results_empty(self, publisher):
        assert publisher._collected_results == []

    def test_topic_stored(self, publisher):
        assert publisher.topic == "test-topic"


class TestOnVlmResult:
    def test_message_appended_to_collected(self, publisher):
        publisher.on_vlm_result(None, 0, 10.0, 15.0, "scene description")
        assert len(publisher._collected_results) == 1

    def test_stream_id_in_message(self, publisher):
        publisher.on_vlm_result(None, 3, 0.0, 5.0, "test")
        msg = publisher._collected_results[0]
        assert msg["stream_id"] == 3

    def test_segment_times(self, publisher):
        publisher.on_vlm_result(None, 0, 10.0, 15.0, "test")
        seg = publisher._collected_results[0]["segment"]
        assert seg["start_time"] == 10.0
        assert seg["end_time"] == 15.0

    def test_segment_duration(self, publisher):
        publisher.on_vlm_result(None, 0, 10.0, 15.0, "test")
        assert publisher._collected_results[0]["segment"]["duration"] == pytest.approx(5.0)

    def test_result_text_stored(self, publisher):
        publisher.on_vlm_result(None, 0, 0.0, 5.0, '{"scene": "highway"}')
        assert publisher._collected_results[0]["result"] == '{"scene": "highway"}'

    def test_metadata_fields(self, publisher):
        publisher.on_vlm_result(None, 0, 0.0, 5.0, "text")
        meta = publisher._collected_results[0]["metadata"]
        assert meta["source"] == "vllm-ds-plugin"
        assert meta["version"] == "1.0"

    def test_multiple_results_accumulate(self, publisher):
        publisher.on_vlm_result(None, 0, 0.0, 5.0, "first")
        publisher.on_vlm_result(None, 1, 5.0, 10.0, "second")
        assert len(publisher._collected_results) == 2

    def test_json_valid_false_for_plain_text(self, publisher):
        publisher.on_vlm_result(None, 0, 0.0, 5.0, "just a plain sentence")
        assert publisher._collected_results[0]["metadata"]["json_valid"] is False

    def test_json_valid_false_for_schema_mismatch(self, publisher):
        """Parseable JSON that misses required DrivingSceneResult fields is invalid."""
        publisher.on_vlm_result(None, 0, 0.0, 5.0, '{"scene_summary": "x"}')
        assert publisher._collected_results[0]["metadata"]["json_valid"] is False

    def test_json_valid_true_for_valid_schema(self, publisher):
        payload = json.dumps(
            {
                "scene_summary": "An intersection with light traffic.",
                "road_type": "intersection",
                "road_features": {
                    "num_lanes": 4,
                    "lane_markings": "dashed_white",
                    "road_surface": "dry_asphalt",
                    "road_condition": "good",
                },
                "weather": "clear",
                "visibility": "good",
                "traffic_density": "moderate",
                "key_objects": [{"type": "vehicle", "description": "sedan"}],
                "ego_vehicle": {"action": "stopped", "estimated_speed": "stationary"},
                "potential_risks": [],
            }
        )
        publisher.on_vlm_result(None, 0, 0.0, 5.0, payload)
        assert publisher._collected_results[0]["metadata"]["json_valid"] is True

    def test_json_valid_true_through_code_fence(self, publisher):
        """Fenced ```json ... ``` output still validates."""
        payload = {
            "scene_summary": "x",
            "road_type": "urban_road",
            "road_features": {
                "num_lanes": 2,
                "lane_markings": "none",
                "road_surface": "concrete",
                "road_condition": "good",
            },
            "weather": "clear",
            "visibility": "good",
            "traffic_density": "sparse",
            "key_objects": [],
            "ego_vehicle": {"action": "stopped", "estimated_speed": "stationary"},
            "potential_risks": [],
        }
        text = "```json\n" + json.dumps(payload) + "\n```"
        publisher.on_vlm_result(None, 0, 0.0, 5.0, text)
        assert publisher._collected_results[0]["metadata"]["json_valid"] is True


class TestPublishDryRun:
    def test_increments_messages_sent(self, publisher, capsys):
        publisher.publish({"test": True}, 0)
        assert publisher.messages_sent == 1

    def test_prints_dry_run_header(self, publisher, capsys):
        publisher.publish({"key": "value"}, 0)
        out = capsys.readouterr().out
        assert "Dry-Run" in out

    def test_prints_topic(self, publisher, capsys):
        publisher.publish({"key": "value"}, 0)
        out = capsys.readouterr().out
        assert "test-topic" in out

    def test_prints_partition_key(self, publisher, capsys):
        publisher.publish({"key": "value"}, 2)
        out = capsys.readouterr().out
        assert "stream_2" in out

    def test_message_json_in_output(self, publisher, capsys):
        msg = {"stream_id": 0, "result": "scene"}
        publisher.publish(msg, 0)
        out = capsys.readouterr().out
        # Value is printed as indent=2 JSON; extract everything after "Value: "
        # up to the trailing separator line
        value_section = out.split("Value: ", 1)[1]
        json_text = value_section.rsplit("\n" + "=" * 80, 1)[0].strip()
        parsed = json.loads(json_text)
        assert parsed["stream_id"] == 0


class TestClose:
    def test_close_prints_stats(self, publisher, capsys):
        publisher.messages_sent = 3
        publisher.messages_failed = 1
        publisher.close()
        out = capsys.readouterr().out
        assert "3" in out
        assert "1" in out
