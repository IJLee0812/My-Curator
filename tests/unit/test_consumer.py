"""Tests for src/consumer.py — message formatting and argument parsing."""

import json

import pytest

from consumer import _build_arg_parser, _format_result_text


class TestFormatResultText:
    """Tests for consumer._format_result_text() — the real production function."""

    def test_json_result_pretty_printed(self):
        result = json.dumps({"scene_summary": "A car on the road.", "road_type": "highway"})
        output = _format_result_text(result)
        assert "Result (JSON):" in output
        assert "scene_summary" in output
        assert "highway" in output

    def test_json_with_nested_objects(self):
        result = json.dumps(
            {
                "scene_summary": "Urban intersection.",
                "key_objects": [
                    {"type": "vehicle", "description": "sedan turning left"},
                    {"type": "pedestrian", "description": "crossing"},
                ],
            }
        )
        output = _format_result_text(result)
        assert "Result (JSON):" in output
        assert "sedan turning left" in output
        assert "pedestrian" in output

    def test_raw_text_short(self):
        output = _format_result_text("Simple scene description.")
        assert "Result: Simple scene description." in output
        assert "Result (JSON):" not in output

    def test_raw_text_long_wraps(self):
        long_text = "A" * 250
        output = _format_result_text(long_text)
        assert "Result:" in output
        lines = [ln for ln in output.strip().split("\n") if ln.strip().startswith("A")]
        assert len(lines) >= 2

    def test_empty_json_object(self):
        output = _format_result_text("{}")
        assert "Result (JSON):" in output

    def test_invalid_json_falls_back_to_raw(self):
        output = _format_result_text("{invalid json content")
        assert "Result (JSON):" not in output
        assert "Result:" in output

    def test_unicode_content_preserved(self):
        result = json.dumps({"scene_summary": "교차로 장면."}, ensure_ascii=False)
        output = _format_result_text(result)
        assert "Result (JSON):" in output
        assert "교차로" in output


class TestConsumerArgParser:
    """Tests for consumer._build_arg_parser() — the real argparse configuration."""

    def test_default_timeout(self):
        args = _build_arg_parser().parse_args([])
        assert args.timeout == 300000

    def test_timeout_zero_parsed_correctly(self):
        args = _build_arg_parser().parse_args(["--timeout", "0"])
        timeout_ms = None if args.timeout == 0 else args.timeout
        assert timeout_ms is None

    def test_timeout_nonzero_preserved(self):
        args = _build_arg_parser().parse_args(["--timeout", "30000"])
        timeout_ms = None if args.timeout == 0 else args.timeout
        assert timeout_ms == 30000

    def test_default_no_reset(self):
        args = _build_arg_parser().parse_args([])
        assert args.reset is False

    def test_reset_flag_activates(self):
        args = _build_arg_parser().parse_args(["--reset"])
        assert args.reset is True

    def test_default_earliest_offset(self):
        args = _build_arg_parser().parse_args([])
        assert args.from_latest is False

    def test_from_latest_flag_activates(self):
        args = _build_arg_parser().parse_args(["--from-latest"])
        assert args.from_latest is True

    def test_default_broker(self):
        args = _build_arg_parser().parse_args([])
        assert args.broker == "localhost:9092"

    def test_default_topic(self):
        args = _build_arg_parser().parse_args([])
        assert args.topic == "vlm-results"

    def test_custom_broker(self):
        args = _build_arg_parser().parse_args(["--broker", "kafka:29092"])
        assert args.broker == "kafka:29092"

    def test_custom_topic(self):
        args = _build_arg_parser().parse_args(["--topic", "my-topic"])
        assert args.topic == "my-topic"
