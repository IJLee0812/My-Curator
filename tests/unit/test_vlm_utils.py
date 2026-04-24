"""Tests for plugin/vlm_utils.py — pure logic, no GStreamer/CUDA deps."""

import json
import os
from types import SimpleNamespace

import pytest
from vlm_utils import (
    VRU_CLASSES,
    YOLO26_CLASS_MAPPING,
    _bbox_to_zone,
    collect_detections,
    compute_sample_interval_ns,
    compute_step_ns,
    format_detection_hints,
    format_user_prompt,
    get_stream_config,
    is_segmentation_config,
    load_class_mapping,
    parse_nvinfer_config,
    parse_vlm_json,
    to_uri,
    validate_driving_scene_json,
)

# ── Helpers ──


def _make_frame(detections=None):
    """Create a SimpleNamespace mimicking BufferData with detections."""
    return SimpleNamespace(detections=detections or [])


def _make_detection(label, confidence, bbox):
    return {"label": label, "confidence": confidence, "bbox": bbox}


# ── YOLO26_CLASS_MAPPING ──


class TestYOLO26ClassMapping:
    def test_expected_classes(self):
        assert YOLO26_CLASS_MAPPING[0] == "Pedestrian"
        assert YOLO26_CLASS_MAPPING[2] == "Car"
        assert YOLO26_CLASS_MAPPING[7] == "Truck"
        assert YOLO26_CLASS_MAPPING[11] == "Trafficsign"

    def test_eight_classes(self):
        assert len(YOLO26_CLASS_MAPPING) == 8

    def test_missing_ids_not_present(self):
        for cid in (4, 6, 8, 10, 12, 79):
            assert cid not in YOLO26_CLASS_MAPPING


# ── format_detection_hints ──


class TestFormatDetectionHints:
    """Tests for the aggregated, VRU-zone detection-hint format."""

    def test_disabled_returns_empty(self):
        frames = [_make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))])]
        assert format_detection_hints(frames, enabled=False) == ""

    def test_empty_frames_returns_empty(self):
        assert format_detection_hints([], enabled=True) == ""

    def test_no_detections_returns_empty(self):
        frames = [_make_frame([]), _make_frame([])]
        assert format_detection_hints(frames, enabled=True) == ""

    def test_frames_without_detections_attr(self):
        """Frames without .detections attribute should be handled gracefully."""
        frame = SimpleNamespace()
        assert format_detection_hints([frame], enabled=True) == ""

    def test_header_format(self):
        frames = [_make_frame([_make_detection("Car", 0.9, (0, 0, 1, 1))])]
        result = format_detection_hints(frames, enabled=True)
        assert result.startswith("[Auxiliary object cues from YOLO26 - may be incomplete or noisy]")

    def test_non_vru_aggregated_count(self):
        """Non-VRU detections are aggregated as `label x total_count`."""
        frames = [
            _make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))]),
            _make_frame(
                [
                    _make_detection("Car", 0.85, (0.1, 0.2, 0.3, 0.4)),
                    _make_detection("Bus", 0.8, (0.5, 0.1, 0.7, 0.3)),
                ]
            ),
        ]
        result = format_detection_hints(frames, enabled=True)
        lines = result.split("\n")
        assert lines[1] == "Object counts across 2 frames: Car x 2, Bus x 1"

    def test_non_vru_sorted_by_count_desc(self):
        frames = [
            _make_frame(
                [
                    _make_detection("Car", 0.9, (0, 0, 1, 1)),
                    _make_detection("Car", 0.9, (0, 0, 1, 1)),
                    _make_detection("Truck", 0.9, (0, 0, 1, 1)),
                    _make_detection("Bus", 0.9, (0, 0, 1, 1)),
                    _make_detection("Bus", 0.9, (0, 0, 1, 1)),
                    _make_detection("Bus", 0.9, (0, 0, 1, 1)),
                ]
            )
        ]
        result = format_detection_hints(frames, enabled=True)
        counts_line = result.split("\n")[1]
        # Bus (3) > Car (2) > Truck (1)
        assert counts_line.index("Bus") < counts_line.index("Car") < counts_line.index("Truck")

    def test_vru_zone_bucketed(self):
        """Same VRU class at the same (zone, proximity) collapses into one bucket."""
        frames = [
            _make_frame([_make_detection("Pedestrian", 0.9, (0.05, 0.2, 0.15, 0.7))]),
            _make_frame([_make_detection("Pedestrian", 0.88, (0.05, 0.2, 0.15, 0.7))]),
            _make_frame([_make_detection("Motorcycle", 0.85, (0.45, 0.25, 0.55, 0.7))]),
        ]
        result = format_detection_hints(frames, enabled=True)
        vru_line = [ln for ln in result.split("\n") if ln.startswith("VRU positions")][0]
        assert "Pedestrian x 2 (left-edge, near)" in vru_line
        assert "Motorcycle x 1 (center, near)" in vru_line

    def test_vru_separate_buckets_for_different_zones(self):
        """Different lateral zones should yield separate VRU buckets."""
        frames = [
            _make_frame(
                [
                    _make_detection("Pedestrian", 0.9, (0.85, 0.4, 0.95, 0.6)),
                    _make_detection("Pedestrian", 0.9, (0.05, 0.4, 0.15, 0.6)),
                ]
            )
        ]
        result = format_detection_hints(frames, enabled=True)
        vru_line = [ln for ln in result.split("\n") if ln.startswith("VRU positions")][0]
        assert "right-edge" in vru_line
        assert "left-edge" in vru_line

    def test_vru_and_non_vru_coexist(self):
        frames = [
            _make_frame(
                [
                    _make_detection("Car", 0.9, (0.4, 0.3, 0.6, 0.7)),
                    _make_detection("Pedestrian", 0.85, (0.05, 0.2, 0.15, 0.7)),
                ]
            )
        ]
        result = format_detection_hints(frames, enabled=True)
        assert "Object counts across 1 frames: Car x 1" in result
        assert "VRU positions:" in result
        assert "Pedestrian" in result

    def test_frame_count_reflects_input_length(self):
        """Frame count in header equals len(frames), even when some are empty."""
        frames = [
            _make_frame([_make_detection("Car", 0.9, (0, 0, 1, 1))]),
            _make_frame([]),
            _make_frame([_make_detection("Bus", 0.9, (0, 0, 1, 1))]),
            _make_frame([]),
            _make_frame([]),
        ]
        result = format_detection_hints(frames, enabled=True)
        assert "across 5 frames" in result

    def test_only_vru_no_non_vru_line(self):
        frames = [_make_frame([_make_detection("Bike", 0.9, (0.3, 0.3, 0.45, 0.7))])]
        result = format_detection_hints(frames, enabled=True)
        assert "Object counts across" not in result
        assert "VRU positions:" in result

    def test_only_non_vru_no_vru_line(self):
        frames = [_make_frame([_make_detection("Trafficlight", 0.9, (0.4, 0.1, 0.5, 0.2))])]
        result = format_detection_hints(frames, enabled=True)
        assert "Object counts across 1 frames: Trafficlight x 1" in result
        assert "VRU positions:" not in result


class TestVruClasses:
    def test_vru_set_contains_expected_labels(self):
        assert {"Pedestrian", "Bike", "Motorcycle"} == VRU_CLASSES

    def test_all_vru_labels_present_in_yolo26_mapping(self):
        yolo26_labels = set(YOLO26_CLASS_MAPPING.values())
        assert VRU_CLASSES.issubset(yolo26_labels)


class TestBboxToZone:
    """Zone and proximity classification from normalized bbox."""

    @pytest.mark.parametrize(
        "x_center,expected_zone",
        [
            (0.05, "left-edge"),
            (0.19, "left-edge"),
            (0.2, "left"),  # exactly 0.2 is the boundary → left
            (0.35, "left"),
            (0.4, "center"),
            (0.5, "center"),
            (0.6, "right"),
            (0.75, "right"),
            (0.8, "right-edge"),
            (0.99, "right-edge"),
        ],
    )
    def test_zone_boundaries(self, x_center, expected_zone):
        bbox = (x_center - 0.01, 0.3, x_center + 0.01, 0.5)
        zone, _ = _bbox_to_zone(bbox)
        assert zone == expected_zone

    @pytest.mark.parametrize(
        "height,expected_prox",
        [
            (0.5, "near"),
            (0.36, "near"),
            (0.35, "near"),  # exactly 0.35 → still near (>= 0.35 threshold)
            (0.34, "mid"),
            (0.2, "mid"),
            (0.1, "mid"),  # exactly 0.1 → still mid
            (0.09, "far"),
            (0.02, "far"),
            (0.0, "far"),
        ],
    )
    def test_proximity_boundaries(self, height, expected_prox):
        # Center the bbox horizontally so zone is "center"
        bbox = (0.45, 0.5 - height / 2, 0.55, 0.5 + height / 2)
        _, prox = _bbox_to_zone(bbox)
        assert prox == expected_prox

    def test_combined_zone_and_proximity(self):
        # Large bbox on left edge → (left-edge, near)
        bbox = (0.02, 0.1, 0.15, 0.9)
        zone, prox = _bbox_to_zone(bbox)
        assert zone == "left-edge"
        assert prox == "near"


class TestParseVlmJson:
    def test_raw_json_parsed(self):
        data, err = parse_vlm_json('{"a": 1, "b": "x"}')
        assert err is None
        assert data == {"a": 1, "b": "x"}

    def test_code_fenced_json_parsed(self):
        text = '```json\n{"scene_summary": "test"}\n```'
        data, err = parse_vlm_json(text)
        assert err is None
        assert data == {"scene_summary": "test"}

    def test_code_fence_without_language_tag(self):
        text = '```\n{"x": 1}\n```'
        data, err = parse_vlm_json(text)
        assert err is None
        assert data == {"x": 1}

    def test_code_fence_with_leading_whitespace(self):
        text = '   ```json\n{"x": 1}\n```   '
        data, err = parse_vlm_json(text)
        assert err is None
        assert data == {"x": 1}

    def test_malformed_json_returns_error(self):
        data, err = parse_vlm_json("{not valid json")
        assert data is None
        assert err is not None
        assert "JSONDecode" in err

    def test_empty_input_returns_error(self):
        data, err = parse_vlm_json("")
        assert data is None
        assert err is not None

    def test_non_string_input_returns_error(self):
        data, err = parse_vlm_json(None)
        assert data is None
        assert err is not None

    def test_non_object_json_returns_error(self):
        """A top-level JSON array is not a valid VLM result."""
        data, err = parse_vlm_json("[1, 2, 3]")
        assert data is None
        assert err is not None
        assert "object" in err.lower()


class TestValidateDrivingSceneJson:
    """Schema validation against DrivingSceneResult."""

    def _valid_payload(self) -> dict:
        return {
            "scene_summary": "An urban intersection with moderate traffic.",
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
            "ego_vehicle": {
                "action": "stopped",
                "estimated_speed": "stationary",
            },
            "potential_risks": ["pedestrian crossing"],
        }

    def test_valid_payload(self):
        ok, err = validate_driving_scene_json(self._valid_payload())
        assert ok is True
        assert err is None

    def test_missing_required_field(self):
        payload = self._valid_payload()
        del payload["weather"]
        ok, err = validate_driving_scene_json(payload)
        assert ok is False
        assert err is not None

    def test_empty_key_objects_valid(self):
        """An empty key_objects list should still validate."""
        payload = self._valid_payload()
        payload["key_objects"] = []
        ok, err = validate_driving_scene_json(payload)
        assert ok is True

    def test_extra_fields_allowed(self):
        """extra='allow' keeps forward-compat on schema evolution."""
        payload = self._valid_payload()
        payload["future_field"] = "something"
        ok, err = validate_driving_scene_json(payload)
        assert ok is True

    def test_wrong_type_rejected(self):
        payload = self._valid_payload()
        payload["key_objects"] = "not a list"
        ok, err = validate_driving_scene_json(payload)
        assert ok is False

    def test_parse_and_validate_round_trip(self):
        text = "```json\n" + json.dumps(self._valid_payload()) + "\n```"
        data, perr = parse_vlm_json(text)
        assert perr is None
        ok, verr = validate_driving_scene_json(data)
        assert ok is True
        assert verr is None

    def test_import_error_fallback(self, monkeypatch):
        """Returns (False, error_msg) gracefully when output_schema is absent."""
        import sys

        monkeypatch.delitem(sys.modules, "output_schema", raising=False)
        monkeypatch.setitem(sys.modules, "output_schema", None)
        ok, err = validate_driving_scene_json(self._valid_payload())
        assert ok is False
        assert err is not None


# ── format_user_prompt ──


class TestFormatUserPrompt:
    def test_all_placeholders(self):
        prompt = (
            "Analyze {num_frames} frames from stream {stream_id} at {timestamps}. {detection_hints}"
        )
        result = format_user_prompt(prompt, stream_id=0, num_frames=5, timestamps="1.0s 2.0s")
        assert "5 frames" in result
        assert "stream 0" in result
        assert "1.0s 2.0s" in result

    def test_no_placeholders(self):
        prompt = "Describe the scene in detail."
        result = format_user_prompt(prompt, stream_id=0, num_frames=3, timestamps="0.0s")
        assert result == "Describe the scene in detail."

    def test_invalid_placeholder_returns_original(self):
        prompt = "Test {unknown_placeholder} here."
        result = format_user_prompt(prompt, stream_id=0, num_frames=1, timestamps="0.0s")
        assert result == prompt

    def test_detection_hints_injected(self):
        prompt = "Scene analysis.\n{detection_hints}\nDescribe."
        hints = "[Object detection hints from YOLO26]\nF0: Car(0.9)"
        result = format_user_prompt(
            prompt, stream_id=0, num_frames=1, timestamps="0.0s", detection_hints=hints
        )
        assert "YOLO26" in result
        assert "Car(0.9)" in result

    def test_empty_prompt(self):
        assert format_user_prompt("", stream_id=0, num_frames=0, timestamps="") == ""

    def test_detection_hints_default_empty(self):
        prompt = "Test {detection_hints} end."
        result = format_user_prompt(prompt, stream_id=0, num_frames=1, timestamps="0.0s")
        assert result == "Test  end."


# ── compute_step_ns ──


class TestComputeStepNs:
    def test_normal_case(self):
        assert compute_step_ns(10, 2) == 8_000_000_000

    def test_no_overlap(self):
        assert compute_step_ns(10, 0) == 10_000_000_000

    def test_overlap_equals_segment(self):
        # max(1, 10-10) = 1
        assert compute_step_ns(10, 10) == 1_000_000_000

    def test_overlap_exceeds_segment(self):
        # max(1, 5-10) = 1
        assert compute_step_ns(5, 10) == 1_000_000_000

    def test_negative_overlap(self):
        # max(1, 10-(-5)) = 15
        assert compute_step_ns(10, -5) == 15_000_000_000


# ── compute_sample_interval_ns ──


class TestComputeSampleIntervalNs:
    def test_fps_2(self):
        assert compute_sample_interval_ns(2) == 500_000_000

    def test_fps_1(self):
        assert compute_sample_interval_ns(1) == 1_000_000_000

    def test_fps_30(self):
        result = compute_sample_interval_ns(30)
        assert result == 33_333_333

    def test_fps_zero_disabled(self):
        assert compute_sample_interval_ns(0) is None

    def test_fps_negative_disabled(self):
        assert compute_sample_interval_ns(-1) is None


# ── get_stream_config ──


class TestGetStreamConfig:
    def test_override_exists(self):
        prompts = {0: {"user_prompt": "Override!", "temperature": 0.1}}
        assert get_stream_config(prompts, 0, "user_prompt", "default") == "Override!"

    def test_fallback_to_default(self):
        prompts = {0: {"user_prompt": "Override!"}}
        assert get_stream_config(prompts, 0, "temperature", 0.7) == 0.7

    def test_stream_not_present(self):
        prompts = {0: {"user_prompt": "Override!"}}
        assert get_stream_config(prompts, 1, "user_prompt", "default") == "default"

    def test_empty_prompts(self):
        assert get_stream_config({}, 0, "user_prompt", "default") == "default"

    def test_none_value_override(self):
        """If the override value is None, it should still be returned."""
        prompts = {0: {"system_prompt": None}}
        assert get_stream_config(prompts, 0, "system_prompt", "fallback") is None


# ── collect_detections ──


class TestCollectDetections:
    def _make_obj(self, class_id, confidence, left, top, width, height):
        return SimpleNamespace(
            class_id=class_id,
            confidence=confidence,
            rect_params=SimpleNamespace(left=left, top=top, width=width, height=height),
        )

    def test_known_class_id(self):
        objs = [self._make_obj(2, 0.9, 100, 200, 50, 60)]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 1
        assert result[0]["label"] == "Car"
        assert result[0]["confidence"] == 0.9

    def test_unknown_class_id_ignored(self):
        objs = [self._make_obj(4, 0.9, 100, 200, 50, 60)]  # class 4 not in mapping
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 0

    def test_below_min_confidence_filtered(self):
        objs = [self._make_obj(2, 0.2, 100, 200, 50, 60)]  # conf 0.2 < 0.3
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 0

    def test_normalized_bbox(self):
        objs = [self._make_obj(2, 0.9, 960, 540, 192, 108)]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        bbox = result[0]["bbox"]
        assert bbox[0] == 0.5  # x1 = 960/1920
        assert bbox[1] == 0.5  # y1 = 540/1080
        assert bbox[2] == 0.6  # x2 = (960+192)/1920
        assert bbox[3] == 0.6  # y2 = (540+108)/1080

    def test_zero_frame_dimensions(self):
        """frame_width=0 or frame_height=0 should not cause division by zero."""
        objs = [self._make_obj(2, 0.9, 100, 200, 50, 60)]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 0, 0)
        assert len(result) == 1  # should not crash

    def test_multiple_detections(self):
        objs = [
            self._make_obj(0, 0.9, 100, 200, 50, 60),  # Pedestrian
            self._make_obj(2, 0.8, 300, 400, 70, 80),  # Car
            self._make_obj(7, 0.7, 500, 600, 90, 100),  # Truck
        ]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 3
        labels = [d["label"] for d in result]
        assert "Pedestrian" in labels
        assert "Car" in labels
        assert "Truck" in labels

    def test_empty_object_items(self):
        result = collect_detections([], YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert result == []


# ── to_uri ──


class TestToUri:
    def test_absolute_path(self):
        result = to_uri("/workspace/video.mp4")
        assert result == "file:///workspace/video.mp4"

    def test_rtsp_passthrough(self):
        uri = "rtsp://192.168.1.100:8554/stream"
        assert to_uri(uri) == uri

    def test_http_passthrough(self):
        uri = "http://example.com/video.mp4"
        assert to_uri(uri) == uri

    def test_file_uri_passthrough(self):
        uri = "file:///workspace/video.mp4"
        assert to_uri(uri) == uri

    def test_relative_path_becomes_absolute(self):
        result = to_uri("video.mp4")
        assert result.startswith("file://")
        assert "/video.mp4" in result
        assert "://" in result


# ── load_class_mapping (YOLOE runtime labels, YOLO26 fallback) ──


class TestLoadClassMapping:
    def test_missing_path_falls_back_to_yolo26(self):
        assert load_class_mapping(None) is YOLO26_CLASS_MAPPING

    def test_empty_string_falls_back_to_yolo26(self):
        assert load_class_mapping("") is YOLO26_CLASS_MAPPING

    def test_nonexistent_file_falls_back_to_yolo26(self):
        assert load_class_mapping("/no/such/labels.txt") is YOLO26_CLASS_MAPPING

    def test_loads_labels_in_order(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("vehicle\nperson\nmotorcycle\n")
        result = load_class_mapping(str(labelfile))
        assert result == {0: "vehicle", 1: "person", 2: "motorcycle"}

    def test_blank_lines_skip_index(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("cat\n\ndog\n")
        result = load_class_mapping(str(labelfile))
        assert result == {0: "cat", 2: "dog"}

    def test_empty_file_falls_back_to_yolo26(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("")
        assert load_class_mapping(str(labelfile)) is YOLO26_CLASS_MAPPING

    def test_single_class(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("face\n")
        assert load_class_mapping(str(labelfile)) == {0: "face"}

    def test_utf8_labels_preserved(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("차량\n보행자\n", encoding="utf-8")
        result = load_class_mapping(str(labelfile))
        assert result == {0: "차량", 1: "보행자"}

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="chmod 0o000 does not restrict the root user (as in the Docker container)",
    )
    def test_unreadable_file_falls_back_to_yolo26(self, tmp_path):
        """An IOError/PermissionError during open falls back to YOLO26 mapping."""
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("cat\ndog\n")
        labelfile.chmod(0o000)
        try:
            result = load_class_mapping(str(labelfile))
            assert result is YOLO26_CLASS_MAPPING
        finally:
            labelfile.chmod(0o644)


# ── parse_nvinfer_config ──


class TestParseNvinferConfig:
    def test_basic_keyvalue(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("[property]\nbatch-size=1\nnetwork-type=3\n")
        result = parse_nvinfer_config(str(cfg))
        assert result["batch-size"] == "1"
        assert result["network-type"] == "3"

    def test_skips_comments_and_sections(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text(
            "# comment line\n"
            "[property]\n"
            "labelfile-path=/workspace/models/labels.txt\n"
            "[class-attrs-all]\n"
            "pre-cluster-threshold=0.25\n"
        )
        result = parse_nvinfer_config(str(cfg))
        assert result["labelfile-path"] == "/workspace/models/labels.txt"
        assert result["pre-cluster-threshold"] == "0.25"

    def test_missing_file_returns_empty(self):
        assert parse_nvinfer_config("/no/such/file.txt") == {}

    def test_first_occurrence_wins(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("batch-size=1\nbatch-size=4\n")
        assert parse_nvinfer_config(str(cfg))["batch-size"] == "1"

    def test_whitespace_around_equals(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("onnx-file = /models/x.onnx\n")
        assert parse_nvinfer_config(str(cfg))["onnx-file"] == "/models/x.onnx"


# ── is_segmentation_config ──


class TestIsSegmentationConfig:
    def test_detection_config_false(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("network-type=0\n")
        assert is_segmentation_config(str(cfg)) is False

    def test_seg_network_type_3_true(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("network-type=3\n")
        assert is_segmentation_config(str(cfg)) is True

    def test_output_instance_mask_true(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("output-instance-mask=1\n")
        assert is_segmentation_config(str(cfg)) is True

    def test_missing_file_false(self):
        assert is_segmentation_config("/no/such/file.txt") is False


# ── format_detection_hints with custom detector_name ──


class TestFormatDetectionHintsDetectorName:
    def test_default_detector_name_yolo26(self):
        frame = _make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))])
        out = format_detection_hints([frame], enabled=True)
        assert "[Auxiliary object cues from YOLO26 - may be incomplete or noisy]" in out

    def test_custom_detector_name(self):
        frame = _make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))])
        out = format_detection_hints([frame], enabled=True, detector_name="YOLOE-26")
        assert "[Auxiliary object cues from YOLOE-26 - may be incomplete or noisy]" in out

    def test_seg_detector_name(self):
        frame = _make_frame([_make_detection("Car", 0.8, (0.0, 0.0, 1.0, 1.0))])
        out = format_detection_hints([frame], enabled=True, detector_name="YOLOE-26 Seg")
        assert "[Auxiliary object cues from YOLOE-26 Seg - may be incomplete or noisy]" in out
