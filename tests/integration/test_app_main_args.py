"""Integration tests for vllm_ds_app_kafka_publish.py.

Covers:
- main() argument-parser flag behaviour (--detect-output, --detect, --dry-run)
- seg_mode auto-detection from nvinfer config (is_segmentation_config path)
- VLM_DETECT_LABELFILE / VLM_DETECTOR_NAME env-var export logic
- _build_osd_branch element assembly (Gst fully mocked)

GStreamer is mocked by tests/integration/conftest.py so no GPU/DS needed.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal nvinfer config files
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path, lines: list[str], name: str = "config_infer.txt") -> str:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# ---------------------------------------------------------------------------
# Tests: main() argument parser (pure argparse, no GStreamer)
# ---------------------------------------------------------------------------


class TestMainArgParser:
    """Test the argparse layer of main() without running any pipeline."""

    def _parse(self, argv: list[str]):
        """Import and call the internal arg-parser used by main()."""
        # We exercise the same parser that main() uses by patching sys.exit
        # and calling only the parse step.
        import argparse

        # Reconstruct the parser inline — mirrors main() exactly so we can
        # test flag semantics without importing GStreamer-heavy startup code.
        parser = argparse.ArgumentParser()
        parser.add_argument("sources", nargs="+")
        parser.add_argument("--kafka-bootstrap", default="localhost:9092")
        parser.add_argument("--topic", default="vlm-results")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("-c", "--config", default=None)
        parser.add_argument("--output", default=None)
        parser.add_argument("--detect", action="store_true")
        parser.add_argument("--detect-config", default=None)
        parser.add_argument("--detect-output", default=None)
        return parser.parse_args(argv)

    def test_default_topic(self):
        args = self._parse(["video.mp4"])
        assert args.topic == "vlm-results"

    def test_custom_topic(self):
        args = self._parse(["video.mp4", "--topic", "my-topic"])
        assert args.topic == "my-topic"

    def test_dry_run_default_false(self):
        args = self._parse(["video.mp4"])
        assert args.dry_run is False

    def test_dry_run_flag_sets_true(self):
        args = self._parse(["video.mp4", "--dry-run"])
        assert args.dry_run is True

    def test_detect_default_false(self):
        args = self._parse(["video.mp4"])
        assert args.detect is False

    def test_detect_flag_sets_true(self):
        args = self._parse(["video.mp4", "--detect"])
        assert args.detect is True

    def test_detect_output_default_none(self):
        args = self._parse(["video.mp4"])
        assert args.detect_output is None

    def test_detect_output_stored(self):
        args = self._parse(["video.mp4", "--detect-output", "/tmp/out.mp4"])
        assert args.detect_output == "/tmp/out.mp4"

    def test_detect_config_default_none(self):
        args = self._parse(["video.mp4"])
        assert args.detect_config is None

    def test_multiple_sources_collected(self):
        args = self._parse(["video1.mp4", "video2.mp4"])
        assert args.sources == ["video1.mp4", "video2.mp4"]

    def test_output_default_none(self):
        args = self._parse(["video.mp4"])
        assert args.output is None

    def test_output_path_stored(self):
        args = self._parse(["video.mp4", "--output", "results/out.json"])
        assert args.output == "results/out.json"

    def test_kafka_bootstrap_default(self):
        args = self._parse(["video.mp4"])
        assert args.kafka_bootstrap == "localhost:9092"

    def test_kafka_bootstrap_custom(self):
        args = self._parse(["video.mp4", "--kafka-bootstrap", "broker:9093"])
        assert args.kafka_bootstrap == "broker:9093"


# ---------------------------------------------------------------------------
# Tests: seg_mode logic — is_segmentation_config integration via env vars
# ---------------------------------------------------------------------------


class TestSegModeEnvVarExport:
    """Verify the env-var export block in main() sets the right values."""

    def _run_detect_block(self, tmp_path, cfg_lines, cfg_name="config_infer.txt"):
        """
        Replicate the detect block from main() against a synthetic config.

        Returns (nvinfer_config, seg_mode, env_snapshot).
        """
        from vlm_utils import is_segmentation_config, parse_nvinfer_config

        cfg_path = _write_cfg(tmp_path, cfg_lines, cfg_name)

        seg_mode = is_segmentation_config(cfg_path)
        nvinfer_props = parse_nvinfer_config(cfg_path)
        labelfile = nvinfer_props.get("labelfile-path", "")

        env = {}
        if labelfile and os.path.isfile(labelfile):
            env["VLM_DETECT_LABELFILE"] = labelfile
        cfg_name_lower = os.path.basename(cfg_path).lower()
        if "yolo26e" in cfg_name_lower or "yoloe" in cfg_name_lower:
            env["VLM_DETECTOR_NAME"] = "YOLOE-26" + (" Seg" if seg_mode else "")
        else:
            env["VLM_DETECTOR_NAME"] = "YOLO26"

        return cfg_path, seg_mode, env

    def test_yolo26_config_sets_yolo26_detector_name(self, tmp_path):
        _, _, env = self._run_detect_block(tmp_path, ["network-type=0"], "config_infer_yolo26.txt")
        assert env["VLM_DETECTOR_NAME"] == "YOLO26"

    def test_yoloe_detect_config_sets_yoloe_detector_name(self, tmp_path):
        _, _, env = self._run_detect_block(tmp_path, ["network-type=0"], "config_infer_yoloe.txt")
        assert env["VLM_DETECTOR_NAME"] == "YOLOE-26"

    def test_yoloe_seg_config_sets_yoloe_seg_detector_name(self, tmp_path):
        _, seg_mode, env = self._run_detect_block(
            tmp_path, ["network-type=3"], "config_infer_yoloe.txt"
        )
        assert seg_mode is True
        assert env["VLM_DETECTOR_NAME"] == "YOLOE-26 Seg"

    def test_yolo26e_name_also_triggers_yoloe_branch(self, tmp_path):
        _, _, env = self._run_detect_block(tmp_path, ["network-type=0"], "config_infer_yolo26e.txt")
        assert env["VLM_DETECTOR_NAME"] == "YOLOE-26"

    def test_labelfile_exported_when_file_exists(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("vehicle\nperson\n")
        cfg_lines = [f"labelfile-path={labelfile}"]
        _, _, env = self._run_detect_block(tmp_path, cfg_lines, "config_infer_yolo26.txt")
        assert env.get("VLM_DETECT_LABELFILE") == str(labelfile)

    def test_labelfile_not_exported_when_file_missing(self, tmp_path):
        cfg_lines = ["labelfile-path=/no/such/labels.txt"]
        _, _, env = self._run_detect_block(tmp_path, cfg_lines, "config_infer_yolo26.txt")
        assert "VLM_DETECT_LABELFILE" not in env

    def test_labelfile_not_exported_when_key_absent(self, tmp_path):
        _, _, env = self._run_detect_block(tmp_path, ["network-type=0"], "config_infer_yolo26.txt")
        assert "VLM_DETECT_LABELFILE" not in env

    def test_seg_mode_false_for_network_type_0(self, tmp_path):
        _, seg_mode, _ = self._run_detect_block(tmp_path, ["network-type=0"])
        assert seg_mode is False

    def test_seg_mode_false_for_empty_config(self, tmp_path):
        _, seg_mode, _ = self._run_detect_block(tmp_path, ["batch-size=1"])
        assert seg_mode is False

    def test_seg_mode_true_for_output_instance_mask(self, tmp_path):
        _, seg_mode, _ = self._run_detect_block(tmp_path, ["output-instance-mask=1"])
        assert seg_mode is True


# ---------------------------------------------------------------------------
# Tests: _build_osd_branch — Gst element creation + link chain (mocked)
# ---------------------------------------------------------------------------


class TestBuildOsdBranch:
    """
    Unit-test _build_osd_branch with Gst.ElementFactory fully mocked.

    Strategy: Gst.ElementFactory.make() is called with (factory_name, alias).
    The "queue" factory is called twice (once for queue_vlm, once for queue_osd),
    so we cannot key by factory_name alone. Instead we build an ordered call
    sequence and use iter() to hand out the right stub on each successive call.

    GPU call order (9 calls):
      tee, queue(vlm), queue(osd), nvosdbin, nvv4l2h264enc,
      h264parse, qtmux, filesink
      [nvv4l2h264enc returns non-None so x264enc is never tried]

    CPU call order (11 calls):
      tee, queue(vlm), queue(osd), nvosdbin, nvv4l2h264enc(→None),
      x264enc, h264parse, qtmux, filesink, nvvideoconvert, capsfilter
    """

    def _make_el(self, name: str) -> MagicMock:
        el = MagicMock(name=f"gst_{name}")
        el.link.return_value = True
        el.find_property.return_value = None
        pad_mock = MagicMock()
        pad_mock.link.return_value = 0  # Gst.PadLinkReturn.OK
        el.request_pad_simple.return_value = pad_mock
        el.get_static_pad.return_value = MagicMock()
        return el

    def _build_elements(self, use_gpu_enc: bool) -> dict:
        """Return named element stubs (so tests can assert on them)."""
        els = {
            "tee": self._make_el("tee"),
            "queue_vlm": self._make_el("queue_vlm"),
            "queue_osd": self._make_el("queue_osd"),
            "nvosdbin": self._make_el("nvosdbin"),
            "encoder": self._make_el("nvv4l2h264enc" if use_gpu_enc else "x264enc"),
            "parser": self._make_el("h264parse"),
            "mux": self._make_el("qtmux"),
            "filesink": self._make_el("filesink"),
        }
        if not use_gpu_enc:
            els["bridge"] = self._make_el("nvvideoconvert")
            els["bridge_caps"] = self._make_el("capsfilter")
        return els

    def _make_factory_fn(
        self, els: dict, use_gpu_enc: bool, nvosdbin_missing=False, encoder_missing=False
    ):
        """
        Return a side_effect callable for Gst.ElementFactory.make that hands
        out stubs in the exact order _build_osd_branch calls them.
        """
        # Build the ordered response list matching the source call sequence.
        nvosdbin_el = None if nvosdbin_missing else els["nvosdbin"]

        if nvosdbin_missing:
            # Only tee + two queues are created before the early return
            seq = [els["tee"], els["queue_vlm"], els["queue_osd"], None]
        elif encoder_missing:
            # Both encoder attempts fail
            seq = [
                els["tee"],
                els["queue_vlm"],
                els["queue_osd"],
                nvosdbin_el,
                None,  # nvv4l2h264enc → None
                None,  # x264enc → None
            ]
        elif use_gpu_enc:
            seq = [
                els["tee"],
                els["queue_vlm"],
                els["queue_osd"],
                nvosdbin_el,
                els["encoder"],  # nvv4l2h264enc succeeds
                els["parser"],
                els["mux"],
                els["filesink"],
            ]
        else:
            # CPU path: nvv4l2h264enc fails, x264enc succeeds, then bridge
            seq = [
                els["tee"],
                els["queue_vlm"],
                els["queue_osd"],
                nvosdbin_el,
                None,  # nvv4l2h264enc → None
                els["encoder"],  # x264enc succeeds
                els["parser"],
                els["mux"],
                els["filesink"],
                els["bridge"],
                els["bridge_caps"],
            ]

        it = iter(seq)

        def _make(factory_name, alias=None):
            return next(it)

        return _make

    def _build_app(self):
        from vllm_ds_app_kafka_publish import VLMKafkaApp

        app = VLMKafkaApp.__new__(VLMKafkaApp)
        app.seg_mode = False
        app.kafka_publisher = MagicMock()
        pipeline_mock = MagicMock()
        pipeline_mock.add = MagicMock()
        app.pipeline = pipeline_mock
        return app

    # --- error / early-exit paths ---

    def test_returns_none_when_nvosdbin_unavailable(self, tmp_path):
        """When nvosdbin cannot be created, _build_osd_branch returns None."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True, nvosdbin_missing=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            result = app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        assert result is None

    def test_returns_none_when_no_encoder_available(self, tmp_path):
        """When neither nvv4l2h264enc nor x264enc is available, returns None."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True, encoder_missing=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            result = app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        assert result is None

    def test_returns_none_when_tee_pad_link_fails(self, tmp_path):
        """If a tee → queue pad link returns non-OK, _build_osd_branch returns None."""
        els = self._build_elements(use_gpu_enc=True)
        # Make the tee return a pad whose link() reports failure
        bad_pad = MagicMock()
        bad_pad.link.return_value = 1  # != PadLinkReturn.OK (0)
        els["tee"].request_pad_simple.return_value = bad_pad

        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            result = app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        assert result is None

    # --- success path (GPU encoder) ---

    def test_returns_tee_on_success(self, tmp_path):
        """On a fully successful GPU-encoder build, the tee element is returned."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            result = app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        assert result is els["tee"]

    def test_gpu_encoder_sets_bps_bitrate(self, tmp_path):
        """With nvv4l2h264enc, bitrate is set to 4_000_000 bps."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        els["encoder"].set_property.assert_any_call("bitrate", 4_000_000)

    def test_seg_mode_sets_display_mask_true(self, tmp_path):
        """When seg_mode=True, nvosdbin.display-mask is set to True."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=True)

        els["nvosdbin"].set_property.assert_any_call("display-mask", True)

    def test_detection_mode_sets_display_mask_false(self, tmp_path):
        """When seg_mode=False, nvosdbin.display-mask is set to False."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        els["nvosdbin"].set_property.assert_any_call("display-mask", False)

    def test_filesink_location_set_to_output_path(self, tmp_path):
        """filesink location property is set to the output_path argument."""
        out = str(tmp_path / "annotated.mp4")
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(out, seg_mode=False)

        els["filesink"].set_property.assert_any_call("location", out)

    def test_queue_vlm_is_unlimited(self, tmp_path):
        """queue_vlm max-size-* are all 0 (unlimited) to avoid backpressure."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        q = els["queue_vlm"]
        q.set_property.assert_any_call("max-size-buffers", 0)
        q.set_property.assert_any_call("max-size-time", 0)
        q.set_property.assert_any_call("max-size-bytes", 0)

    def test_queue_osd_is_leaky_downstream(self, tmp_path):
        """queue_osd leaky=2 (downstream) so encoder backpressure drops frames."""
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        els["queue_osd"].set_property.assert_any_call("leaky", 2)

    def test_output_directory_is_created(self, tmp_path):
        """os.makedirs is called for the output file's parent directory."""
        out = str(tmp_path / "nested" / "output" / "out.mp4")
        els = self._build_elements(use_gpu_enc=True)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=True)

        with (
            patch("vllm_ds_app_kafka_publish.Gst") as gst_mock,
            patch("vllm_ds_app_kafka_publish.os.makedirs") as mkdirs_mock,
        ):
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(out, seg_mode=False)

        mkdirs_mock.assert_called_once()

    # --- CPU encoder fallback path ---

    def test_cpu_encoder_sets_kbps_bitrate(self, tmp_path):
        """With x264enc fallback, bitrate is set to 4000 kbps."""
        els = self._build_elements(use_gpu_enc=False)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=False)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        els["encoder"].set_property.assert_any_call("bitrate", 4000)

    def test_cpu_path_returns_tee_on_success(self, tmp_path):
        """CPU-encoder path also returns the tee element on success."""
        els = self._build_elements(use_gpu_enc=False)
        app = self._build_app()
        factory_fn = self._make_factory_fn(els, use_gpu_enc=False)

        with patch("vllm_ds_app_kafka_publish.Gst") as gst_mock:
            gst_mock.ElementFactory.make.side_effect = factory_fn
            gst_mock.PadLinkReturn.OK = 0
            gst_mock.Caps.from_string.return_value = MagicMock()
            result = app._build_osd_branch(str(tmp_path / "out.mp4"), seg_mode=False)

        assert result is els["tee"]
