###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###################################################################################################

"""VLMKafkaApp — DeepStream pipeline orchestrator.

Extracted from src/vllm_ds_app_kafka_publish.py during the R-5 god-module split.
Owns the GStreamer pipeline, the bus loop, and the per-stream source_map
derivation.  Result publishing is delegated to
``my_curator.application.pipeline.publisher.VLMKafkaSignalPublisher``; the OSD
visualization branch is delegated to
``my_curator.application.pipeline.osd_branch.build_osd_branch``.

Module-level GStreamer plugin registration is preserved here so that
``Gst.Element.register("nvvllmvlm", NvVllmVLM)`` runs exactly once when this
module is imported — same contract as the legacy god-module.
"""

from __future__ import annotations

import json
import os
import re

try:
    import gi

    gi.require_version("Gst", "1.0")  # noqa: E402, I003, BLK100
    from gi.repository import GLib, Gst  # noqa: E402, I003

    from my_curator.adapters.gst.nvvllmvlm import NvVllmVLM  # noqa: E402

    Gst.Element.register(None, "nvvllmvlm", Gst.Rank.NONE, NvVllmVLM)
    GI_AVAILABLE = True
except (ImportError, ValueError):
    GI_AVAILABLE = False

from my_curator.adapters.gst.utils import (
    get_video_resolution,
    load_class_mapping,
    move_built_engine,
)
from my_curator.application.pipeline.osd_branch import build_osd_branch
from my_curator.application.pipeline.publisher import VLMKafkaSignalPublisher


class VLMKafkaApp:
    """DeepStream VLM app with Kafka publishing via signals
    (single or multi-stream, file or RTSP sources)"""

    def __init__(
        self,
        input_uris,
        kafka_config,
        topic,
        dry_run=False,
        output_path=None,
        nvinfer_config=None,
        osd_output_path=None,
        seg_mode=False,
        source_clip_id_override=None,
        engine=None,
    ):
        """
        Initialize application.

        Args:
            input_uris: List of GStreamer-compatible URIs (file:// or rtsp://)
            kafka_config: Kafka connection configuration
            topic: Kafka topic name
            dry_run: If True, print messages instead of sending to Kafka
            output_path: Path to save results as JSON file (optional)
            nvinfer_config: Path to nvinfer config (detection enabled when set)
            osd_output_path: Path to MP4 output with OSD overlay (bbox + masks)
            seg_mode: True when nvinfer_config is a segmentation network
                (network-type=3); enables ``display-mask`` on nvdsosd
            source_clip_id_override: Override source_clip_id for single-source
                invocations; ignored when multiple sources are provided
        """
        self.input_uris = input_uris
        self.num_sources = len(input_uris)
        self.pipeline = None
        self.loop = None
        self.streams_eos = set()
        self.output_path = output_path
        self.nvinfer_config = nvinfer_config
        self.osd_output_path = osd_output_path
        self.seg_mode = seg_mode
        self._engine = engine
        self._class_mapping = load_class_mapping(os.environ.get("VLM_DETECT_LABELFILE"))

        # Auto-derive source_map: stream_id → (source_clip_id, source_video_path).
        # source_clip_id = filename stem (e.g. "66751" from "66751.mp4").
        # source_video_path = VIDEO_DATA_ROOT-relative path stored as file:// blob_uri.
        # When --source-clip-id is provided and exactly one source is given, the
        # override replaces the auto-derived stem for that single stream.
        _video_root = os.environ.get("VIDEO_DATA_ROOT", "")
        _source_map: dict[int, tuple[str | None, str | None]] = {}
        _single_source = len(input_uris) == 1
        for _i, _uri in enumerate(input_uris):
            if _uri.startswith("file://"):
                _abs = _uri[len("file://") :]
                _clip_id = os.path.splitext(os.path.basename(_abs))[0]
                if source_clip_id_override and _single_source:
                    _clip_id = source_clip_id_override
                _rel: str | None = None
                if _video_root:
                    try:
                        import pathlib as _pl

                        _rel = str(_pl.Path(_abs).relative_to(_video_root))
                    except ValueError:
                        pass
                _source_map[_i] = (_clip_id, _rel)
        self._source_map = _source_map

        # R-4 positive assert: silent GI_AVAILABLE=False fallback would otherwise
        # let the pipeline build with a broken GStreamer registration.  Failing
        # loudly here is the acceptance gate's requirement (plan §7 risk row).
        assert GI_AVAILABLE is True, (
            "GStreamer / gstnvvllmvlm import failed at module load — "
            "VLMKafkaApp cannot run without a working GI environment."
        )

        # P2-4: load ScoutConfig + BestOfNAggregator for curation wiring
        _scout_config = None
        _aggregator = None
        _scout_yaml = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "..",
                "configs",
                "scout.yaml",
            )
        )
        if os.path.exists(_scout_yaml):
            try:
                from my_curator.domain.scout.aggregator import BestOfNAggregator
                from my_curator.domain.scout.base import ScoutConfig

                _scout_config = ScoutConfig.from_yaml(_scout_yaml)
                _aggregator = BestOfNAggregator()
                print(
                    f"✓ Scout config loaded (N={_scout_config.n}, topics: "
                    f"{_scout_config.kafka_topic_scouted} / "
                    f"{_scout_config.kafka_topic_needs_review})"
                )
            except Exception as e:
                print(f"✗ Failed to load scout config: {e} — curation disabled")

        # Initialize Kafka publisher
        self.kafka_publisher = VLMKafkaSignalPublisher(
            kafka_config,
            topic,
            dry_run,
            detect_hints=bool(nvinfer_config),
            aggregator=_aggregator,
            scout_config=_scout_config,
            source_map=self._source_map,
        )

    def bus_call(self, bus, message, loop):
        """Handle GStreamer bus messages"""
        t = message.type

        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()

        return True

    def pad_probe_callback(self, pad, info, stream_id):
        """Probe to detect per-stream EOS"""
        gst_buffer = info.get_buffer()
        if gst_buffer:
            if gst_buffer.pts == Gst.CLOCK_TIME_NONE:
                print(f"Stream {stream_id}: Received EOS")
                self.streams_eos.add(stream_id)

                if len(self.streams_eos) == self.num_sources:
                    print(f"All {self.num_sources} stream(s) finished")

        return Gst.PadProbeReturn.OK

    def build_pipeline(self):
        """Build the GStreamer pipeline.

        Uses uridecodebin per stream so that both file:// and rtsp:// URIs
        are supported transparently. uridecodebin selects the appropriate
        source plugin, demuxer, parser, and hardware decoder automatically.
        """
        print(f"Building pipeline for {self.num_sources} source(s)...")

        has_live = any(
            uri.startswith("rtsp://") or uri.startswith("rtsps://") for uri in self.input_uris
        )

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("vlm-kafka-signal-pipeline")

        # Create streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        # Size the muxer to source native resolution; a fixed W×H up/down-scales
        # every input and wastes vision tokens. Multi-source -> max; fallback 1920x1080.
        mux_w, mux_h = 1920, 1080
        probed = [r for r in (get_video_resolution(u) for u in self.input_uris) if r]
        if probed:
            mux_w = max(w for w, _ in probed)
            mux_h = max(h for _, h in probed)
        streammux.set_property("width", mux_w)
        streammux.set_property("height", mux_h)
        print(f"  nvstreammux resolution: {mux_w}x{mux_h}")
        streammux.set_property("batch-size", self.num_sources)
        streammux.set_property("live-source", has_live)
        if not has_live:
            streammux.set_property("batched-push-timeout", 4000000)

        # Add to pipeline
        self.pipeline.add(streammux)

        # Pre-request mux sink pads so pad-added callbacks can link into them
        mux_sink_pads = []
        for i in range(self.num_sources):
            sink_pad = streammux.request_pad_simple(f"sink_{i}")
            if not sink_pad:
                print(f"Error: Could not get sink pad {i} from streammux")
                return None
            sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.pad_probe_callback, i)
            mux_sink_pads.append(sink_pad)

        # Create one uridecodebin per source
        for i, uri in enumerate(self.input_uris):
            print(f"  Source {i}: {uri}")

            uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{i}")
            if not uri_decode_bin:
                print(f"Error: Could not create uridecodebin for stream {i}")
                return None

            uri_decode_bin.set_property("uri", uri)
            self.pipeline.add(uri_decode_bin)

            # Capture loop variables via default args
            def on_pad_added(
                element,
                pad,
                mux_sinkpad=mux_sink_pads[i],
                stream_id=i,
            ):
                caps = pad.get_current_caps()
                if not caps:
                    caps = pad.query_caps()
                if not caps:
                    return
                structure = caps.get_structure(0)
                if "video" in structure.get_name():
                    if not mux_sinkpad.is_linked():
                        if pad.link(mux_sinkpad) == Gst.PadLinkReturn.OK:
                            print(f"  Linked uridecodebin → streammux.sink_{stream_id}")

            uri_decode_bin.connect("pad-added", on_pad_added)

        # Object detection via nvinfer (--detect mode)
        nvinfer = None
        if self.nvinfer_config:
            # Parse model-engine-file from config for engine auto-detect
            self._engine_dest = None
            try:
                with open(self.nvinfer_config) as f:
                    for line in f:
                        m = re.match(r"model-engine-file\s*=\s*(\S+)", line)
                        if m:
                            self._engine_dest = m.group(1)
                            break
            except Exception:
                pass

            nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
            if nvinfer:
                nvinfer.set_property("config-file-path", self.nvinfer_config)
                self.pipeline.add(nvinfer)
                print(f"✓ nvinfer loaded: {self.nvinfer_config}")
            else:
                print("✗ Failed to create nvinfer element")

        # Tracker (NvDCF) — assigns stable object_id per detected object
        nvtracker = None
        if nvinfer:
            nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if nvtracker:
                # Locate tracker config relative to the repo root (this module
                # lives at my_curator/application/pipeline/ds_app.py — 4 levels
                # deep) so the relative path resolves the same as the legacy
                # src/ location.
                script_dir = os.path.dirname(os.path.abspath(__file__))
                tracker_config = os.path.normpath(
                    os.path.join(
                        script_dir,
                        "..",
                        "..",
                        "..",
                        "configs",
                        "config_tracker_NvDCF_perf.yml",
                    )
                )
                tracker_lib = (
                    "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
                )
                nvtracker.set_property("ll-config-file", tracker_config)
                nvtracker.set_property("ll-lib-file", tracker_lib)
                self.pipeline.add(nvtracker)
                print("✓ nvtracker (NvDCF) loaded")
            else:
                print("✗ Failed to create nvtracker element")

        # Video converter
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvvidconv.set_property("nvbuf-memory-type", 0)

        # Caps filter for RGB
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        caps_rgb = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGB")
        caps_filter.set_property("caps", caps_rgb)

        # VLM plugin
        nvvllm = Gst.ElementFactory.make("nvvllmvlm", "vlm-infer")

        # Inject a shared engine for --warm reuse (element self-loads otherwise).
        if self._engine is not None and hasattr(nvvllm, "set_engine"):
            nvvllm.set_engine(self._engine)

        # Connect signal to Kafka publisher
        nvvllm.connect("vlm-result", self.kafka_publisher.on_vlm_result)
        print("✓ Connected vlm-result signal to Kafka publisher")

        # Fakesink
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")
        sink.set_property("sync", False)

        # Add elements to pipeline
        self.pipeline.add(nvvidconv)
        self.pipeline.add(caps_filter)
        self.pipeline.add(nvvllm)
        self.pipeline.add(sink)

        # Optional OSD branch (only when --detect-output and --detect both on)
        osd_tee = None
        if nvinfer and self.osd_output_path:
            osd_tee = build_osd_branch(
                self.pipeline, self._class_mapping, self.osd_output_path, self.seg_mode
            )

        # Link pipeline
        #   With OSD tee:  streammux → nvinfer → nvtracker → tee ─┬→ queue_vlm → nvvidconv → caps → nvvllm → sink
        #                                                         └→ queue_osd → nvosdbin → ... → filesink
        #   Without OSD:   streammux → [nvinfer → nvtracker →] nvvidconv → caps → nvvllm → sink
        if osd_tee is not None:
            streammux.link(nvinfer)
            if nvtracker:
                nvinfer.link(nvtracker)
                nvtracker.link(osd_tee)
            else:
                nvinfer.link(osd_tee)
            queue_vlm = self.pipeline.get_by_name("queue_vlm")
            queue_vlm.link(nvvidconv)
        elif nvinfer:
            streammux.link(nvinfer)
            if nvtracker:
                nvinfer.link(nvtracker)
                nvtracker.link(nvvidconv)
            else:
                nvinfer.link(nvvidconv)
        else:
            streammux.link(nvvidconv)
        nvvidconv.link(caps_filter)
        caps_filter.link(nvvllm)
        nvvllm.link(sink)

        print("Pipeline built successfully\n")

        return self.pipeline

    def _build_osd_branch(self, output_path, seg_mode):
        """Backward-compat thin wrapper around osd_branch.build_osd_branch.

        R-5 extracted the OSD branch builder into a free function in
        my_curator.application.pipeline.osd_branch; existing test patch
        sites (``tests/integration/test_app_main_args.TestBuildOsdBranch``)
        invoke this method on a VLMKafkaApp instance constructed via
        ``__new__`` (bypassing __init__), so the helper attributes may
        not exist.  ``getattr`` guards keep the surface defensive.
        Removed in R-7 alongside the test patch-site updates.
        """
        class_mapping = getattr(self, "_class_mapping", {})
        return build_osd_branch(self.pipeline, class_mapping, output_path, seg_mode)

    def run(self):
        """Run the application"""
        # Build pipeline
        pipeline = self.build_pipeline()

        # Set up bus
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        # Create main loop
        self.loop = GLib.MainLoop()
        bus.connect("message", self.bus_call, self.loop)

        # Start pipeline
        print("Starting pipeline...")
        pipeline.set_state(Gst.State.PLAYING)

        try:
            print("Running... (Press Ctrl+C to stop)\n")
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")

        # Cleanup
        print("\nStopping pipeline...")
        pipeline.set_state(Gst.State.NULL)

        # Engine auto-detect: nvinfer builds engine with pattern
        # model_b*_gpu*_fp*.engine in CWD. Rename to the path in config
        # so next run skips rebuild.
        engine_dest = getattr(self, "_engine_dest", None)
        moved = move_built_engine(engine_dest)
        if moved:
            print(f"✓ Engine moved to: {moved}")

        # Save results to JSON file if requested
        _results = self.kafka_publisher.get_collected_results()
        if self.output_path and _results:
            output_data = {
                "sources": [uri for uri in self.input_uris],
                "total_segments": len(_results),
                "segments": _results,
            }
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Results saved to {self.output_path}")

        # Close Kafka publisher
        self.kafka_publisher.close()
