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
        """Handle GStreamer bus messages (runtime source add/delete)."""
        t = message.type

        if t == Gst.MessageType.EOS:
            print("End-of-stream (all sources)")
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            # nvstreammux posts a per-source "stream-eos" ELEMENT message on the bus,
            # but bus messages are async and can be delivered late — a clip's stream-eos
            # can surface at the START of the NEXT clip's loop.run() and quit it before
            # any frame arrives (every other clip comes up empty). We IGNORE it and rely
            # solely on the synchronous, self-removing mux-sink EOS probe in _add_source.
            struct = message.get_structure()
            if struct is not None and struct.get_name() == "stream-eos":
                ok, stream_id = struct.get_uint("stream-id")
                if ok:
                    print(f"Per-source EOS: stream {stream_id} (bus, ignored)")
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            self._pipeline_error = True
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

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("vlm-kafka-signal-pipeline")

        # nvstreammux — PERSISTENT. A keepalive source holds sink_0 for the whole
        # session so the muxer never hits a 0-source stall; real clips are added/
        # removed at runtime on sink_1 (add_source / stop_release_source). The vLLM
        # engine loads once per session and NVMM pools are reused across clips.
        mux_w, mux_h = 1920, 1080
        probed = [r for r in (get_video_resolution(u) for u in self.input_uris) if r]
        if probed:
            mux_w = max(w for w, _ in probed)
            mux_h = max(h for _, h in probed)
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property("width", mux_w)
        streammux.set_property("height", mux_h)
        print(f"  nvstreammux resolution (persistent, max-of-corpus): {mux_w}x{mux_h}")
        streammux.set_property("batch-size", 2)  # keepalive (sink_0) + 1 real slot (sink_1)
        streammux.set_property("live-source", 1)  # required for runtime source add/remove
        streammux.set_property("batched-push-timeout", 4000000)
        streammux.set_property(
            "drop-pipeline-eos", 1
        )  # per-source EOS -> stream-eos, not pipeline EOS
        self.pipeline.add(streammux)
        self._streammux = streammux

        # Keepalive source on sink_0 (permanent): a tiny solid-black live feed that
        # keeps the muxer batching so a clip added to sink_1 at runtime is delivered
        # cleanly (no cold-add frame drop). nvvllmvlm skips this stream_id.
        ks = Gst.ElementFactory.make("videotestsrc", "keepalive-src")
        ks.set_property("pattern", 2)
        ks.set_property("is-live", True)
        ks.set_property("do-timestamp", True)
        kc = Gst.ElementFactory.make("nvvideoconvert", "keepalive-conv")
        kf = Gst.ElementFactory.make("capsfilter", "keepalive-caps")
        kf.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=NV12,width=320,height=240,framerate=5/1"
            ),
        )
        for _e in (ks, kc, kf):
            self.pipeline.add(_e)
        ks.link(kc)
        kc.link(kf)
        kf.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
        self._keepalive_stream_id = 0
        self._real_slot = 1

        # Runtime source-management state for the real slot (populated in add_source).
        self._source_bins: dict[int, object] = {}
        self._source_pads: dict[int, object] = {}

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
        self._nvvllm = nvvllm  # for per-clip flush_stream() during runtime source swap
        nvvllm._keepalive_stream_id = self._keepalive_stream_id  # never curate the keepalive feed

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

    def _update_source_map(self, index: int) -> None:
        """Point the publisher's source_map at the current clip (real stream_id = _real_slot)."""
        self.kafka_publisher._source_map = {
            self._real_slot: self._source_map.get(index, (None, None))
        }

    def _add_source(self, uri: str) -> None:
        """Attach a uridecodebin for *uri* to streammux.sink_<real_slot> at runtime
        (pipeline stays PLAYING). The mux sink pad is requested inside pad-added
        (NVIDIA reference pattern), and a one-shot EOS probe on it quits the loop so
        run() advances to flush + release."""
        slot = self._real_slot
        src = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{slot}")
        src.set_property("uri", uri)

        def on_pad_added(element, pad):
            caps = pad.get_current_caps() or pad.query_caps()
            if not caps or "video" not in caps.get_structure(0).get_name():
                return
            if self._streammux.get_static_pad(f"sink_{slot}"):
                return  # guard against a second video pad re-requesting the sink
            mux_sink = self._streammux.request_pad_simple(f"sink_{slot}")
            if not mux_sink or pad.link(mux_sink) != Gst.PadLinkReturn.OK:
                return
            self._source_pads[slot] = mux_sink

            def _eos_probe(_pad, info, _u):
                if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
                    ev = info.get_event()
                    if ev and ev.type == Gst.EventType.EOS:
                        GLib.idle_add(self.loop.quit)
                        return Gst.PadProbeReturn.REMOVE
                return Gst.PadProbeReturn.OK

            mux_sink.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, _eos_probe, None)

        src.connect("pad-added", on_pad_added)
        self._source_bins[slot] = src
        self.pipeline.add(src)
        src.sync_state_with_parent()

    def _stop_release_source(self) -> None:
        """Full source teardown: flush-stop → release request pad → NULL → remove."""
        slot = self._real_slot
        src = self._source_bins.pop(slot, None)
        sink_pad = self._source_pads.pop(slot, None)
        if src is not None:
            src.set_state(Gst.State.NULL)
        if sink_pad is not None:
            # flush-stop resets this sink pad's flushing state before release; the next
            # clip's uridecodebin supplies its own fresh segment. (Per-source EOS is
            # handled by the synchronous mux-sink probe, so no in-flight EOS races here.)
            sink_pad.send_event(Gst.Event.new_flush_stop(False))
            self._streammux.release_request_pad(sink_pad)
        if src is not None:
            self.pipeline.remove(src)

    def run(self):
        """Run the app — ONE persistent pipeline kept PLAYING; a keepalive feed holds
        sink_0 while clips are added/removed on sink_<real_slot> (DeepStream runtime
        source add/delete). Each clip's last segment is flushed via
        nvvllmvlm.flush_stream(real_slot) on its per-source EOS, so the vLLM engine +
        NVMM pools are allocated once for the whole session (no per-clip rebuild)."""
        pipeline = self.build_pipeline()
        self._pipeline_error = False

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        self.loop = GLib.MainLoop()
        bus.connect("message", self.bus_call, self.loop)

        print("Starting persistent pipeline...")
        pipeline.set_state(Gst.State.PLAYING)

        total = len(self.input_uris)
        try:
            for i, uri in enumerate(self.input_uris):
                print(f"\n[persistent {i + 1}/{total}] {uri}")
                self._update_source_map(i)
                self._add_source(uri)
                try:
                    self.loop.run()  # blocks until this source's EOS probe quits it (or error)
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    self._stop_release_source()
                    break
                if self._pipeline_error:
                    print(f"[persistent] pipeline error on clip {i} — stopping")
                    break
                # Flush this clip's segments (worker + engine stay up), then release
                # the source; the pipeline keeps PLAYING for the next clip.
                self._nvvllm.flush_stream(self._real_slot)
                self._stop_release_source()
        finally:
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
