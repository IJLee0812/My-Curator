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

"""
DeepStream VLM application with Kafka publishing and signal-based result
handling.
Supports both single-stream and multi-stream processing with file and RTSP
sources.

Features:
- Single-stream and multi-stream VLM processing
- Real-time result delivery via GObject signals
- Kafka topic publishing for downstream processing
- Dry-run mode for testing without Kafka
- Efficient event-driven architecture
- File and RTSP source support via uridecodebin
"""

import json
import os
import re
import sys
import time
from typing import Optional

import gi

gi.require_version("Gst", "1.0")  # noqa: E402, I003, BLK100
# Register the custom plugin
import gstnvvllmvlm  # noqa: E402
from gi.repository import GLib, Gst  # noqa: E402, I003

Gst.Element.register(None, "nvvllmvlm", Gst.Rank.NONE, gstnvvllmvlm.NvVllmVLM)

# Kafka imports (with graceful fallback)
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Run: pip install kafka-python")


from vlm_utils import (  # noqa: E402
    check_onnx_exists,
    is_segmentation_config,
    move_built_engine,
    parse_nvinfer_config,
    parse_vlm_json,
    to_uri,
    validate_driving_scene_json,
)


class VLMKafkaSignalPublisher:
    """
    Kafka publisher that uses GObject signals to receive VLM results.
    More efficient than polling - publishes immediately when results are
    available.
    """

    def __init__(
        self,
        kafka_config: dict,
        topic: str,
        dry_run: bool = False,
        detect_hints: bool = False,
    ):
        """
        Initialize Kafka publisher.

        Args:
            kafka_config: Kafka connection configuration
            topic: Topic name to publish to
            dry_run: If True, print messages instead of sending to Kafka
            detect_hints: If True, include detect_hints flag in message metadata
        """
        self.topic = topic
        self.dry_run = dry_run
        self.detect_hints = detect_hints
        self.producer: KafkaProducer | None = None
        self.messages_sent = 0
        self.messages_failed = 0
        self._collected_results: list = []

        # Initialize Kafka producer
        if not dry_run and KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=kafka_config.get("bootstrap_servers", "localhost:9092"),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    retries=3,
                    # Required for idempotence
                    max_in_flight_requests_per_connection=1,
                    enable_idempotence=True,
                    compression_type="gzip",
                    linger_ms=100,
                    batch_size=16384,
                )
                print(f"✓ Kafka producer initialized (topic: {self.topic})")
            except Exception as e:
                print(f"✗ Failed to initialize Kafka producer: {e}")
                print("  Falling back to dry-run mode (console output only)")
                self.dry_run = True
                self.producer = None
                print("✓ Dry-run mode enabled")
        else:
            if not KAFKA_AVAILABLE:
                print("✗ Kafka not available - dry-run mode enabled")
            else:
                print("✓ Dry-run mode enabled (console output only)")
            self.producer = None

    def on_vlm_result(self, element, stream_id, start_time, end_time, result_text):
        """
        Signal handler for vlm-result signal.
        Called immediately when VLM inference completes.

        Args:
            element: The nvvllmvlm element that emitted the signal
            stream_id: Stream identifier
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            result_text: VLM inference result
        """
        # Validate VLM output against the DrivingSceneResult schema.
        # Invalid output is still published — flag-only, never dropped.
        parsed, parse_err = parse_vlm_json(result_text)
        if parsed is None:
            json_valid = False
        else:
            ok, _ = validate_driving_scene_json(parsed)
            json_valid = ok
        if not json_valid:
            reason = parse_err or "schema validation failed"
            print(
                f"VLMKafkaPublisher: json_valid=False for stream {stream_id} "
                f"[{start_time:.2f}s-{end_time:.2f}s] — {reason}"
            )

        # Construct message
        message = {
            "stream_id": stream_id,
            "timestamp": time.time(),
            "segment": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            },
            "result": result_text,
            "metadata": {
                "source": "vllm-ds-plugin",
                "version": "1.0",
                **({"detect_hints": True} if self.detect_hints else {}),
                "json_valid": json_valid,
            },
        }

        # Collect for JSON output
        self._collected_results.append(message)

        # Publish to Kafka or print to console
        self.publish(message, stream_id)

    def publish(self, message: dict, stream_id: int):
        """
        Publish message to Kafka or print to console.

        Args:
            message: Message payload
            stream_id: Stream ID (used as partition key)
        """
        # Use stream_id as partition key for ordering
        partition_key = f"stream_{stream_id}"

        if self.dry_run or self.producer is None:
            # Dry-run mode: print to console
            print(f"\n{'=' * 80}")
            print("📤 KAFKA MESSAGE (Dry-Run)")
            print(f"{'=' * 80}")
            print(f"Topic: {self.topic}")
            print(f"Key: {partition_key}")
            print(f"Value: {json.dumps(message, indent=2)}")
            print(f"{'=' * 80}\n")
            self.messages_sent += 1
        else:
            # Send to Kafka
            try:
                future = self.producer.send(self.topic, key=partition_key, value=message)

                # Optional: wait for acknowledgment
                record_metadata = future.get(timeout=10)

                self.messages_sent += 1
                print(
                    f"✓ Published to Kafka: stream={stream_id}, "
                    f"time={message['segment']['start_time']:.1f}s-"
                    f"{message['segment']['end_time']:.1f}s, "
                    f"partition={record_metadata.partition}, "
                    f"offset={record_metadata.offset}"
                )

            except KafkaError as e:
                self.messages_failed += 1
                print(f"✗ Kafka publish failed: {e}")
            except Exception as e:
                self.messages_failed += 1
                print(f"✗ Unexpected error during publish: {e}")

    def close(self):
        """Close Kafka producer and print statistics"""
        if self.producer:
            print("\nFlushing Kafka producer...")
            self.producer.flush(timeout=10)
            self.producer.close()

        print(f"\n{'=' * 80}")
        print("KAFKA PUBLISHER STATISTICS")
        print(f"{'=' * 80}")
        print(f"Messages sent: {self.messages_sent}")
        print(f"Messages failed: {self.messages_failed}")
        print(f"{'=' * 80}\n")


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

        # Initialize Kafka publisher
        self.kafka_publisher = VLMKafkaSignalPublisher(
            kafka_config, topic, dry_run, detect_hints=bool(nvinfer_config)
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
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
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

        # Video converter
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvvidconv.set_property("nvbuf-memory-type", 0)

        # Caps filter for RGB
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        caps_rgb = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGB")
        caps_filter.set_property("caps", caps_rgb)

        # VLM plugin
        nvvllm = Gst.ElementFactory.make("nvvllmvlm", "vlm-infer")

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
            osd_tee = self._build_osd_branch(self.osd_output_path, self.seg_mode)

        # Link pipeline
        #   With OSD tee:  streammux → nvinfer → tee ─┬→ queue_vlm → nvvidconv → caps → nvvllm → sink
        #                                             └→ queue_osd → nvvideoconvert → nvdsosd → ... → filesink
        #   Without OSD:   streammux → [nvinfer →] nvvidconv → caps → nvvllm → sink
        if osd_tee is not None:
            streammux.link(nvinfer)
            nvinfer.link(osd_tee)
            queue_vlm = self.pipeline.get_by_name("queue_vlm")
            queue_vlm.link(nvvidconv)
        elif nvinfer:
            streammux.link(nvinfer)
            nvinfer.link(nvvidconv)
        else:
            streammux.link(nvvidconv)
        nvvidconv.link(caps_filter)
        caps_filter.link(nvvllm)
        nvvllm.link(sink)

        print("Pipeline built successfully\n")

        return self.pipeline

    def _build_osd_branch(self, output_path, seg_mode):
        """Attach OSD visualization branch (bbox + instance mask) via tee.

        Pipeline topology (DS 9.0, using nvosdbin convenience bin)::

            nvinfer → tee ─┬→ queue_vlm (unlimited)  → [caller: nvvllm path]
                           └→ queue_osd (leaky down) → nvosdbin
                               → nvv4l2h264enc → h264parse → qtmux → filesink

        Design notes:
        - ``nvosdbin`` (DS 9.0) internally wraps ``queue → nvvidconv → queue
          → nvdsosd``, so we don't need to manage RGBA caps or format
          conversions ourselves. Its sink accepts almost any raw format
          (NV12 from nvinfer included).
        - queue_vlm: unlimited so VLM inference backpressure never stalls
          the tee. VLM plugin already has its own internal Python queue.
        - queue_osd: leaky=downstream drops frames rather than blocking the
          tee if the encoder falls behind real-time. Visualization is
          best-effort; VLM accuracy takes priority.
        - Encoder bitrate units differ: x264enc kbps, nvv4l2h264enc bps.
          Prefer nvv4l2h264enc (GPU) when available on DeepStream images.

        Returns the ``tee`` element so the caller can link ``nvinfer → tee``.
        Returns None when required DS elements or an H264 encoder is
        missing (VLM branch keeps running without OSD output).
        """
        tee = Gst.ElementFactory.make("tee", "detect_tee")
        queue_vlm = Gst.ElementFactory.make("queue", "queue_vlm")
        queue_osd = Gst.ElementFactory.make("queue", "queue_osd")
        nvosdbin = Gst.ElementFactory.make("nvosdbin", "nvosdbin")
        if nvosdbin is None:
            print("✗ Failed to create nvosdbin (OSD output disabled)")
            return None
        nvosdbin.set_property("display-text", True)
        nvosdbin.set_property("display-bbox", True)
        nvosdbin.set_property("display-mask", bool(seg_mode))

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264enc")
        encoder_is_gpu = encoder is not None
        if encoder is None:
            encoder = Gst.ElementFactory.make("x264enc", "h264enc")
        if encoder is None:
            print("✗ No H264 encoder available (OSD output disabled)")
            return None
        if encoder_is_gpu:
            encoder.set_property("bitrate", 4_000_000)  # bps
            if encoder.find_property("maxperf-enable") is not None:
                encoder.set_property("maxperf-enable", True)
        else:
            encoder.set_property("bitrate", 4000)  # kbps
            encoder.set_property("tune", "zerolatency")
            encoder.set_property("speed-preset", "ultrafast")

        parser = Gst.ElementFactory.make("h264parse", "h264parse")
        mux = Gst.ElementFactory.make("qtmux", "qtmux")
        filesink = Gst.ElementFactory.make("filesink", "osd_sink")
        filesink.set_property("location", output_path)
        filesink.set_property("sync", False)
        filesink.set_property("async", False)

        # When falling back to x264enc, insert nvvideoconvert → I420 in system
        # memory so the CPU encoder receives a format it can handle.
        bridge = None
        bridge_caps = None
        if not encoder_is_gpu:
            bridge = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_enc")
            bridge_caps = Gst.ElementFactory.make("capsfilter", "caps_enc")
            bridge_caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        # Queue tuning: keep VLM branch lossless, OSD branch ~1s of slack
        # so downstream encoder backpressure never freezes the tee.
        queue_vlm.set_property("max-size-buffers", 0)
        queue_vlm.set_property("max-size-time", 0)
        queue_vlm.set_property("max-size-bytes", 0)
        queue_osd.set_property("leaky", 2)  # GST_QUEUE_LEAK_DOWNSTREAM
        queue_osd.set_property("max-size-buffers", 60)
        queue_osd.set_property("max-size-time", 1_000_000_000)
        queue_osd.set_property("max-size-bytes", 0)

        elements = [tee, queue_vlm, queue_osd, nvosdbin]
        if bridge is not None:
            elements += [bridge, bridge_caps]
        elements += [encoder, parser, mux, filesink]
        for el in elements:
            self.pipeline.add(el)

        # Tee fan-out requires per-branch request pads.
        for label, sink_el in (("vlm", queue_vlm), ("osd", queue_osd)):
            src_pad = tee.request_pad_simple("src_%u")
            sink_pad = sink_el.get_static_pad("sink")
            if (
                src_pad is None
                or sink_pad is None
                or src_pad.link(sink_pad) != Gst.PadLinkReturn.OK
            ):
                print(f"✗ tee → queue_{label} link failed")
                return None

        link_steps = [("queue_osd→nvosdbin", queue_osd, nvosdbin)]
        if bridge is not None:
            link_steps += [
                ("nvosdbin→bridge", nvosdbin, bridge),
                ("bridge→bridge_caps", bridge, bridge_caps),
                ("bridge_caps→encoder", bridge_caps, encoder),
            ]
        else:
            link_steps.append(("nvosdbin→encoder", nvosdbin, encoder))
        link_steps += [
            ("encoder→parser", encoder, parser),
            ("parser→mux", parser, mux),
            ("mux→filesink", mux, filesink),
        ]
        for label, a, b in link_steps:
            if not a.link(b):
                print(f"✗ OSD link failed at: {label}")
                return None

        mask_state = "ON" if seg_mode else "OFF"
        enc_name = "nvv4l2h264enc" if encoder_is_gpu else "x264enc"
        print(
            f"✓ OSD branch attached → {output_path} (display-mask={mask_state}, encoder={enc_name})"
        )
        return tee

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
        if self.output_path and self.kafka_publisher._collected_results:
            output_data = {
                "sources": [uri for uri in self.input_uris],
                "total_segments": len(self.kafka_publisher._collected_results),
                "segments": self.kafka_publisher._collected_results,
            }
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Results saved to {self.output_path}")

        # Close Kafka publisher
        self.kafka_publisher.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepStream VLM app with Kafka publishing "
        "(single-stream or multi-stream, file or RTSP sources)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
URIs can be:
  File paths:   /path/to/video.mp4  (auto-converted to file:// URI)
  File URIs:    file:///path/to/video.mp4
  RTSP streams: rtsp://user:pass@host:port/stream

Examples:
  # Single file with dry-run (console output)
  python3 vllm_ds_app_kafka_publish.py video1.mp4 --dry-run

  # RTSP stream with dry-run
  python3 vllm_ds_app_kafka_publish.py rtsp://192.168.1.100:8554/stream \\
      --dry-run

  # Single file with Kafka publishing
  python3 vllm_ds_app_kafka_publish.py video1.mp4 \\
      --kafka-bootstrap localhost:9092 \\
      --topic vlm-results

  # Multi-stream with mixed sources and Kafka
  python3 vllm_ds_app_kafka_publish.py \\
      video1.mp4 rtsp://192.168.1.100:8554/stream \\
      --kafka-bootstrap localhost:9092 \\
      --topic vlm-results
        """,
    )

    parser.add_argument(
        "sources",
        nargs="+",
        help="Video file paths or URIs to process (file paths, file://, rtsp://)",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--topic",
        default="vlm-results",
        help="Kafka topic name (default: vlm-results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages to console instead of sending to Kafka",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to YAML config file (e.g., configs/config_describe_scene_en.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to a JSON file (e.g., results/output.json)",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Enable object detection (nvinfer) before VLM for detection hint injection",
    )
    parser.add_argument(
        "--detect-config",
        default=None,
        help="Custom nvinfer config path (default: configs/config_infer_yolo26.txt)",
    )
    parser.add_argument(
        "--detect-output",
        default=None,
        help="Write OSD-annotated MP4 (bbox + seg mask) to this path. "
        "Requires --detect. Detection-only configs render bboxes; seg configs "
        "(network-type=3) additionally render instance masks.",
    )

    args = parser.parse_args()

    # Initialize config singleton before GStreamer/pipeline starts
    if args.config:
        from config_loader import get_config

        get_config(args.config)

    # Resolve nvinfer config path
    nvinfer_config = None
    seg_mode = False
    if args.detect:
        if args.detect_config:
            nvinfer_config = args.detect_config
        else:
            # Default bundled config
            script_dir = os.path.dirname(os.path.abspath(__file__))
            nvinfer_config = os.path.join(script_dir, "..", "configs", "config_infer_yolo26.txt")
            nvinfer_config = os.path.normpath(nvinfer_config)
        if not os.path.exists(nvinfer_config):
            print(f"Error: nvinfer config not found: {nvinfer_config}")
            sys.exit(1)

        # Check ONNX file exists (parse from nvinfer config)
        missing_onnx = check_onnx_exists(nvinfer_config)
        if missing_onnx is not None:
            print(f"Error: ONNX model not found: {missing_onnx}")
            print("  Export first (inside container):")
            if "yoloe" in nvinfer_config.lower() or "yolo26e" in nvinfer_config.lower():
                print(
                    "  python3 /workspace/scripts/export_yoloe.py "
                    '-w /workspace/models/yoloe-26m-seg.pt --custom-classes "vehicle,person" '
                    "--dynamic --simplify"
                )
            else:
                print("  cd /workspace/models")
                print(
                    "  wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
                )
                print("  python3 /workspace/scripts/export_yolo26.py -w yolo26m.pt --simplify")
            sys.exit(1)

        # Seg mode (network-type=3) → enables mask rendering on OSD branch
        seg_mode = is_segmentation_config(nvinfer_config)

        # Export labelfile + detector name to env so the VLM plugin picks
        # up the right class mapping when its __init__ runs during pipeline
        # build. Legacy configs without labelfile-path fall back to the
        # hardcoded YOLO26 mapping inside the plugin.
        nvinfer_props = parse_nvinfer_config(nvinfer_config)
        labelfile = nvinfer_props.get("labelfile-path", "")
        if labelfile and os.path.isfile(labelfile):
            os.environ["VLM_DETECT_LABELFILE"] = labelfile
        cfg_name = os.path.basename(nvinfer_config).lower()
        if "yolo26e" in cfg_name or "yoloe" in cfg_name:
            os.environ["VLM_DETECTOR_NAME"] = "YOLOE-26" + (" Seg" if seg_mode else "")
        else:
            os.environ["VLM_DETECTOR_NAME"] = "YOLO26"

    # Warn if --detect but detection_hints not enabled in YAML config
    if args.detect and args.config:
        from config_loader import get_config

        cfg = get_config()
        if not cfg.detection_hints_enabled:
            print(
                "Warning: --detect is enabled but detection_hints.enabled "
                "is false in config. Hints will not be injected into the "
                "VLM prompt. Use a config with detection_hints.enabled: true "
                "(e.g., config_driving_scene_with_detect.yaml)."
            )

    # Initialize GStreamer
    Gst.init(None)

    # Convert bare file paths to file:// URIs; validate files exist
    input_uris = []
    for src in args.sources:
        uri = to_uri(src)
        if uri.startswith("file://"):
            file_path = uri[len("file://") :]
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
        input_uris.append(uri)

    # Kafka configuration
    kafka_config = {"bootstrap_servers": args.kafka_bootstrap}

    # Validate --detect-output is only used with --detect
    if args.detect_output and not args.detect:
        print("Error: --detect-output requires --detect")
        sys.exit(1)

    # Create and run app
    app = VLMKafkaApp(
        input_uris=input_uris,
        kafka_config=kafka_config,
        topic=args.topic,
        dry_run=args.dry_run,
        output_path=args.output,
        nvinfer_config=nvinfer_config,
        osd_output_path=args.detect_output,
        seg_mode=seg_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
