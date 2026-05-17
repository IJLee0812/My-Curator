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

import concurrent.futures
import json
import os
import re
import sys
import time
import uuid as _uuid_module
from typing import Optional

try:
    import gi

    gi.require_version("Gst", "1.0")  # noqa: E402, I003, BLK100
    # Register the custom plugin
    import gstnvvllmvlm  # noqa: E402
    from gi.repository import GLib, Gst  # noqa: E402, I003

    Gst.Element.register(None, "nvvllmvlm", Gst.Rank.NONE, gstnvvllmvlm.NvVllmVLM)
    GI_AVAILABLE = True
except (ImportError, ValueError):
    GI_AVAILABLE = False

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
    load_class_mapping,
    move_built_engine,
    parse_nvinfer_config,
    parse_vlm_json,
    to_uri,
    validate_driving_scene_json,
)


def _upload_frames_sync(minio_client, key_prefix: str, frames, bucket: str) -> None:
    """Upload 8 JPEG frames to MinIO synchronously (runs in background thread).

    Args:
        minio_client: boto3 S3 client.
        key_prefix: MinIO key prefix, e.g. ``frames/{session_id}/{clip_id}``.
        frames: ``[8, C, H, W]`` uint8 CPU tensor.
        bucket: MinIO bucket name.
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image

    for i in range(frames.shape[0]):
        arr = frames[i].permute(1, 2, 0).numpy().astype(np.uint8)  # [H, W, C]
        img = Image.fromarray(arr, "RGB").resize((336, 336), Image.BILINEAR)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        minio_client.put_object(
            Bucket=bucket,
            Key=f"{key_prefix}/frame_{i}.jpg",
            Body=buf.getvalue(),
            ContentType="image/jpeg",
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
        aggregator=None,
        scout_config=None,
        source_map: dict[int, tuple[str | None, str | None]] | None = None,
    ):
        """
        Initialize Kafka publisher.

        Args:
            kafka_config: Kafka connection configuration
            topic: Topic name to publish to (used for legacy path and dry-run default)
            dry_run: If True, print messages instead of sending to Kafka
            detect_hints: If True, include detect_hints flag in message metadata
            aggregator: BestOfNAggregator instance (P2-4; None = legacy path)
            scout_config: ScoutConfig instance (P2-4; None = legacy path)
            source_map: Mapping of stream_id → (source_clip_id, source_video_path).
                Auto-derived from input URIs at app init time (P3-4+). Both
                values are included in every published message so the consumer
                can persist source_clip_id and construct a file:// blob_uri.
        """
        self.topic = topic
        self.dry_run = dry_run
        self.detect_hints = detect_hints
        self.producer: KafkaProducer | None = None
        self.messages_sent = 0
        self.messages_failed = 0
        self._collected_results: list = []
        self._source_map: dict[int, tuple[str | None, str | None]] = source_map or {}

        # P3-1: frame capture — MinIO boto3 client + background upload executor
        self._upload_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="frame-upload"
        )
        self._session_id = os.environ.get("CURATOR_SESSION_ID", "default")
        self._frames_bucket = os.environ.get("MINIO_FRAMES_BUCKET", "frames")
        self._minio_client = None
        _ep = os.environ.get("MINIO_ENDPOINT")
        _ak = os.environ.get("MINIO_ACCESS_KEY")
        _sk = os.environ.get("MINIO_SECRET_KEY")
        if _ep and _ak and _sk:
            try:
                import boto3
                from botocore.config import Config as _BotoConfig

                self._minio_client = boto3.client(
                    "s3",
                    endpoint_url=_ep,
                    aws_access_key_id=_ak,
                    aws_secret_access_key=_sk,
                    config=_BotoConfig(signature_version="s3v4"),
                )
            except Exception as _exc:
                print(f"✗ MinIO client init failed ({_exc}) — frame capture disabled")

        # P2-4: Scout + Aggregator curation (None = legacy path, backward-compatible)
        self._aggregator = aggregator
        self._scout_config = scout_config
        self._scout = None  # lazy-init on first vlm-result via element.get_llm()
        self._partial_count: int = 0  # consecutive partial-failure counter for N=1 fallback

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

    def _source_fields(self, stream_id: int) -> dict:
        """Return source_clip_id / source_video_path fields for a Kafka message."""
        clip_id, video_path = self._source_map.get(stream_id, (None, None))
        fields: dict = {}
        if clip_id:
            fields["source_clip_id"] = clip_id
        if video_path:
            fields["source_video_path"] = video_path
        return fields

    def on_vlm_result(self, element, stream_id, start_time, end_time, result_text):
        """Signal handler for vlm-result signal (called from _infer_thread)."""

        if self._aggregator is None or self._scout_config is None:
            # ── Legacy path (no Scout/Aggregator) — backward-compatible ────────
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
            clip_id = _uuid_module.uuid4()
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
                "clip_id": str(clip_id),
                **self._source_fields(stream_id),
            }
            self._collected_results.append(message)
            self.publish(message, stream_id)
            return

        # ── P2-4 curation path: Scout + Best-of-N Aggregator ────────────────

        # Lazy Scout init via element.get_llm()
        if self._scout is None:
            llm = element.get_llm() if element is not None and hasattr(element, "get_llm") else None
            if llm is not None:
                from src.scouts.cosmos_reason import CosmosReasonScout

                self._scout = CosmosReasonScout(llm=llm)

        # Retrieve per-segment data stored by _inference_worker / _run_vlm_batch
        ctx = None
        if element is not None and hasattr(element, "stream_contexts"):
            ctx = element.stream_contexts.get(stream_id)
        inventory: dict[str, int] = ctx.last_inventory if ctx is not None else {}
        last_inputs: dict | None = ctx.last_inputs if ctx is not None else None

        # Scout sampling (T=0.5 + T=0.7 batch; T=0.3 already computed as t0_result)
        if self._scout is not None and last_inputs is not None:
            reports = self._scout.sample(last_inputs, {}, self._scout_config, t0_result=result_text)
        else:
            # Fallback: wrap t0_result as single partial report
            from src.scouts.base import ScoutReport

            t0_temp = self._scout_config.temperatures[0] if self._scout_config.temperatures else 0.3
            reports = [
                ScoutReport(
                    text=result_text,
                    temperature=t0_temp,
                    seed=self._scout_config.seed_for(t0_temp),
                    latency_ms=0.0,
                    partial_sampling=True,
                )
            ]

        # P3-1: generate clip_id and capture 8 frames before releasing ctx.last_inputs
        clip_id = _uuid_module.uuid4()
        frames_blob_uri = None
        if (
            last_inputs is not None
            and (end_time - start_time) >= 3.0
            and self._minio_client is not None
        ):
            try:
                import numpy as np

                video_tuple = last_inputs["multi_modal_data"]["video"]
                batch_tensor = video_tuple[0]  # [T, C, H, W] cpu uint8
                T = batch_tensor.shape[0]
                indices = np.linspace(0, T - 1, 8).astype(int)
                sampled = batch_tensor[indices].cpu()
                frames_blob_uri = f"frames/{self._session_id}/{clip_id}"
                self._upload_executor.submit(
                    _upload_frames_sync,
                    self._minio_client,
                    frames_blob_uri,
                    sampled,
                    self._frames_bucket,
                )
            except Exception as _exc:
                print(f"VLMKafkaPublisher: frame capture failed: {_exc}")

        # Release per-segment resources immediately after Scout completes
        if ctx is not None:
            ctx.last_inputs = None
            ctx.last_inventory = {}

        # Best-of-N selection
        best = self._aggregator.select(reports, inventory)
        n_samples = len(reports)

        if best is None:
            # Defensive: empty reports list (shouldn't happen in practice)
            from src.scouts.base import ScoutReport

            t0_temp = self._scout_config.temperatures[0] if self._scout_config.temperatures else 0.3
            best = ScoutReport(
                text=result_text,
                temperature=t0_temp,
                seed=self._scout_config.seed_for(t0_temp),
                latency_ms=0.0,
                partial_sampling=True,
            )
            n_samples = 0

        # Routing decision
        needs_review = False
        reason = None

        if best.partial_sampling:
            needs_review = True
            reason = "partial_batch"
            self._partial_count += 1
            if self._partial_count >= 3:
                self._scout_config.n = 1
                print(
                    f"VLMKafkaPublisher: N=1 fallback activated after "
                    f"{self._partial_count} consecutive partial failures "
                    f"(stream {stream_id})"
                )
        else:
            self._partial_count = 0  # reset on success
            # Zero-grounding: inventory non-empty but no class matched in selected report
            if inventory and self._aggregator.score(best, inventory) == 0:
                needs_review = True
                reason = "zero_grounding"

        # Extract and schema-validate DNA JSON from CoT output (P2-6)
        from src.scouts.dna_validator import DNAValidator

        _validator = DNAValidator()
        dna_dict = _validator.extract_json(best.text)
        if dna_dict is None:
            needs_review = True
            reason = reason or "rejected_schema_invalid"
            json_valid = False
        else:
            _dna_valid, _ = _validator.validate(dna_dict)
            if not _dna_valid:
                needs_review = True
                reason = reason or "rejected_schema_invalid"
            json_valid = _dna_valid

        message = {
            "stream_id": stream_id,
            "timestamp": time.time(),
            "segment": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            },
            "result": best.text,
            "curation": {
                "temperature": best.temperature,
                "seed": best.seed,
                "latency_ms": round(best.latency_ms, 1),
                "partial_sampling": best.partial_sampling,
                "n_samples": n_samples,
                "needs_review": needs_review,
                "reason": reason,
            },
            "metadata": {
                "source": "vllm-ds-plugin",
                "version": "1.0",
                **({"detect_hints": True} if self.detect_hints else {}),
                "json_valid": json_valid,
            },
            "clip_id": str(clip_id),
            **({"frames_blob_uri": frames_blob_uri} if frames_blob_uri else {}),
            **self._source_fields(stream_id),
        }

        self._collected_results.append(message)

        pub_topic = (
            self._scout_config.kafka_topic_needs_review
            if needs_review
            else self._scout_config.kafka_topic_scouted
        )
        self.publish(message, stream_id, topic=pub_topic)

    def publish(self, message: dict, stream_id: int, topic: str | None = None):
        """
        Publish message to Kafka or print to console.

        Args:
            message: Message payload
            stream_id: Stream ID (used as partition key)
            topic: Kafka topic override; defaults to self.topic when None
        """
        topic = topic or self.topic
        # Use stream_id as partition key for ordering
        partition_key = f"stream_{stream_id}"

        if self.dry_run or self.producer is None:
            # Dry-run mode: print to console
            print(f"\n{'=' * 80}")
            print("📤 KAFKA MESSAGE (Dry-Run)")
            print(f"{'=' * 80}")
            print(f"Topic: {topic}")
            print(f"Key: {partition_key}")
            print(f"Value: {json.dumps(message, indent=2)}")
            print(f"{'=' * 80}\n")
            self.messages_sent += 1
        else:
            # Send to Kafka
            try:
                future = self.producer.send(topic, key=partition_key, value=message)

                # Optional: wait for acknowledgment
                record_metadata = future.get(timeout=10)

                self.messages_sent += 1
                print(
                    f"✓ Published to Kafka: stream={stream_id}, "
                    f"time={message['segment']['start_time']:.1f}s-"
                    f"{message['segment']['end_time']:.1f}s, "
                    f"topic={topic}, "
                    f"partition={record_metadata.partition}, "
                    f"offset={record_metadata.offset}"
                )

            except KafkaError as e:
                self.messages_failed += 1
                print(f"✗ Kafka publish failed: {e}")
            except Exception as e:
                self.messages_failed += 1
                print(f"✗ Unexpected error during publish: {e}")

    def get_collected_results(self) -> list:
        """Return the list of messages emitted during this run.

        Exposed as an accessor (R-3.5 preflight) so cross-module callers
        (``VLMKafkaApp.run`` JSON dump) no longer reach into the
        ``_collected_results`` private attribute.  Returns the live list by
        reference — identical semantics to the previous direct attribute
        access — so existing JSON-dump code keeps working unchanged.
        """
        return self._collected_results

    def close(self):
        """Close Kafka producer and print statistics"""
        self._upload_executor.shutdown(wait=True)
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
        source_clip_id_override=None,
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

        # P2-4: load ScoutConfig + BestOfNAggregator for curation wiring
        _scout_config = None
        _aggregator = None
        _scout_yaml = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "scout.yaml")
        )
        if os.path.exists(_scout_yaml):
            try:
                from src.scouts.aggregator import BestOfNAggregator
                from src.scouts.base import ScoutConfig

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

        # Tracker (NvDCF) — assigns stable object_id per detected object
        nvtracker = None
        if nvinfer:
            nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if nvtracker:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                tracker_config = os.path.normpath(
                    os.path.join(script_dir, "..", "configs", "config_tracker_NvDCF_perf.yml")
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

        # Attach OSD label probe: overlays "ClassName (TrackID: N)" on each bbox
        try:
            from probes.osd_label_probe import make_osd_label_probe

            osd_sink = nvosdbin.get_static_pad("sink")
            if osd_sink:
                osd_sink.add_probe(
                    Gst.PadProbeType.BUFFER,
                    make_osd_label_probe(self._class_mapping),
                )
                print("✓ OSD label probe attached (ClassName + TrackID)")
        except ImportError:
            print("✗ probes.osd_label_probe not importable; OSD labels disabled")

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
    parser.add_argument(
        "--source-clip-id",
        default=None,
        help="Override source_clip_id stored in PG (single-source only; "
        "ignored when multiple sources are provided). Defaults to filename stem.",
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

    # Convert bare file paths / session dirs to file:// URIs; validate files exist.
    # A directory source is treated as a session dir: all */video/*.mp4 under it
    # are expanded so callers can pass the session root without listing files.
    input_uris = []
    import pathlib

    for src in args.sources:
        if not src.startswith("rtsp://") and os.path.isdir(src):
            found = sorted(pathlib.Path(src).glob("*/video/*.mp4"))
            if not found:
                print(f"Error: No */video/*.mp4 files found under session dir: {src}")
                sys.exit(1)
            for mp4 in found:
                input_uris.append(f"file://{mp4.resolve()}")
        else:
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
        source_clip_id_override=args.source_clip_id,
    )
    app.run()


if __name__ == "__main__":
    main()
