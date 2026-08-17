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

"""VLMKafkaSignalPublisher — Kafka publisher driven by the DS plugin vlm-result signal.

Extracted from src/vllm_ds_app_kafka_publish.py (R-5 god-module split).
Handles Scout sampling, Best-of-N selection, DNA schema validation, MinIO
frame upload, and Kafka publishing.  No GStreamer pipeline ownership.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import uuid as _uuid_module

from my_curator.adapters.gst.utils import parse_vlm_json, validate_driving_scene_json
from my_curator.application.pipeline._upload_frames import _upload_frames_sync

# Kafka imports (with graceful fallback)
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Run: pip install kafka-python")


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
                from my_curator.adapters.scout.cosmos_reason import CosmosReasonScout

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
            from my_curator.domain.scout.base import ScoutReport

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
        if last_inputs is not None and self._minio_client is not None:
            try:
                import numpy as np

                video_tuple = last_inputs["multi_modal_data"]["video"]
                batch_tensor = video_tuple[0]  # [T, C, H, W] cpu uint8
                T = batch_tensor.shape[0]
                # Gate on frames actually sampled, not wall-clock length: a
                # trailing segment is short but still carries frames, and
                # linspace repeats indices when T < 8.
                if T >= 2:
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
            from my_curator.domain.scout.base import ScoutReport

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

        # Mirror the consumer's normalize+inject so json_valid == what gets stored.
        from my_curator.domain.scout.dna_normalizer import ensure_managed_fields, normalize_dna
        from my_curator.domain.scout.dna_validator import DNAValidator

        _validator = DNAValidator()
        dna_dict = _validator.extract_json(best.text)
        if dna_dict is None:
            needs_review = True
            reason = reason or "rejected_schema_invalid"
            json_valid = False
        else:
            dna_dict = normalize_dna(dna_dict)
            ensure_managed_fields(
                dna_dict,
                dna_version="0.2.0",
                clip_id=clip_id,
                start_s=start_time,
                end_s=end_time,
            )
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
