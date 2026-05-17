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

"""DeepStream-VLM pipeline CLI entrypoint (R-5 successor of vllm_ds_app_kafka_publish.main).

Implements the config-priming gate (plan §4 R-5 / §7 risk row): the YAML config
path is parsed out of ``sys.argv`` and fed into the ``get_config()`` Singleton
*before* any ``my_curator.adapters.gst.*`` import triggers the plugin module
load.  Only after that priming step are GStreamer / VLMKafkaApp imported.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    """Main entry point — orchestrates the DS pipeline run."""

    # ── Step 1: prime the config Singleton BEFORE any my_curator.adapters.gst.*
    # import.  gstnvvllmvlm calls get_config() at module load to read model
    # paths, segment lengths, etc.  Importing ds_app (which imports nvvllmvlm
    # and registers the Gst element) without priming first would freeze the
    # Singleton at default values.
    for i, arg in enumerate(sys.argv):
        if arg in ("-c", "--config") and i + 1 < len(sys.argv):
            from my_curator.adapters.gst.config_loader import get_config

            get_config(sys.argv[i + 1])
            break

    # ── Step 2: argparse.
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
  python3 -m my_curator.cli.run_pipeline video1.mp4 --dry-run

  # RTSP stream with dry-run
  python3 -m my_curator.cli.run_pipeline rtsp://192.168.1.100:8554/stream \\
      --dry-run

  # Single file with Kafka publishing
  python3 -m my_curator.cli.run_pipeline video1.mp4 \\
      --kafka-bootstrap localhost:9092 \\
      --topic vlm-results

  # Multi-stream with mixed sources and Kafka
  python3 -m my_curator.cli.run_pipeline \\
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

    # Resolve nvinfer config path
    nvinfer_config = None
    seg_mode = False
    from my_curator.adapters.gst.utils import (
        check_onnx_exists,
        is_segmentation_config,
        parse_nvinfer_config,
        to_uri,
    )

    if args.detect:
        if args.detect_config:
            nvinfer_config = args.detect_config
        else:
            # Default bundled config — resolve relative to repo root (this module
            # lives at my_curator/cli/run_pipeline.py — 3 levels deep).
            script_dir = os.path.dirname(os.path.abspath(__file__))
            nvinfer_config = os.path.join(
                script_dir, "..", "..", "configs", "config_infer_yolo26.txt"
            )
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
        from my_curator.adapters.gst.config_loader import get_config

        cfg = get_config()
        if not cfg.detection_hints_enabled:
            print(
                "Warning: --detect is enabled but detection_hints.enabled "
                "is false in config. Hints will not be injected into the "
                "VLM prompt. Use a config with detection_hints.enabled: true "
                "(e.g., config_driving_scene_with_detect.yaml)."
            )

    # ── Step 3: GStreamer import + VLMKafkaApp.  Order matters — the
    # ds_app module triggers ``Gst.Element.register(nvvllmvlm, ...)`` at
    # import time, which depends on the get_config() Singleton primed in
    # Step 1.
    from gi.repository import Gst

    from my_curator.application.pipeline.ds_app import VLMKafkaApp

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
