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

"""OSD visualization branch (bbox + instance mask) for the DS pipeline.

Extracted from src/vllm_ds_app_kafka_publish.py::VLMKafkaApp._build_osd_branch
during the R-5 god-module split.  ``build_osd_branch`` takes the live
Gst.Pipeline + the YOLO class mapping and attaches a tee → queue_osd →
nvosdbin → encoder → filesink branch, returning the tee element so the
caller can wire ``nvinfer → tee``.
"""

from __future__ import annotations

import os

# Module-level Gst with try/except fallback.  Tests patch
# ``my_curator.application.pipeline.osd_branch.Gst`` to inject a mock; in
# real DS containers this binds to the real GStreamer Gst.  On host venvs
# without gi installed we expose ``Gst = None`` so the module is at least
# collectable — calls to build_osd_branch raise at runtime instead.
try:
    from gi.repository import Gst
except (ImportError, ValueError):
    Gst = None


def build_osd_branch(pipeline, class_mapping, output_path, seg_mode):
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
    if Gst is None:
        raise RuntimeError(
            "GStreamer / gi not available — build_osd_branch can only run in "
            "the DS container or with gi-mocked tests."
        )
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
        from my_curator.adapters.gst.probes.osd_label import make_osd_label_probe

        osd_sink = nvosdbin.get_static_pad("sink")
        if osd_sink:
            osd_sink.add_probe(
                Gst.PadProbeType.BUFFER,
                make_osd_label_probe(class_mapping),
            )
            print("✓ OSD label probe attached (ClassName + TrackID)")
    except ImportError:
        print("✗ osd_label probe not importable; OSD labels disabled")

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
        pipeline.add(el)

    # Tee fan-out requires per-branch request pads.
    for label, sink_el in (("vlm", queue_vlm), ("osd", queue_osd)):
        src_pad = tee.request_pad_simple("src_%u")
        sink_pad = sink_el.get_static_pad("sink")
        if src_pad is None or sink_pad is None or src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
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
    print(f"✓ OSD branch attached → {output_path} (display-mask={mask_state}, encoder={enc_name})")
    return tee
