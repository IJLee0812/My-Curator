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

"""GStreamer pad probe that overlays ClassName (TrackID: N) labels via pyservicemaker.

Attach to the nvosdbin sink pad to draw track-ID annotations on each bbox.
Requires DS 9.0 pyservicemaker (pyds is deprecated in DS 9.0).
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from pyservicemaker import Buffer, osd
from vlm_utils import format_object_label

_MAX_TEXTS_PER_DISPLAY_META = 16


def make_osd_label_probe(class_mapping: dict[int, str]):
    """Return a Gst.PadProbeType.BUFFER callback that stamps track-ID labels.

    Each detected object gets 'ClassName (TrackID: N)' rendered just above
    its bounding box. When object_id is absent (no tracker in pipeline), the
    label falls back to plain 'ClassName'.

    Args:
        class_mapping: class_id → label name (from load_class_mapping).

    Returns:
        Callable suitable for pad.add_probe(Gst.PadProbeType.BUFFER, cb).
    """

    def _cb(pad, info):
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        buf = Buffer(gst_buffer)
        batch_meta = buf.batch_meta
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        for frame_meta in batch_meta.frame_items:
            display_meta = batch_meta.acquire_display_meta()
            texts_in_meta = 0

            for object_meta in frame_meta.object_items:
                if texts_in_meta >= _MAX_TEXTS_PER_DISPLAY_META:
                    frame_meta.append(display_meta)
                    display_meta = batch_meta.acquire_display_meta()
                    texts_in_meta = 0

                # Suppress the default nvinfer/nvtracker label so only our text renders
                object_meta.text_params.display_text = ""

                label = class_mapping.get(object_meta.class_id, f"class_{object_meta.class_id}")
                track_id = getattr(object_meta, "object_id", None)
                display_text = format_object_label(label, track_id)

                rect = object_meta.rect_params
                text = osd.Text()
                text.display_text = display_text.encode("ascii", errors="replace")
                text.x_offset = int(rect.left)
                text.y_offset = max(0, int(rect.top) - 16)
                text.font.name = osd.FontFamily.Serif
                text.font.size = 10
                text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                text.set_bg_color = True
                text.bg_color = osd.Color(0.0, 0.0, 0.0, 0.8)
                display_meta.add_text(text)
                texts_in_meta += 1

            if texts_in_meta > 0:
                frame_meta.append(display_meta)

        return Gst.PadProbeReturn.OK

    return _cb
