###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
###################################################################################################

"""Shim — moved to my_curator.adapters.gst.probes.osd_label.  Removed in R-7."""

from my_curator.adapters.gst.probes.osd_label import (  # noqa: F401
    make_osd_label_probe,
)

__all__ = ["make_osd_label_probe"]
