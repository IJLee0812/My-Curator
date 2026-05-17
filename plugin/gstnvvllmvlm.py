###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
###################################################################################################

"""Shim — moved to my_curator.adapters.gst.nvvllmvlm.  Removed in R-7.

Keeps ``import gstnvvllmvlm`` working for callers that rely on the bare-name
import (currently src/vllm_ds_app_kafka_publish.py:47); R-5 retires that
bare import.
"""

from my_curator.adapters.gst.nvvllmvlm import NvVllmVLM  # noqa: F401

__all__ = ["NvVllmVLM"]
