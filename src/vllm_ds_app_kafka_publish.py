###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
###################################################################################################

"""Shim — split into my_curator.application.pipeline.{publisher, ds_app,
osd_branch, _upload_frames} + my_curator.cli.run_pipeline.  Removed in R-7.

Re-exports the public surface (VLMKafkaSignalPublisher, VLMKafkaApp, main,
_upload_frames_sync, GI_AVAILABLE, KAFKA_AVAILABLE) so existing test imports
``from vllm_ds_app_kafka_publish import …`` keep resolving through the
conftest sys.path injection.
"""

from my_curator.application.pipeline._upload_frames import _upload_frames_sync  # noqa: F401
from my_curator.application.pipeline.ds_app import GI_AVAILABLE, VLMKafkaApp  # noqa: F401
from my_curator.application.pipeline.publisher import (  # noqa: F401
    KAFKA_AVAILABLE,
    VLMKafkaSignalPublisher,
)
from my_curator.cli.run_pipeline import main  # noqa: F401

__all__ = [
    "GI_AVAILABLE",
    "KAFKA_AVAILABLE",
    "VLMKafkaApp",
    "VLMKafkaSignalPublisher",
    "_upload_frames_sync",
    "main",
]


if __name__ == "__main__":
    main()
