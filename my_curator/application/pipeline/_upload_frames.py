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

"""Background-thread MinIO frame upload helper (extracted from the P2-4 god-module)."""


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
