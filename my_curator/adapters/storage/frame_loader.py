"""Download JPEG frames from MinIO and build the [1, 8, 3, 336, 336] tensor for Cosmos-Embed1.

Host-importable: PIL, numpy, and torch are lazy-imported inside ``load_frames``
so this module can be collected by pytest on a bare venv without those packages.
"""

from __future__ import annotations

import asyncio
import logging

NUM_FRAMES = 8
FRAME_SIZE = 336  # Cosmos-Embed1-336p spatial resolution

# DS pipeline publishes the Kafka curation.clip.scouted message immediately and
# uploads the 8 frames asynchronously via a background executor.  The embedder
# worker therefore can race the upload and observe NoSuchKey for the first
# fetch.  Retry with exponential back-off so the message survives the race.
_RETRY_DELAYS_S = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

log = logging.getLogger(__name__)


async def _download_with_retry(minio, bucket: str, key: str) -> bytes:
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS_S)):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await minio.download_bytes(bucket, key)
        except Exception as exc:  # botocore NoSuchKey or transient S3 errors
            last_exc = exc
            if "NoSuchKey" not in str(type(exc).__name__) and "NoSuchKey" not in str(exc):
                raise
            log.debug("frame %s not yet uploaded (attempt %d); retrying", key, attempt + 1)
    assert last_exc is not None
    raise last_exc


async def load_frames(
    minio,
    bucket: str,
    key_prefix: str,
    *,
    num_frames: int = NUM_FRAMES,
    frame_size: int = FRAME_SIZE,
):
    """Download ``num_frames`` JPEG frames from MinIO and return a uint8 tensor.

    Args:
        minio: ``MinIORepository`` instance.
        bucket: MinIO bucket name (e.g. ``"frames"``).
        key_prefix: Common prefix of frame keys (e.g. ``"frames/sess/clip-uuid"``).
            Frame keys are constructed as ``{key_prefix}/frame_{i}.jpg``.
        num_frames: Number of frames; must equal Cosmos-Embed1's ``num_video_frames`` (8).
        frame_size: Target H/W; Cosmos-Embed1-336p requires 336.

    Returns:
        ``[1, 8, 3, 336, 336]`` uint8 :class:`torch.Tensor`.
    """
    from io import BytesIO

    import numpy as np
    import torch
    from PIL import Image

    frames = []
    for i in range(num_frames):
        key = f"{key_prefix}/frame_{i}.jpg"
        data = await _download_with_retry(minio, bucket, key)
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.size != (frame_size, frame_size):
            img = img.resize((frame_size, frame_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        frames.append(t)
    stacked = torch.stack(frames)  # [8, 3, H, W]
    return stacked.unsqueeze(0)  # [1, 8, 3, H, W]
