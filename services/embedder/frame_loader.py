"""Download JPEG frames from MinIO and build the [1, 8, 3, 336, 336] tensor for Cosmos-Embed1.

Host-importable: PIL, numpy, and torch are lazy-imported inside ``load_frames``
so this module can be collected by pytest on a bare venv without those packages.
"""

from __future__ import annotations

NUM_FRAMES = 8
FRAME_SIZE = 336  # Cosmos-Embed1-336p spatial resolution


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
        data = await minio.download_bytes(bucket, key)
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.size != (frame_size, frame_size):
            img = img.resize((frame_size, frame_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        frames.append(t)
    stacked = torch.stack(frames)  # [8, 3, H, W]
    return stacked.unsqueeze(0)  # [1, 8, 3, H, W]
