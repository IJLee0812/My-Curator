"""Resolve minio:// (and legacy bare bucket/key) blob_uri to a presigned URL.

Supports two formats:
  minio://bucket/key          new canonical form
  bucket/key                  legacy form (clips/, raw/, frames/, artifacts/)

stream:// and file:// are not handled here; callers must dispatch before
calling this function.
"""

from __future__ import annotations

from src.storage.minio import MinIORepository


async def get_presigned_url(
    minio: MinIORepository,
    blob_uri: str,
    *,
    expires_in: int = 3600,
) -> str | None:
    """Return a presigned GET URL for a MinIO-backed blob_uri.

    Returns None for unrecognised or non-MinIO schemes so the caller
    can fall back to a null presigned_url without raising.
    """
    if blob_uri.startswith("minio://"):
        path = blob_uri[len("minio://"):]
    elif blob_uri.startswith(("stream://", "file://")):
        return None
    else:
        path = blob_uri

    bucket, _, key = path.partition("/")
    if not bucket or not key:
        return None

    return await minio.presigned_url(bucket, key, expires_in=expires_in)
