"""MinIO DAL for My-Curator (boto3 S3-compatible client, sync-under-async).

Public surface:
  MinIORepository.create()   — async factory (boto3 S3 client)
  upload_bytes / upload_file — write objects
  download_bytes / download_file — read objects
  object_exists              — HEAD check
  delete_object              — remove a single object
  presigned_url              — time-limited GET URL (P3-2 FastAPI)
  put_bucket_lifecycle       — set expiry rule (raw/ → 1 day)

Object key layout contract (callers are responsible for constructing keys):
  raw/       {session_id}/{clip_id}.mp4          original video hot-cache (1-day lifecycle)
  clips/     {session_id}/{clip_id}.mp4          segmented clip
  frames/    {clip_id}/{frame_idx:06d}.jpg       extracted frames @ 1 FPS (UI thumbnails)
  artifacts/ {clip_id}/scout_reports.jsonl       Scout VLM raw outputs
             {clip_id}/dna_v{version}.json       DNA archive

raw/ eviction policy: S3 lifecycle rule (1-day, MinIO minimum) + explicit
delete_object() call after clips/ upload succeeds in the ingest pipeline.
"""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_S3_CONFIG = Config(signature_version="s3v4")


def _make_client(endpoint_url: str, access_key: str, secret_key: str) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=_S3_CONFIG,
    )


class MinIORepository:
    def __init__(self, client: Any, endpoint_url: str) -> None:
        self._client = client
        self._endpoint_url = endpoint_url

    @classmethod
    async def create(
        cls,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ) -> MinIORepository:
        """Async factory — creates a boto3 S3 client pointed at MinIO."""
        client = await asyncio.to_thread(_make_client, endpoint_url, access_key, secret_key)
        return cls(client, endpoint_url)

    # boto3 clients have no meaningful close(); kept for interface symmetry.
    async def close(self) -> None:
        pass

    # ── writes ────────────────────────────────────────────────────────────────

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload raw bytes to *bucket/key*."""
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    async def upload_file(
        self,
        bucket: str,
        key: str,
        local_path: Path,
        content_type: str | None = None,
    ) -> None:
        """Upload a local file to *bucket/key* (content-type auto-detected if omitted)."""
        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(local_path))
            content_type = guessed or "application/octet-stream"
        extra = {"ContentType": content_type}
        await asyncio.to_thread(
            self._client.upload_file,
            str(local_path),
            bucket,
            key,
            ExtraArgs=extra,
        )

    # ── reads ─────────────────────────────────────────────────────────────────

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        """Download *bucket/key* and return its contents as bytes."""

        def _get() -> bytes:
            resp = self._client.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()

        return await asyncio.to_thread(_get)

    async def download_file(self, bucket: str, key: str, local_path: Path) -> None:
        """Download *bucket/key* to *local_path*."""
        await asyncio.to_thread(
            self._client.download_file,
            bucket,
            key,
            str(local_path),
        )

    async def object_exists(self, bucket: str, key: str) -> bool:
        """Return True if *bucket/key* exists (HEAD request)."""

        def _head() -> bool:
            try:
                self._client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    return False
                raise

        return await asyncio.to_thread(_head)

    # ── delete ────────────────────────────────────────────────────────────────

    async def delete_object(self, bucket: str, key: str) -> None:
        """Remove a single object.  No-op if the key does not exist."""
        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=bucket,
            Key=key,
        )

    # ── presigned URL ─────────────────────────────────────────────────────────

    async def presigned_url(
        self,
        bucket: str,
        key: str,
        *,
        expires_in: int = 3600,
    ) -> str:
        """Return a time-limited presigned GET URL for *bucket/key*."""

        def _sign() -> str:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )

        return await asyncio.to_thread(_sign)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def put_bucket_lifecycle(self, bucket: str, expiry_days: int) -> None:
        """Apply an expiry lifecycle rule to *bucket* (minimum 1 day on MinIO/S3).

        Replaces any existing lifecycle configuration on the bucket.
        Intended for raw/ hot-cache eviction:
            await repo.put_bucket_lifecycle("raw", expiry_days=1)
        """
        rule = {
            "ID": f"auto-evict-{bucket}",
            "Filter": {"Prefix": ""},
            "Status": "Enabled",
            "Expiration": {"Days": expiry_days},
        }
        await asyncio.to_thread(
            self._client.put_bucket_lifecycle_configuration,
            Bucket=bucket,
            LifecycleConfiguration={"Rules": [rule]},
        )
