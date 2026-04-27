"""Integration tests for MinIORepository (P1-5).

Requires the MinIO service from infra/compose.base.yml to be running:
    docker compose -f infra/compose.base.yml --env-file .env up -d minio

Tests are auto-skipped when MinIO is not reachable on :9000.

Credentials are read from the repo-root .env file (if present) so the tests
work against the local compose stack without manually exporting env vars.
Fallback: MINIO_USER=minio-admin / MINIO_PASSWORD=minio-password.

Each test uses a unique key prefix derived from the test run UUID so tests
are fully isolated.  Keys are deleted in fixture teardown.
"""

from __future__ import annotations

import asyncio
import pathlib
import socket
import urllib.request
import uuid

import pytest
import pytest_asyncio

from src.storage.minio import MinIORepository

REPO_ROOT = pathlib.Path(__file__).parents[2]

pytestmark = pytest.mark.integration


# ── credentials & availability ────────────────────────────────────────────────


def _read_dotenv() -> dict[str, str]:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


_ENV = _read_dotenv()
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_USER = _ENV.get("MINIO_USER", "minio-admin")
MINIO_PASSWORD = _ENV.get("MINIO_PASSWORD", "minio-password")

# Buckets created by compose minio-init sidecar.
TEST_BUCKET = "clips"
RAW_BUCKET = "raw"


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


MINIO_UP = _tcp_open("127.0.0.1", 9000)

requires_minio = pytest.mark.skipif(
    not MINIO_UP,
    reason="MinIO not up (run: docker compose -f infra/compose.base.yml --env-file .env up -d)",
)

pytestmark = [pytest.mark.integration, requires_minio]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def repo():
    r = await MinIORepository.create(MINIO_ENDPOINT, MINIO_USER, MINIO_PASSWORD)
    yield r
    await r.close()


@pytest.fixture
def prefix() -> str:
    """Unique key prefix per test — avoids cross-test collisions."""
    return f"_test_{uuid.uuid4().hex}/"


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_upload_download_bytes(repo: MinIORepository, prefix: str):
    """Uploaded bytes are returned byte-for-byte on download."""
    key = f"{prefix}hello.bin"
    payload = b"My-Curator P1-5 test payload"

    await repo.upload_bytes(TEST_BUCKET, key, payload, content_type="application/octet-stream")
    result = await repo.download_bytes(TEST_BUCKET, key)

    assert result == payload

    await repo.delete_object(TEST_BUCKET, key)


async def test_upload_download_file(repo: MinIORepository, prefix: str, tmp_path: pathlib.Path):
    """upload_file / download_file preserve file contents."""
    key = f"{prefix}video.mp4"
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    src.write_bytes(b"\x00\x01\x02video-bytes")

    await repo.upload_file(TEST_BUCKET, key, src)
    await repo.download_file(TEST_BUCKET, key, dst)

    assert dst.read_bytes() == src.read_bytes()

    await repo.delete_object(TEST_BUCKET, key)


async def test_object_exists_true(repo: MinIORepository, prefix: str):
    """object_exists returns True after upload."""
    key = f"{prefix}exists.bin"
    await repo.upload_bytes(TEST_BUCKET, key, b"data")
    assert await repo.object_exists(TEST_BUCKET, key) is True
    await repo.delete_object(TEST_BUCKET, key)


async def test_object_exists_false(repo: MinIORepository):
    """object_exists returns False for a key that was never uploaded."""
    key = f"_test_{uuid.uuid4().hex}/nonexistent.bin"
    assert await repo.object_exists(TEST_BUCKET, key) is False


async def test_delete_object(repo: MinIORepository, prefix: str):
    """delete_object removes the object; subsequent exists check returns False."""
    key = f"{prefix}to_delete.bin"
    await repo.upload_bytes(TEST_BUCKET, key, b"delete-me")
    assert await repo.object_exists(TEST_BUCKET, key) is True

    await repo.delete_object(TEST_BUCKET, key)
    assert await repo.object_exists(TEST_BUCKET, key) is False


async def test_upload_overwrites(repo: MinIORepository, prefix: str):
    """Re-uploading the same key replaces the previous content."""
    key = f"{prefix}overwrite.bin"
    await repo.upload_bytes(TEST_BUCKET, key, b"original")
    await repo.upload_bytes(TEST_BUCKET, key, b"replaced")

    result = await repo.download_bytes(TEST_BUCKET, key)
    assert result == b"replaced"

    await repo.delete_object(TEST_BUCKET, key)


async def test_presigned_url(repo: MinIORepository, prefix: str):
    """Presigned URL is accessible via plain HTTP and returns the correct content."""
    key = f"{prefix}presigned.txt"
    content = b"presigned-content"
    await repo.upload_bytes(TEST_BUCKET, key, content, content_type="text/plain")

    url = await repo.presigned_url(TEST_BUCKET, key, expires_in=60)
    assert url.startswith("http")

    with urllib.request.urlopen(url) as resp:
        fetched = resp.read()
    assert fetched == content

    await repo.delete_object(TEST_BUCKET, key)


async def test_raw_lifecycle_policy(repo: MinIORepository):
    """put_bucket_lifecycle applies a 1-day expiry rule to the raw/ bucket."""
    await repo.put_bucket_lifecycle(RAW_BUCKET, expiry_days=1)

    def _get_lifecycle():
        return repo._client.get_bucket_lifecycle_configuration(Bucket=RAW_BUCKET)

    config = await asyncio.to_thread(_get_lifecycle)
    rules = config.get("Rules", [])
    assert any(
        r.get("Status") == "Enabled" and r.get("Expiration", {}).get("Days") == 1 for r in rules
    ), f"Expected 1-day Enabled rule, got: {rules}"
