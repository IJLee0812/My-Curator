"""Integration tests for the P3-1 embedder worker.

Tests use AsyncMock DALs and a mock model — no GPU, compose stack, torch,
or Pillow required.  ``load_frames`` is patched so the worker logic can be
exercised on a bare pytest venv.

Run:
    pytest tests/integration/test_embedder.py -q
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model_mock(dim: int = 768) -> MagicMock:
    mock = MagicMock()
    mock.embed.return_value = [0.1] * dim
    return mock


def _make_minio_mock() -> AsyncMock:
    return AsyncMock()


def _make_milvus_mock() -> AsyncMock:
    return AsyncMock()


# ── test_single_clip_embed ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_clip_embed():
    """Full path: message → load_frames → model.embed → Milvus.upsert."""
    from services.embedder.worker import EmbedderWorker

    mock_tensor = MagicMock()
    mock_minio = _make_minio_mock()
    mock_milvus = _make_milvus_mock()
    mock_model = _make_model_mock()

    clip_id = str(uuid.uuid4())
    msg = {
        "clip_id": clip_id,
        "frames_blob_uri": f"frames/sess/{clip_id}",
        "segment": {"duration": 5.0},
    }

    with patch(
        "services.embedder.frame_loader.load_frames",
        new=AsyncMock(return_value=mock_tensor),
    ):
        worker = EmbedderWorker(mock_model, mock_minio, mock_milvus)
        await worker.handle(msg)

    assert worker.embedded == 1
    assert worker.skipped == 0
    assert worker.errors == 0

    # model.embed received the tensor from load_frames
    mock_model.embed.assert_called_once_with(mock_tensor)

    # Milvus upsert called with matching clip_id and 768-dim vector
    mock_milvus.upsert.assert_called_once()
    upsert_clip_id, upsert_vec = mock_milvus.upsert.call_args[0]
    assert str(upsert_clip_id) == clip_id
    assert len(upsert_vec) == 768


# ── test_bulk_embed_parquet ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bulk_embed_parquet(tmp_path):
    """bulk_embed: N parquet rows → batch_upsert → row count matches."""
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    from services.embedder.worker import EmbedderWorker

    n_rows = 5
    clip_ids = [str(uuid.uuid4()) for _ in range(n_rows)]
    table = pyarrow.table(
        {
            "clip_id": clip_ids,
            "frames_blob_uri": [f"frames/sess/{c}" for c in clip_ids],
            "duration": [5.0] * n_rows,
        }
    )
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(table, str(parquet_path))

    mock_tensor = MagicMock()
    mock_minio = _make_minio_mock()
    mock_milvus = _make_milvus_mock()
    mock_model = _make_model_mock()

    with patch(
        "services.embedder.frame_loader.load_frames",
        new=AsyncMock(return_value=mock_tensor),
    ):
        worker = EmbedderWorker(mock_model, mock_minio, mock_milvus)
        embedded = await worker.bulk_embed(str(parquet_path))

    assert embedded == n_rows
    assert mock_milvus.batch_upsert.called
    total_upserted = sum(len(call[0][0]) for call in mock_milvus.batch_upsert.call_args_list)
    assert total_upserted == n_rows


# ── test_partial_segment_skip ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_segment_skip():
    """Segment duration < 3 s must not produce a Milvus entry."""
    from services.embedder.worker import EmbedderWorker

    worker = EmbedderWorker(_make_model_mock(), _make_minio_mock(), _make_milvus_mock())

    await worker.handle(
        {
            "clip_id": str(uuid.uuid4()),
            "frames_blob_uri": "frames/sess/x",
            "segment": {"duration": 2.9},
        }
    )

    assert worker.skipped == 1
    assert worker.embedded == 0
    worker._milvus.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_no_frames_blob_uri_skip():
    """Messages without frames_blob_uri (legacy or partial-segment) are skipped."""
    from services.embedder.worker import EmbedderWorker

    worker = EmbedderWorker(_make_model_mock(), _make_minio_mock(), _make_milvus_mock())

    await worker.handle(
        {
            "clip_id": str(uuid.uuid4()),
            "segment": {"duration": 5.0},
            # frames_blob_uri intentionally absent
        }
    )

    assert worker.skipped == 1
    worker._milvus.upsert.assert_not_called()


# ── test_milvus_dim_768 ───────────────────────────────────────────────────────


def test_milvus_dim_768():
    """MilvusRepository.DIM must equal 768 after the P3-1 fix."""
    from src.storage.milvus import DIM

    assert DIM == 768, f"Expected DIM=768, got {DIM}"
