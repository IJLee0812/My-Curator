"""Directory-traversal hardening for `file://` blob_uri resolution.

Covers ``my_curator.adapters.storage.streaming.resolve_path`` and
``my_curator.domain.timestamp.get_precise_times``: every
``blob_uri`` payload that escapes the configured ``VIDEO_DATA_ROOT``
mount must either raise ``HTTPException(403)`` (streaming path) or
silently fall back to the caller-supplied ``start_s`` / ``end_s``
values (sidecar path) — never open or serve a file outside the root.

References:
  Issue #42 — Harden file:// blob_uri path resolution against directory
              traversal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from my_curator.adapters.storage.streaming import (
    _safe_resolve,
    resolve_path,
    serve_segment,
)
from my_curator.domain.timestamp import get_precise_times

# ── _safe_resolve direct tests ────────────────────────────────────────────────


@pytest.mark.unit
class TestSafeResolveContainment:
    def test_benign_relative_inside_root(self, tmp_path):
        (tmp_path / "session" / "video").mkdir(parents=True)
        target = tmp_path / "session" / "video" / "clip.mp4"
        target.touch()
        out = _safe_resolve(tmp_path, "session/video/clip.mp4")
        assert out == target.resolve()

    def test_relative_escape_raises_403(self, tmp_path):
        (tmp_path / "inside").mkdir()
        with pytest.raises(HTTPException) as exc:
            _safe_resolve(tmp_path / "inside", "../../etc/passwd")
        assert exc.value.status_code == 403

    def test_symlink_escape_raises_403(self, tmp_path):
        inside = tmp_path / "inside"
        inside.mkdir()
        try:
            (inside / "escape").symlink_to("/etc")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this filesystem")
        with pytest.raises(HTTPException) as exc:
            _safe_resolve(inside, "escape/passwd")
        assert exc.value.status_code == 403


# ── resolve_path end-to-end ───────────────────────────────────────────────────


@pytest.mark.unit
class TestResolvePath:
    def _set_root(self, monkeypatch, root: Path):
        monkeypatch.setenv("VIDEO_DATA_ROOT", str(root))

    def test_benign_file_uri_resolves(self, tmp_path, monkeypatch):
        self._set_root(monkeypatch, tmp_path)
        (tmp_path / "session").mkdir()
        target = tmp_path / "session" / "clip.mp4"
        target.touch()
        out = resolve_path("file://session/clip.mp4")
        assert out == target.resolve()

    def test_non_file_scheme_raises_422(self, tmp_path, monkeypatch):
        self._set_root(monkeypatch, tmp_path)
        for uri in ("minio://bucket/key", "stream://7/0-5", "http://x/y", "bucket/key"):
            with pytest.raises(HTTPException) as exc:
                resolve_path(uri)
            assert exc.value.status_code == 422, uri

    def test_absolute_payload_coerced_to_in_root(self, tmp_path, monkeypatch):
        """``file:///etc/passwd`` is neutralised by lstrip — the leading slash
        is stripped so the payload becomes ``etc/passwd`` *relative* to the
        configured root, never the host ``/etc/passwd``.  The resolved path
        therefore stays inside the root (it may not exist, but containment is
        preserved)."""
        self._set_root(monkeypatch, tmp_path)
        out = resolve_path("file:///etc/passwd")
        assert out.is_relative_to(tmp_path.resolve())

    def test_relative_escape_raises_403(self, tmp_path, monkeypatch):
        nested = tmp_path / "data" / "videos"
        nested.mkdir(parents=True)
        self._set_root(monkeypatch, nested)
        with pytest.raises(HTTPException) as exc:
            resolve_path("file://../../etc/passwd")
        assert exc.value.status_code == 403

    def test_deeper_relative_escape_raises_403(self, tmp_path, monkeypatch):
        nested = tmp_path / "deeply" / "nested" / "root"
        nested.mkdir(parents=True)
        self._set_root(monkeypatch, nested)
        with pytest.raises(HTTPException) as exc:
            resolve_path("file://../../../../../etc/passwd")
        assert exc.value.status_code == 403

    def test_symlink_escape_raises_403(self, tmp_path, monkeypatch):
        self._set_root(monkeypatch, tmp_path)
        try:
            (tmp_path / "escape").symlink_to("/etc")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this filesystem")
        with pytest.raises(HTTPException) as exc:
            resolve_path("file://escape/passwd")
        assert exc.value.status_code == 403


# ── get_precise_times — silent fall-through on escape ─────────────────────────


@pytest.mark.unit
class TestGetPreciseTimesContainment:
    def test_non_file_scheme_returns_raw(self, tmp_path):
        out = get_precise_times("stream://7/0-5", 1.0, 2.0, str(tmp_path))
        assert out == (1.0, 2.0)

    def test_missing_sidecar_returns_raw(self, tmp_path):
        out = get_precise_times("file://session/clip.mp4", 1.0, 2.0, str(tmp_path))
        assert out == (1.0, 2.0)

    def test_relative_escape_returns_raw_silently(self, tmp_path):
        """Containment escape must fall through to raw values without
        raising, AND must never open the attacker-controlled sidecar even
        if it happens to exist on disk."""
        nested = tmp_path / "data"
        nested.mkdir()
        # Plant a fake .timestamp file at tmp_path/etc/passwd.timestamp.
        # If containment is broken, the parser would read it and the function
        # would return (0.0, 0.0) from _frame_aligned_times (frame_idx=0).
        attacker_dir = tmp_path / "etc"
        attacker_dir.mkdir()
        (attacker_dir / "passwd.timestamp").write_text("FPS,30\nSize,0,0\n0,1000\n")
        out = get_precise_times("file://../etc/passwd", 1.0, 2.0, str(nested))
        # Silent fall-through: raw start_s / end_s returned unchanged.
        assert out == (1.0, 2.0)

    def test_absolute_payload_falls_through_in_root(self, tmp_path):
        """``file:///etc/passwd`` is neutralised by lstrip — the resolved
        sidecar path stays inside the root.  Because no such sidecar exists
        there, the function returns the raw values."""
        out = get_precise_times("file:///etc/passwd", 1.0, 2.0, str(tmp_path))
        assert out == (1.0, 2.0)

    def test_benign_in_root_uses_sidecar(self, tmp_path):
        """When the sidecar exists inside the root, precise alignment fires."""
        (tmp_path / "session").mkdir()
        (tmp_path / "session" / "clip.mp4").touch()
        # 30 fps, 150 frames → 5s clip; start_s=0.0 end_s=5.0 → frame-aligned
        # output should match exactly (start_frame=0, end_frame=149).
        ts_path = tmp_path / "session" / "clip.timestamp"
        ts_path.write_text(
            "FPS,30\nSize,1920,1080\n" + "\n".join(f"{i},{i * 33333}" for i in range(150))
        )
        out = get_precise_times("file://session/clip.mp4", 0.0, 5.0, str(tmp_path))
        # start = 0/30 = 0.0; end = min(150, 149)/30 = 149/30 ≈ 4.9667
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        assert out[1] == pytest.approx(149 / 30, abs=1e-6)


# ── serve_segment anti-download response headers (#44) ────────────────────────


@pytest.mark.unit
class TestServeSegmentAntiDownloadHeaders:
    """Issue #44: serve_segment must set Content-Disposition / Cache-Control /
    X-Content-Type-Options on every successful response so the browser does
    not offer a save-as prompt and does not persist the bytes to disk cache.
    The byte-range header (Accept-Ranges: bytes) must be preserved — the
    video player's scrub/seek behaviour depends on it.

    Runs ``serve_segment`` via ``asyncio.run`` (not the pytest-asyncio
    decorator) so the FileResponse — which holds an OS file handle — is
    constructed and dropped inside a single-purpose event loop that
    exits cleanly within the test scope.  Letting an asyncio fixture's
    loop outlive the response object leaks state into ``sys.modules``
    under the default suite collection order.
    """

    @staticmethod
    def _build_response(tmp_path, monkeypatch):
        import asyncio

        monkeypatch.setenv("VIDEO_DATA_ROOT", str(tmp_path))
        (tmp_path / "session").mkdir()
        target = tmp_path / "session" / "clip.mp4"
        target.write_bytes(b"\x00" * 16)
        return asyncio.run(serve_segment("file://session/clip.mp4"))

    def test_serve_segment_emits_all_anti_download_headers(self, tmp_path, monkeypatch):
        response = self._build_response(tmp_path, monkeypatch)
        # Starlette MutableHeaders lower-cases keys; access is case-insensitive.
        assert response.headers["content-disposition"] == 'inline; filename=""'
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["accept-ranges"] == "bytes"

    def test_serve_segment_preserves_media_type(self, tmp_path, monkeypatch):
        response = self._build_response(tmp_path, monkeypatch)
        assert response.media_type == "video/mp4"
