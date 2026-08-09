"""GT-7: golden review_status UPSERT equivalence.

Exercises ``PATCH /v1/clips/{id}/review`` through the approve → reject → approve
cycle and verifies that the ON CONFLICT(clip_id) UPSERT path keeps a single row
per clip with the latest state.

Requires the compose stack (postgres + curation-api).  Credentials are read
from the project ``.env`` so the test follows whatever the running compose
stack is using; if either the API is unreachable or postgres auth fails the
test is skipped (not failed) — that signals "not deployed", not "broken".

References:
  docs/refactoring_plan.md  §3.1 GT-7.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


_API_BASE = os.environ.get("CURATION_API_BASE", "http://localhost:8001")


def _load_dotenv_into_env() -> None:
    """Lightweight .env loader for tests — populates ``os.environ`` for missing keys."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv_into_env()


def _pg_dsn() -> str:
    if os.environ.get("CURATOR_DB_URL"):
        return os.environ["CURATOR_DB_URL"]
    user = os.environ.get("PG_USER", "curation")
    password = os.environ.get("PG_PASSWORD", "curation-password")
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5432")
    db = os.environ.get("PG_DB", "curation")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _api_reachable() -> bool:
    if httpx is None:
        return False
    try:
        return httpx.get(f"{_API_BASE}/health", timeout=1.5).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(asyncpg is None or httpx is None, reason="asyncpg/httpx not installed"),
    pytest.mark.skipif(not _api_reachable(), reason="curation-api not reachable on compose"),
]


async def _try_connect():
    """Return an open asyncpg connection or None on credential/connect failure."""
    try:
        return await asyncpg.connect(_pg_dsn())
    except (asyncpg.InvalidPasswordError, asyncpg.InvalidCatalogNameError, OSError):
        return None


def _pick_existing_clip_id() -> str | None:
    if httpx is None:
        return None
    try:
        r = httpx.post(
            f"{_API_BASE}/v1/search",
            json={"query": "scene", "limit": 1, "top_k": 50},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        items = r.json().get("results") or []
        return items[0]["clip_id"] if items else None
    except Exception:
        return None


async def _restore_review_row(conn, clip_id, before) -> None:
    """Put a clip's review row back to its pre-test state (or remove it)."""
    if before is None:
        await conn.execute("DELETE FROM review_queue WHERE clip_id = $1", clip_id)
        return
    await conn.execute(
        """
        UPDATE review_queue
           SET state = $2, reason = $3, reviewer = $4, reviewed_at = $5
         WHERE clip_id = $1
        """,
        clip_id,
        before["state"],
        before["reason"],
        before["reviewer"],
        before["reviewed_at"],
    )


@pytest.mark.asyncio
async def test_golden_review_upsert_lifecycle():
    """approve → reject → approve must keep exactly one row per clip_id and end as approved."""
    clip_id = _pick_existing_clip_id()
    if clip_id is None:
        pytest.skip("no clips in corpus to drive review lifecycle")

    conn = await _try_connect()
    if conn is None:
        pytest.skip("postgres unreachable / auth failed for DSN derived from .env")

    cid = uuid.UUID(clip_id)
    # The clip is a live corpus row, so snapshot its review state and put it
    # back in the finally block — the test must not leave it approved.
    before = await conn.fetchrow(
        "SELECT state, reason, reviewer, reviewed_at FROM review_queue WHERE clip_id = $1", cid
    )

    try:
        async with httpx.AsyncClient(base_url=_API_BASE, timeout=5.0) as client:
            for action in ("approve", "reject", "approve"):
                r = await client.patch(f"/v1/clips/{clip_id}/review", json={"action": action})
                assert r.status_code in (200, 204), f"{action}: {r.status_code} {r.text}"

        # UPSERT: exactly one row per clip_id, latest state = approved.
        row_count = await conn.fetchval("SELECT count(*) FROM review_queue WHERE clip_id = $1", cid)
        state = await conn.fetchval("SELECT state FROM review_queue WHERE clip_id = $1", cid)
        assert row_count == 1, f"UPSERT regression: expected 1 row, got {row_count}"
        assert state == "approved", f"final state drift: {state!r}"
    finally:
        try:
            await _restore_review_row(conn, cid, before)
        finally:
            await conn.close()
