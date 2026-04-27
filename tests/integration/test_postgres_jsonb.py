"""Integration tests for PGRepository (P1-3).

Spins up a fresh postgres:17-alpine container via testcontainers, applies
infra/init-sql/001_schema.sql through asyncpg, then exercises the DAL.
Auto-skipped when Docker is unavailable.

Event-loop note: the container fixture is module-scoped (sync, no loop
involvement) while all async fixtures are function-scoped so they share
the same loop as each test function — avoiding asyncpg "Future attached
to a different loop" errors.
"""

from __future__ import annotations

import pathlib
import shutil
import uuid
from datetime import datetime, timezone

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from src.storage.pg import PGRepository

REPO_ROOT = pathlib.Path(__file__).parents[2]
INIT_SQL = (REPO_ROOT / "infra" / "init-sql" / "001_schema.sql").read_text()

DOCKER_AVAILABLE = bool(shutil.which("docker"))

pytestmark = pytest.mark.integration

# ── fixed IDs so tests are order-independent ──────────────────────────────

SESSION_ID = "sess_p13_test"
CLIP_A = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
CLIP_B = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")

_BASE_DNA: dict = {
    "dna_version": "0.1.0",
    "clip_id": str(CLIP_A),
    "odd": {"weather": "clear", "lighting": "day", "sensor_fidelity": []},
    "topology": {
        "road_type": "primary",
        "lane_event": "normal",
        "intersection_type": "none",
    },
    "actor_dynamics": [],
    "planner_logic": {
        "ego_maneuver": "cruise",
        "risk_level": "nominal",
        "causal_trigger_actor_index": None,
    },
    "confidence": {"overall": 0.9, "scout_agreement": 1.0, "hallucination_flags": []},
    "provenance": {
        "scout_models": ["Cosmos-Reason2-8B-FP8"],
        "scout_prompt_hash": "deadbeef",
        "judge_model": None,
        "judge_prompt_hash": None,
        "pipeline_version": "0.1.0",
        "is_synthetic": False,
        "reference_standards": [],
    },
}

# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pg_container():
    """Start one Postgres container for the entire module (sync, no loop)."""
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker not available — skipping Postgres integration tests")
    with PostgresContainer("postgres:17.9-alpine3.23", driver=None) as pg:
        yield pg


@pytest.fixture(scope="module")
def pg_dsn(pg_container: PostgresContainer) -> str:
    """Extract DSN once; reused by every function-scoped repo fixture."""
    return pg_container.get_connection_url(driver=None)


@pytest_asyncio.fixture
async def repo(pg_dsn: str):
    """Fresh pool per test — same event loop as the test function."""
    conn = await asyncpg.connect(pg_dsn)
    await conn.execute(INIT_SQL)
    await conn.close()
    r = await PGRepository.create(pg_dsn, min_size=1, max_size=2)
    yield r
    await r.close()


@pytest_asyncio.fixture(autouse=True)
async def seed_session(repo: PGRepository):
    """Insert shared test session before each test (idempotent)."""
    await repo.insert_session(
        session_id=SESSION_ID,
        dataset="test_proj",
        subset="test_sub",
        dataset_version="v0",
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_kind="real",
    )


# ── tests ──────────────────────────────────────────────────────────────────


async def test_insert_clip_and_upsert_dna(repo: PGRepository):
    """Basic insert_clip + upsert_dna round-trip."""
    await repo.insert_clip(
        clip_id=CLIP_A,
        session_id=SESSION_ID,
        blob_uri="s3://clips/test/clip_a.mp4",
        start_s=0.0,
        end_s=10.0,
    )
    await repo.upsert_dna(
        clip_id=CLIP_A,
        dna_version="0.1.0",
        dna_json=_BASE_DNA,
        scout_prompt_hash="deadbeef",
        pipeline_version="0.1.0",
    )
    result = await repo.get_dna(CLIP_A)
    assert result is not None
    assert result["dna_version"] == "0.1.0"
    assert result["odd"]["weather"] == "clear"
    assert result["planner_logic"]["risk_level"] == "nominal"


async def test_upsert_dna_overwrites_existing(repo: PGRepository):
    """Second upsert on the same clip_id updates the row (no duplicate)."""
    await repo.insert_clip(
        clip_id=CLIP_A,
        session_id=SESSION_ID,
        blob_uri="s3://clips/test/clip_a.mp4",
        start_s=0.0,
        end_s=10.0,
    )
    await repo.upsert_dna(
        clip_id=CLIP_A,
        dna_version="0.1.0",
        dna_json=_BASE_DNA,
        scout_prompt_hash="deadbeef",
        pipeline_version="0.1.0",
    )
    updated = {
        **_BASE_DNA,
        "odd": {**_BASE_DNA["odd"], "weather": "heavy_rain", "lighting": "night"},
    }
    await repo.upsert_dna(
        clip_id=CLIP_A,
        dna_version="0.1.0",
        dna_json=updated,
        scout_prompt_hash="deadbeef",
        pipeline_version="0.1.0",
    )
    result = await repo.get_dna(CLIP_A)
    assert result["odd"]["weather"] == "heavy_rain"
    assert result["odd"]["lighting"] == "night"


async def test_write_clip_with_dna_atomic(repo: PGRepository):
    """write_clip_with_dna inserts clip + DNA in a single transaction."""
    dna_b = {**_BASE_DNA, "clip_id": str(CLIP_B)}
    await repo.write_clip_with_dna(
        session_id=SESSION_ID,
        clip_id=CLIP_B,
        blob_uri="s3://clips/test/clip_b.mp4",
        start_s=10.0,
        end_s=20.0,
        dna_version="0.1.0",
        dna_json=dna_b,
        scout_prompt_hash="cafebabe",
        pipeline_version="0.1.0",
    )
    result = await repo.get_dna(CLIP_B)
    assert result is not None
    assert result["planner_logic"]["ego_maneuver"] == "cruise"


async def test_query_dna_by_json_gin_index(repo: PGRepository):
    """query_dna_by_json uses the GIN index with a jsonpath predicate."""
    dna_b = {**_BASE_DNA, "clip_id": str(CLIP_B)}
    await repo.write_clip_with_dna(
        session_id=SESSION_ID,
        clip_id=CLIP_B,
        blob_uri="s3://clips/test/clip_b.mp4",
        start_s=10.0,
        end_s=20.0,
        dna_version="0.1.0",
        dna_json=dna_b,
        scout_prompt_hash="cafebabe",
        pipeline_version="0.1.0",
    )
    results = await repo.query_dna_by_json('$.planner_logic.risk_level == "nominal"')
    assert len(results) >= 1
    clip_ids = {r["clip_id"] for r in results}
    assert CLIP_B in clip_ids


async def test_query_dna_no_match(repo: PGRepository):
    """query_dna_by_json returns [] when no rows match."""
    results = await repo.query_dna_by_json('$.planner_logic.risk_level == "critical"')
    assert results == []


async def test_get_dna_missing_returns_none(repo: PGRepository):
    """get_dna returns None for an unknown clip_id."""
    missing = uuid.UUID("99999999-9999-9999-9999-999999999999")
    assert await repo.get_dna(missing) is None


async def test_insert_clip_idempotent(repo: PGRepository):
    """Inserting the same clip twice does not raise (ON CONFLICT DO NOTHING)."""
    for _ in range(2):
        await repo.insert_clip(
            clip_id=CLIP_A,
            session_id=SESSION_ID,
            blob_uri="s3://clips/test/clip_a.mp4",
            start_s=0.0,
            end_s=10.0,
        )


async def test_dna_jsonb_roundtrip_preserves_types(repo: PGRepository):
    """JSONB codec round-trips Python types correctly."""
    await repo.insert_clip(
        clip_id=CLIP_A,
        session_id=SESSION_ID,
        blob_uri="s3://clips/test/clip_a.mp4",
        start_s=0.0,
        end_s=10.0,
    )
    await repo.upsert_dna(
        clip_id=CLIP_A,
        dna_version="0.1.0",
        dna_json=_BASE_DNA,
        scout_prompt_hash="deadbeef",
        pipeline_version="0.1.0",
    )
    result = await repo.get_dna(CLIP_A)
    assert isinstance(result, dict)
    assert isinstance(result["confidence"]["overall"], float)
    assert isinstance(result["actor_dynamics"], list)
    assert result["planner_logic"]["causal_trigger_actor_index"] is None
