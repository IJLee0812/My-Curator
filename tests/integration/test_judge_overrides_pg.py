"""Integration tests for the P4-6 judge_overrides table + DAL (testcontainers postgres,
applying 001_schema.sql; auto-skips without Docker)."""

from __future__ import annotations

import pathlib
import shutil
import uuid
from datetime import datetime, timezone

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from my_curator.adapters.storage.pg import PGRepository

REPO_ROOT = pathlib.Path(__file__).parents[2]
INIT_SQL = (REPO_ROOT / "infra" / "init-sql" / "001_schema.sql").read_text()
DOCKER_AVAILABLE = bool(shutil.which("docker"))

pytestmark = pytest.mark.integration

SESSION_ID = "sess_p46_test"
CLIP_V02 = uuid.UUID("cccccccc-0000-0000-0000-000000000001")
CLIP_V01 = uuid.UUID("dddddddd-0000-0000-0000-000000000002")


def _dna(version: str, clip_id: uuid.UUID, risk: str = "elevated") -> dict:
    return {
        "dna_version": version,
        "clip_id": str(clip_id),
        "scene_description": "A short scene.",
        "planner_logic": {
            "risk_level": risk,
            "risk_level_rationale": "because reasons",
            "safety_event": {
                "has_event": False,
                "event_type": "none",
                "collision_type": None,
                "severity_estimate": None,
            },
        },
        "provenance": {"judge_model": None, "judge_prompt_hash": None},
    }


@pytest.fixture(scope="module")
def pg_container():
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker not available — skipping Postgres integration tests")
    with PostgresContainer("postgres:17.9-alpine3.23", driver=None) as pg:
        yield pg


@pytest.fixture(scope="module")
def pg_dsn(pg_container: PostgresContainer) -> str:
    return pg_container.get_connection_url(driver=None)


@pytest_asyncio.fixture
async def repo(pg_dsn: str):
    conn = await asyncpg.connect(pg_dsn)
    await conn.execute(INIT_SQL)
    await conn.execute(
        "TRUNCATE judge_overrides, review_queue, scenario_dna, clips, sessions CASCADE"
    )
    await conn.close()
    r = await PGRepository.create(pg_dsn, min_size=1, max_size=2)
    yield r
    await r.close()


@pytest_asyncio.fixture(autouse=True)
async def seed(repo: PGRepository):
    await repo.insert_session(
        session_id=SESSION_ID,
        dataset="p",
        subset="s",
        dataset_version="v0",
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_kind="real",
    )
    for clip, ver in ((CLIP_V02, "0.2.0"), (CLIP_V01, "0.1.0")):
        await repo.insert_clip(
            clip_id=clip,
            session_id=SESSION_ID,
            blob_uri=f"s3://c/{clip}.mp4",
            start_s=0.0,
            end_s=5.0,
        )
        await repo.upsert_dna(
            clip_id=clip,
            dna_version=ver,
            dna_json=_dna(ver, clip),
            scout_prompt_hash="scout-hash",
            pipeline_version="0.2.0",
        )


async def test_insert_and_get_override(repo: PGRepository):
    oid = await repo.insert_judge_override(
        clip_id=CLIP_V02,
        field="risk_level",
        scout_value="elevated",
        judge_value="critical",
        gt_value="critical",
    )
    assert isinstance(oid, int)
    rows = await repo.get_judge_overrides(CLIP_V02)
    assert len(rows) == 1
    assert rows[0]["field"] == "risk_level"
    assert rows[0]["judge_value"] == "critical"


async def test_append_only_history_and_latest_only(repo: PGRepository):
    await repo.insert_judge_override(
        clip_id=CLIP_V02, field="risk_level", scout_value="nominal", judge_value="elevated"
    )
    await repo.insert_judge_override(
        clip_id=CLIP_V02, field="risk_level", scout_value="nominal", judge_value="critical"
    )
    history = await repo.get_judge_overrides(CLIP_V02)
    assert len(history) == 2  # append-only
    latest = await repo.get_judge_overrides(CLIP_V02, latest_only=True)
    assert len(latest) == 1
    assert latest[0]["judge_value"] == "critical"  # most recent wins


async def test_apply_override_dna_preserves_scout_provenance(repo: PGRepository):
    updated = _dna("0.2.0", CLIP_V02, risk="critical")
    updated["provenance"]["judge_model"] = "qwen3-8b-awq"
    await repo.apply_judge_override_dna(
        clip_id=CLIP_V02, dna_json=updated, judge_prompt_hash="judge-hash-abc"
    )
    row = await repo._pool.fetchrow(  # noqa: SLF001 — test asserts internal columns
        "SELECT dna_version, dna_json, scout_prompt_hash, judge_prompt_hash, pipeline_version "
        "FROM scenario_dna WHERE clip_id = $1",
        CLIP_V02,
    )
    assert row["dna_version"] == "0.2.0"  # untouched
    assert row["scout_prompt_hash"] == "scout-hash"  # untouched
    assert row["pipeline_version"] == "0.2.0"  # untouched
    assert row["judge_prompt_hash"] == "judge-hash-abc"  # updated
    assert dict(row["dna_json"])["planner_logic"]["risk_level"] == "critical"  # updated


async def test_list_v02_dna_excludes_v01(repo: PGRepository):
    rows = await repo.list_v02_dna()
    ids = {r["clip_id"] for r in rows}
    assert CLIP_V02 in ids
    assert CLIP_V01 not in ids  # v0.1 excluded


async def test_list_v02_dna_by_session_and_clip_ids(repo: PGRepository):
    by_session = await repo.list_v02_dna(session_id=SESSION_ID)
    assert [r["clip_id"] for r in by_session] == [CLIP_V02]
    by_ids = await repo.list_v02_dna(clip_ids=[CLIP_V02, CLIP_V01])
    assert [r["clip_id"] for r in by_ids] == [CLIP_V02]  # v0.1 still filtered out
    none_match = await repo.list_v02_dna(clip_ids=[CLIP_V01])
    assert none_match == []


async def test_override_cascades_on_clip_delete(repo: PGRepository):
    await repo.insert_judge_override(
        clip_id=CLIP_V02, field="scene_description", scout_value="old", judge_value="new"
    )
    await repo._pool.execute("DELETE FROM clips WHERE clip_id = $1", CLIP_V02)  # noqa: SLF001
    assert await repo.get_judge_overrides(CLIP_V02) == []  # FK ON DELETE CASCADE
