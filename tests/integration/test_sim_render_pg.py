"""The render ledger must provision itself, and record failures as faithfully as successes.

Same premise as the road index: ``infra/init-sql`` only runs on a fresh volume, so a table
added there never appears on a deployed database. A render that cannot write its ledger row
is a render whose failure goes unrecorded, which is the one outcome the ledger exists to
prevent.
"""

from __future__ import annotations

import pathlib
import shutil
import uuid

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from my_curator.adapters.storage.pg import SIM_RENDER_DDL, PGRepository
from my_curator.domain.sim.reasons import RenderFailure
from my_curator.domain.sim.render import FAILED, RENDERED

REPO_ROOT = pathlib.Path(__file__).parents[2]
INIT_SQL = (REPO_ROOT / "infra" / "init-sql" / "001_schema.sql").read_text()
DOCKER_AVAILABLE = bool(shutil.which("docker"))

pytestmark = pytest.mark.integration


def _schema_without_render_ledger() -> str:
    """The schema as it stood before P5-4, i.e. what a deployed volume actually holds."""
    start = INIT_SQL.index("-- sim_render (P5-4)")
    end = INIT_SQL.index("COMMIT;", start)
    return INIT_SQL[:start] + INIT_SQL[end:]


def _record(clip_id: uuid.UUID, **over) -> dict:
    base = dict(
        clip_id=clip_id,
        source_clip_id="source-a",
        segment_index=1,
        status=RENDERED,
        failure_reason=None,
        town="Town03",
        road_id=42,
        lane_id=-1,
        duration_s=1.733,
        ego_key="synthetic/x/ego.mp4",
        chase_key="synthetic/x/chase.mp4",
        compare_key="synthetic/x/compare.mp4",
    )
    base.update(over)
    return base


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker unavailable")
class TestRenderLedger:
    @pytest_asyncio.fixture
    async def repo(self):
        with PostgresContainer("postgres:16-alpine") as pg:
            dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
            conn = await asyncpg.connect(dsn)
            await conn.execute(_schema_without_render_ledger())
            await conn.close()
            repository = await PGRepository.create(dsn)
            try:
                yield repository
            finally:
                await repository.close()

    @pytest_asyncio.fixture
    async def clip_id(self, repo):
        """A render row references a real clip, so one has to exist first."""
        new_id = uuid.uuid4()
        async with repo._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions
                    (session_id, dataset, subset, dataset_version, recorded_at, source_kind)
                VALUES ('s1', 'd', 's', 'v', now(), 'real')
                ON CONFLICT DO NOTHING
                """
            )
            await conn.execute(
                """
                INSERT INTO clips (clip_id, session_id, blob_uri, start_s, end_s, source_clip_id)
                VALUES ($1, 's1', 'file://a.mp4', 0, 5, 'source-a')
                """,
                new_id,
            )
        return new_id

    async def _table_exists(self, repo: PGRepository) -> bool:
        async with repo._pool.acquire() as conn:
            return await conn.fetchval("SELECT to_regclass('public.sim_render') IS NOT NULL")

    async def test_the_deployed_schema_really_lacks_the_table(self, repo):
        """Guards the premise: if this fails the other tests prove nothing."""
        assert not await self._table_exists(repo)

    async def test_recording_provisions_the_table_on_the_way(self, repo, clip_id):
        assert await repo.record_sim_render(_record(clip_id)) > 0
        assert await self._table_exists(repo)

    async def test_provisioning_twice_is_harmless(self, repo):
        await repo.ensure_sim_render()
        await repo.ensure_sim_render()
        assert await self._table_exists(repo)

    async def test_a_successful_render_records_all_three_keys(self, repo, clip_id):
        await repo.record_sim_render(_record(clip_id))
        row = (await repo.list_sim_renders())[0]
        assert row["status"] == RENDERED
        assert row["compare_key"] == "synthetic/x/compare.mp4"
        assert row["failure_reason"] is None

    async def test_a_failure_records_its_reason_and_no_keys(self, repo, clip_id):
        await repo.record_sim_render(
            _record(
                clip_id,
                status=FAILED,
                failure_reason=RenderFailure.SPAWN_REJECTED.value,
                ego_key=None,
                chase_key=None,
                compare_key=None,
            )
        )
        row = (await repo.list_sim_renders())[0]
        assert row["status"] == FAILED
        assert row["failure_reason"] == "spawn_rejected"
        assert row["ego_key"] is None

    async def test_attempts_accumulate_rather_than_overwrite(self, repo, clip_id):
        await repo.record_sim_render(_record(clip_id, status=FAILED, failure_reason="x"))
        await repo.record_sim_render(_record(clip_id))
        rows = await repo.list_sim_renders()
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {RENDERED, FAILED}

    async def test_renders_can_be_looked_up_by_source_clip(self, repo, clip_id):
        await repo.record_sim_render(_record(clip_id))
        assert len(await repo.list_sim_renders(source_clip_id="source-a")) == 1
        assert await repo.list_sim_renders(source_clip_id="source-b") == []


class TestDdlParity:
    def test_the_ledger_and_init_sql_declare_the_same_table(self):
        def normalize(text: str) -> str:
            body = text[text.index("CREATE TABLE IF NOT EXISTS sim_render") :]
            return " ".join(body[: body.index("idx_sim_render_clip")].split())

        assert normalize(INIT_SQL) == normalize(SIM_RENDER_DDL)
