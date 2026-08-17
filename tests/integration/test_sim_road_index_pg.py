"""The road index must provision itself on a database that predates its DDL.

``infra/init-sql`` only runs on a fresh volume, so a table added there never appears on a
deployed database — that is how ``judge_overrides`` ended up missing in P4-6. These tests
start from a schema with the table deliberately removed and check the builder recovers.
"""

from __future__ import annotations

import pathlib
import shutil

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from my_curator.adapters.storage.pg import PGRepository

REPO_ROOT = pathlib.Path(__file__).parents[2]
INIT_SQL = (REPO_ROOT / "infra" / "init-sql" / "001_schema.sql").read_text()
DOCKER_AVAILABLE = bool(shutil.which("docker"))

pytestmark = pytest.mark.integration


def _schema_without_road_index() -> str:
    """The schema as it stood before P5-3, i.e. what a deployed volume actually holds."""
    start = INIT_SQL.index("-- sim_road_index (P5-3)")
    end = INIT_SQL.index("COMMIT;", start)
    return INIT_SQL[:start] + INIT_SQL[end:]


def _rows(count: int = 3) -> list[dict]:
    return [
        {
            "town": "Town03",
            "road_id": 10 + i,
            "lane_id": -1,
            "lane_section_s": 0.0,
            "lane_section_end_s": 200.0,
            "driving_lanes": 2,
            "speed_kph": 40.0,
            "lane_types": frozenset({"driving", "sidewalk"}),
            "junction_forms": frozenset({"signalized"}),
            "in_junction": False,
        }
        for i in range(count)
    ]


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker unavailable")
class TestSelfProvisioning:
    @pytest_asyncio.fixture
    async def repo(self):
        with PostgresContainer("postgres:16-alpine") as pg:
            dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
            conn = await asyncpg.connect(dsn)
            await conn.execute(_schema_without_road_index())
            await conn.close()
            repository = await PGRepository.create(dsn)
            try:
                yield repository
            finally:
                await repository.close()

    async def _table_exists(self, repo: PGRepository) -> bool:
        async with repo._pool.acquire() as conn:
            return await conn.fetchval("SELECT to_regclass('public.sim_road_index') IS NOT NULL")

    async def test_the_deployed_schema_really_lacks_the_table(self, repo):
        """Guards the premise: if this fails the other tests prove nothing."""
        assert not await self._table_exists(repo)

    async def test_the_builder_creates_the_table_it_needs(self, repo):
        await repo.ensure_sim_road_index()
        assert await self._table_exists(repo)

    async def test_provisioning_twice_is_harmless(self, repo):
        await repo.ensure_sim_road_index()
        await repo.ensure_sim_road_index()
        assert await self._table_exists(repo)

    async def test_writing_the_index_provisions_on_the_way(self, repo):
        """The operator runs the builder, not a migration; it must not need one."""
        written = await repo.replace_sim_road_index(_rows())
        assert written == 3
        assert len(await repo.list_sim_road_index()) == 3

    async def test_the_index_round_trips(self, repo):
        await repo.replace_sim_road_index(_rows(1))
        row = (await repo.list_sim_road_index())[0]
        assert row["town"] == "Town03"
        assert row["speed_kph"] == 40.0
        assert set(row["lane_types"]) == {"driving", "sidewalk"}
        assert set(row["junction_forms"]) == {"signalized"}
        assert row["in_junction"] is False

    async def test_rebuilding_replaces_rather_than_appends(self, repo):
        await repo.replace_sim_road_index(_rows(3))
        await repo.replace_sim_road_index(_rows(2))
        assert len(await repo.list_sim_road_index()) == 2

    async def test_towns_can_be_filtered(self, repo):
        await repo.replace_sim_road_index(_rows(2))
        assert await repo.list_sim_road_index(towns=["Town05"]) == []
        assert len(await repo.list_sim_road_index(towns=["Town03"])) == 2
