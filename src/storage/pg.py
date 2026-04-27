"""PostgreSQL DAL for My-Curator (asyncpg-based).

Public surface:
  dsn_from_env()          — build a DSN from environment variables
  PGRepository.create()   — async factory (creates pool, registers JSONB codec)
  PGRepository.close()    — drain pool
  insert_session / insert_clip / upsert_dna — individual writes
  write_clip_with_dna     — clip + DNA in a single atomic transaction
  get_dna / query_dna_by_json — reads
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg


def dsn_from_env() -> str:
    """Build DSN from PG_USER, PG_PASSWORD, PG_HOST (default localhost),
    PG_PORT (default 5432), PG_DB (default curation)."""
    user = os.environ["PG_USER"]
    password = os.environ["PG_PASSWORD"]
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5432")
    db = os.environ.get("PG_DB", "curation")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def _setup_connection(conn: asyncpg.Connection) -> None:
    """Register JSONB <-> dict codec so callers never touch json.dumps/loads."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


class PGRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(
        cls,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> PGRepository:
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            init=_setup_connection,
        )
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    # ── individual writes ──────────────────────────────────────────────────

    async def insert_session(
        self,
        *,
        session_id: str,
        dataset: str,
        subset: str,
        dataset_version: str,
        recorded_at: datetime,
        source_kind: str,
        notes: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO sessions
                (session_id, dataset, subset, dataset_version, recorded_at, source_kind, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id,
            dataset,
            subset,
            dataset_version,
            recorded_at,
            source_kind,
            notes,
        )

    async def insert_clip(
        self,
        *,
        clip_id: UUID,
        session_id: str,
        blob_uri: str,
        start_s: float,
        end_s: float,
        frame_count: int | None = None,
        is_gold: bool = False,
        is_synthetic: bool = False,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO clips
                (clip_id, session_id, blob_uri, start_s, end_s, frame_count, is_gold, is_synthetic)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (clip_id) DO NOTHING
            """,
            clip_id,
            session_id,
            blob_uri,
            start_s,
            end_s,
            frame_count,
            is_gold,
            is_synthetic,
        )

    async def upsert_dna(
        self,
        *,
        clip_id: UUID,
        dna_version: str,
        dna_json: dict[str, Any],
        scout_prompt_hash: str,
        pipeline_version: str,
        judge_prompt_hash: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO scenario_dna
                (clip_id, dna_version, dna_json, scout_prompt_hash, judge_prompt_hash, pipeline_version)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (clip_id) DO UPDATE SET
                dna_version       = EXCLUDED.dna_version,
                dna_json          = EXCLUDED.dna_json,
                scout_prompt_hash = EXCLUDED.scout_prompt_hash,
                judge_prompt_hash = EXCLUDED.judge_prompt_hash,
                pipeline_version  = EXCLUDED.pipeline_version
            """,
            clip_id,
            dna_version,
            dna_json,
            scout_prompt_hash,
            judge_prompt_hash,
            pipeline_version,
        )

    # ── atomic composite write ─────────────────────────────────────────────

    async def write_clip_with_dna(
        self,
        *,
        session_id: str,
        clip_id: UUID,
        blob_uri: str,
        start_s: float,
        end_s: float,
        dna_version: str,
        dna_json: dict[str, Any],
        scout_prompt_hash: str,
        pipeline_version: str,
        frame_count: int | None = None,
        is_gold: bool = False,
        is_synthetic: bool = False,
        judge_prompt_hash: str | None = None,
    ) -> None:
        """Insert clip + scenario_dna atomically in one transaction."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO clips
                    (clip_id, session_id, blob_uri, start_s, end_s,
                     frame_count, is_gold, is_synthetic)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (clip_id) DO NOTHING
                """,
                clip_id,
                session_id,
                blob_uri,
                start_s,
                end_s,
                frame_count,
                is_gold,
                is_synthetic,
            )
            await conn.execute(
                """
                INSERT INTO scenario_dna
                    (clip_id, dna_version, dna_json, scout_prompt_hash,
                     judge_prompt_hash, pipeline_version)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (clip_id) DO UPDATE SET
                    dna_version       = EXCLUDED.dna_version,
                    dna_json          = EXCLUDED.dna_json,
                    scout_prompt_hash = EXCLUDED.scout_prompt_hash,
                    judge_prompt_hash = EXCLUDED.judge_prompt_hash,
                    pipeline_version  = EXCLUDED.pipeline_version
                """,
                clip_id,
                dna_version,
                dna_json,
                scout_prompt_hash,
                judge_prompt_hash,
                pipeline_version,
            )

    # ── reads ──────────────────────────────────────────────────────────────

    async def get_dna(self, clip_id: UUID) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            "SELECT dna_json FROM scenario_dna WHERE clip_id = $1",
            clip_id,
        )
        return dict(row["dna_json"]) if row else None

    async def query_dna_by_json(self, jsonpath: str) -> list[dict[str, Any]]:
        """Return rows matching a jsonpath predicate (uses GIN index).

        Example: '$.planner_logic.risk_level == "critical"'
        """
        rows = await self._pool.fetch(
            "SELECT clip_id, dna_json FROM scenario_dna WHERE dna_json @@ $1::jsonpath",
            jsonpath,
        )
        return [{"clip_id": r["clip_id"], "dna_json": dict(r["dna_json"])} for r in rows]
