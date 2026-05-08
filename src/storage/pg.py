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
        frames_blob_uri: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO clips
                (clip_id, session_id, blob_uri, start_s, end_s,
                 frame_count, is_gold, is_synthetic, frames_blob_uri)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
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
            frames_blob_uri,
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
        curation_meta: dict[str, Any] | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO scenario_dna
                (clip_id, dna_version, dna_json, scout_prompt_hash, judge_prompt_hash,
                 pipeline_version, curation_meta)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (clip_id) DO UPDATE SET
                dna_version       = EXCLUDED.dna_version,
                dna_json          = EXCLUDED.dna_json,
                scout_prompt_hash = EXCLUDED.scout_prompt_hash,
                judge_prompt_hash = EXCLUDED.judge_prompt_hash,
                pipeline_version  = EXCLUDED.pipeline_version,
                curation_meta     = EXCLUDED.curation_meta
            """,
            clip_id,
            dna_version,
            dna_json,
            scout_prompt_hash,
            judge_prompt_hash,
            pipeline_version,
            curation_meta or {},
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
        curation_meta: dict[str, Any] | None = None,
        frames_blob_uri: str | None = None,
    ) -> None:
        """Insert clip + scenario_dna atomically in one transaction."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO clips
                    (clip_id, session_id, blob_uri, start_s, end_s,
                     frame_count, is_gold, is_synthetic, frames_blob_uri)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
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
                frames_blob_uri,
            )
            await conn.execute(
                """
                INSERT INTO scenario_dna
                    (clip_id, dna_version, dna_json, scout_prompt_hash,
                     judge_prompt_hash, pipeline_version, curation_meta)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (clip_id) DO UPDATE SET
                    dna_version       = EXCLUDED.dna_version,
                    dna_json          = EXCLUDED.dna_json,
                    scout_prompt_hash = EXCLUDED.scout_prompt_hash,
                    judge_prompt_hash = EXCLUDED.judge_prompt_hash,
                    pipeline_version  = EXCLUDED.pipeline_version,
                    curation_meta     = EXCLUDED.curation_meta
                """,
                clip_id,
                dna_version,
                dna_json,
                scout_prompt_hash,
                judge_prompt_hash,
                pipeline_version,
                curation_meta or {},
            )

    async def insert_review_queue(
        self,
        *,
        clip_id: UUID,
        state: str,
        reason: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO review_queue (clip_id, state, reason, reviewer)
            VALUES ($1, $2, $3, $4)
            """,
            clip_id,
            state,
            reason,
            reviewer,
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

    async def filter_dna_by_ids(
        self,
        clip_ids: list[UUID],
        filters: dict[str, str | list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Filter scenario_dna rows by clip_id list + optional DNA field conditions.

        Always uses clip_id = ANY($1) — never issues a full table scan.
        Scalar DNA fields use GIN jsonpath; array fields use JSONB operators.

        Args:
            clip_ids: Candidate set from Milvus ANN search.
            filters: Mapping of DNA field name → value or list of values.
                     Recognised fields: weather, lighting, sensor_fidelity,
                     road_type, lane_event, intersection_type, actor_class,
                     actor_state, risk_level, ego_maneuver.
            limit: Maximum number of rows to return (safety cap only;
                   caller re-ranks by Milvus score before applying user limit).

        Returns:
            List of {"clip_id": UUID, "dna_json": dict}.
        """
        _SCALAR_PATH: dict[str, tuple[str, str]] = {
            "weather": ("odd", "weather"),
            "lighting": ("odd", "lighting"),
            "road_type": ("topology", "road_type"),
            "lane_event": ("topology", "lane_event"),
            "intersection_type": ("topology", "intersection_type"),
            "risk_level": ("planner_logic", "risk_level"),
            "ego_maneuver": ("planner_logic", "ego_maneuver"),
        }

        conditions: list[str] = ["clip_id = ANY($1)"]
        params: list[Any] = [clip_ids]
        idx = 2

        for field, val in filters.items():
            values: list[str] = [val] if isinstance(val, str) else list(val)
            if field in _SCALAR_PATH:
                parent, key = _SCALAR_PATH[field]
                conditions.append(f"dna_json -> '{parent}' ->> '{key}' = ANY(${idx})")
                params.append(values)
                idx += 1
            elif field == "sensor_fidelity":
                conditions.append(f"dna_json -> 'odd' -> 'sensor_fidelity' ?| ${idx}")
                params.append(values)
                idx += 1
            elif field == "actor_class":
                conditions.append(
                    f"EXISTS ("
                    f"SELECT 1 FROM jsonb_array_elements(dna_json -> 'actor_dynamics') AS elem"
                    f" WHERE elem ->> 'actor_class' = ANY(${idx}))"
                )
                params.append(values)
                idx += 1
            elif field == "actor_state":
                conditions.append(
                    f"EXISTS ("
                    f"SELECT 1 FROM jsonb_array_elements(dna_json -> 'actor_dynamics') AS elem"
                    f" WHERE elem ->> 'state' = ANY(${idx}))"
                )
                params.append(values)
                idx += 1

        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"SELECT clip_id, dna_json FROM scenario_dna WHERE {where} LIMIT ${idx}"

        rows = await self._pool.fetch(sql, *params)
        return [{"clip_id": r["clip_id"], "dna_json": dict(r["dna_json"])} for r in rows]

    async def get_clip_with_blob_uri(self, clip_id: UUID) -> dict[str, Any] | None:
        """Return clip metadata joined with its scenario_dna row.

        Returns:
            Dict with keys: clip_id, session_id, blob_uri, frames_blob_uri,
            start_s, end_s, dna_version, dna_json.  None if the clip does not exist.
        """
        row = await self._pool.fetchrow(
            """
            SELECT c.clip_id, c.session_id, c.blob_uri, c.frames_blob_uri,
                   c.start_s, c.end_s, sd.dna_version, sd.dna_json
            FROM clips c
            LEFT JOIN scenario_dna sd ON c.clip_id = sd.clip_id
            WHERE c.clip_id = $1
            """,
            clip_id,
        )
        if row is None:
            return None
        return {
            "clip_id": row["clip_id"],
            "session_id": row["session_id"],
            "blob_uri": row["blob_uri"],
            "frames_blob_uri": row["frames_blob_uri"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
            "dna_version": row["dna_version"],
            "dna_json": dict(row["dna_json"]) if row["dna_json"] else None,
        }
