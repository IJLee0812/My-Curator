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

from my_curator.domain.scout.versioning import CURRENT_DNA_VERSION


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


#: sim_road_index (P5-3) — one row per driving lane per lane section of the built-in
#: CARLA towns, rebuilt from the shipped OpenDRIVE files. Kept out of the repository
#: because it is derived data, and provisioned by its builder because init-sql only runs
#: on a fresh volume. Mirrored verbatim in infra/init-sql/001_schema.sql for new volumes;
#: tests/unit/test_sim_road_index_ddl.py asserts the two stay identical.
SIM_ROAD_INDEX_DDL = """
CREATE TABLE IF NOT EXISTS sim_road_index (
    town               TEXT             NOT NULL,
    road_id            INTEGER          NOT NULL,
    lane_id            INTEGER          NOT NULL,
    lane_section_s     DOUBLE PRECISION NOT NULL,
    lane_section_end_s DOUBLE PRECISION NOT NULL,
    driving_lanes      SMALLINT         NOT NULL,
    speed_kph          DOUBLE PRECISION NOT NULL,
    lane_types         TEXT[]           NOT NULL,
    junction_forms     TEXT[]           NOT NULL,
    in_junction        BOOLEAN          NOT NULL,
    PRIMARY KEY (town, road_id, lane_section_s, lane_id)
);
CREATE INDEX IF NOT EXISTS idx_sim_road_town ON sim_road_index(town);
"""

SIM_RENDER_DDL = """
CREATE TABLE IF NOT EXISTS sim_render (
    render_id      BIGSERIAL        PRIMARY KEY,
    clip_id        UUID             NOT NULL REFERENCES clips(clip_id) ON DELETE CASCADE,
    source_clip_id TEXT             NOT NULL,
    segment_index  SMALLINT         NOT NULL,
    status         TEXT             NOT NULL,
    failure_reason TEXT,
    town           TEXT,
    road_id        INTEGER,
    lane_id        INTEGER,
    duration_s     DOUBLE PRECISION,
    ego_key        TEXT,
    chase_key      TEXT,
    compare_key    TEXT,
    rendered_at    TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sim_render_clip ON sim_render(clip_id);
CREATE INDEX IF NOT EXISTS idx_sim_render_source ON sim_render(source_clip_id);
"""


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
        is_synthetic: bool = False,
        frames_blob_uri: str | None = None,
        source_clip_id: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO clips
                (clip_id, session_id, blob_uri, start_s, end_s,
                 frame_count, is_synthetic, frames_blob_uri,
                 source_clip_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (clip_id) DO NOTHING
            """,
            clip_id,
            session_id,
            blob_uri,
            start_s,
            end_s,
            frame_count,
            is_synthetic,
            frames_blob_uri,
            source_clip_id,
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
        is_synthetic: bool = False,
        judge_prompt_hash: str | None = None,
        curation_meta: dict[str, Any] | None = None,
        frames_blob_uri: str | None = None,
        source_clip_id: str | None = None,
    ) -> None:
        """Insert clip + scenario_dna atomically in one transaction."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO clips
                    (clip_id, session_id, blob_uri, start_s, end_s,
                     frame_count, is_synthetic, frames_blob_uri,
                     source_clip_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (clip_id) DO NOTHING
                """,
                clip_id,
                session_id,
                blob_uri,
                start_s,
                end_s,
                frame_count,
                is_synthetic,
                frames_blob_uri,
                source_clip_id,
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
            ON CONFLICT (clip_id) DO NOTHING
            """,
            clip_id,
            state,
            reason,
            reviewer,
        )

    async def set_review_status(self, clip_id: UUID, state: str) -> None:
        """Upsert review state for a clip (overwrite-allowed, P3-5)."""
        await self._pool.execute(
            """
            INSERT INTO review_queue (clip_id, state, reviewed_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (clip_id) DO UPDATE SET
                state       = EXCLUDED.state,
                reviewed_at = NOW()
            """,
            clip_id,
            state,
        )

    @staticmethod
    def _review_state_filter(status: str | None) -> tuple[str, list[Any]]:
        """Map a UI status tab to a SQL predicate + bind params.

        status filter:
          None / "all"     → all states
          "pending"        → state = 'pending'
          "approved"       → state = 'approved'
          "rejected"       → state = 'rejected'
          "schema_invalid" → state = 'rejected_schema_invalid'
        """
        if status in (None, "all"):
            return "", []
        if status == "schema_invalid":
            return "WHERE state = 'rejected_schema_invalid'", []
        return "WHERE state = $1", [status]

    @classmethod
    def _review_filters(cls, status: str | None, risk: str | None) -> tuple[str, list[Any]]:
        """State predicate plus an optional planner_logic.risk_level predicate.

        Risk lives in scenario_dna, so a risk filter forces the JOIN even on the
        count query; rows without DNA (schema-invalid ones) drop out, which is
        the intended reading of "clips at this risk level".
        """
        where, params = cls._review_state_filter(status)
        if risk in (None, "all"):
            return where, params
        params.append(risk)
        clause = f"sd.dna_json->'planner_logic'->>'risk_level' = ${len(params)}"
        return (f"{where} AND {clause}" if where else f"WHERE {clause}"), params

    async def count_review_queue(self, status: str | None = None, risk: str | None = None) -> int:
        """Total review_queue rows matching the status tab (drives pagination)."""
        where, params = self._review_filters(status, risk)
        where = where.replace("state =", "rq.state =")
        row = await self._pool.fetchrow(
            f"""
            SELECT COUNT(*) AS n
            FROM review_queue rq
            LEFT JOIN scenario_dna sd ON sd.clip_id = rq.clip_id
            {where}
            """,
            *params,
        )
        return int(row["n"]) if row else 0

    async def get_review_queue(
        self,
        status: str | None = None,
        *,
        risk: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a page of review_queue rows joined with clip + DNA data.

        Ordered by ``created_at DESC``; ``limit``/``offset`` page within the
        status tab.  See :meth:`count_review_queue` for the tab total.
        """
        capped = max(1, min(int(limit), 200))
        off = max(0, int(offset))
        where, params = self._review_filters(status, risk)
        # rq.state is the same predicate as the count query; the filter helper
        # uses an unqualified column name, so re-qualify it for the JOIN query.
        where = where.replace("state =", "rq.state =")
        params.append(capped)
        limit_param = f"${len(params)}"
        params.append(off)
        offset_param = f"${len(params)}"
        rows = await self._pool.fetch(
            f"""
            SELECT rq.queue_id, rq.clip_id, rq.state, rq.reviewed_at,
                   rq.reason, rq.created_at,
                   c.blob_uri, c.frames_blob_uri, c.start_s, c.end_s,
                   sd.dna_json
            FROM review_queue rq
            JOIN clips c ON c.clip_id = rq.clip_id
            LEFT JOIN scenario_dna sd ON sd.clip_id = rq.clip_id
            {where}
            ORDER BY rq.created_at DESC
            LIMIT {limit_param} OFFSET {offset_param}
            """,
            *params,
        )
        return [
            {
                "queue_id": r["queue_id"],
                "clip_id": r["clip_id"],
                "state": r["state"],
                "reviewed_at": r["reviewed_at"],
                "reason": r["reason"],
                "created_at": r["created_at"],
                "blob_uri": r["blob_uri"],
                "frames_blob_uri": r["frames_blob_uri"],
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "dna_json": dict(r["dna_json"]) if r["dna_json"] else None,
            }
            for r in rows
        ]

    # ── judge overrides (P4-6) ───────────────────────────────────────────────

    async def insert_judge_override(
        self,
        *,
        clip_id: UUID,
        field: str,
        scout_value: str | None,
        judge_value: str | None,
        gt_value: str | None = None,
    ) -> int:
        """Append one override-audit row; return its BIGSERIAL id."""
        return await self._pool.fetchval(
            """
            INSERT INTO judge_overrides (clip_id, field, scout_value, judge_value, gt_value)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            clip_id,
            field,
            scout_value,
            judge_value,
            gt_value,
        )

    async def get_judge_overrides(
        self,
        clip_id: UUID | None = None,
        *,
        latest_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return override-audit rows, newest first (``latest_only`` collapses per clip+field)."""
        where = "WHERE clip_id = $1" if clip_id is not None else ""
        params: list[Any] = [clip_id] if clip_id is not None else []
        if latest_only:
            sql = f"""
                SELECT DISTINCT ON (clip_id, field)
                       id, clip_id, field, scout_value, judge_value, gt_value, created_at
                FROM judge_overrides
                {where}
                ORDER BY clip_id, field, created_at DESC
            """
        else:
            sql = f"""
                SELECT id, clip_id, field, scout_value, judge_value, gt_value, created_at
                FROM judge_overrides
                {where}
                ORDER BY created_at DESC
            """
        rows = await self._pool.fetch(sql, *params)
        return [dict(r) for r in rows]

    async def apply_judge_override_dna(
        self,
        *,
        clip_id: UUID,
        dna_json: dict[str, Any],
        judge_prompt_hash: str,
    ) -> None:
        """Targeted UPDATE of ``dna_json`` + ``judge_prompt_hash`` only; Scout provenance
        (``scout_prompt_hash`` / ``pipeline_version`` / ``dna_version``) is left untouched."""
        await self._pool.execute(
            """
            UPDATE scenario_dna
            SET dna_json = $2, judge_prompt_hash = $3
            WHERE clip_id = $1
            """,
            clip_id,
            dna_json,
            judge_prompt_hash,
        )

    async def list_dna(
        self,
        *,
        dna_version: str = CURRENT_DNA_VERSION,
        session_id: str | None = None,
        clip_ids: list[UUID] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return ``scenario_dna`` rows as ``[{clip_id, dna_json}]`` for one dna_version.

        ``clip_ids`` and ``session_id`` compose (AND). The version is a parameter rather
        than a literal so a schema bump does not require touching every caller.
        """
        conditions = ["sd.dna_version = $1"]
        params: list[Any] = [dna_version]
        idx = 2
        if clip_ids is not None:
            conditions.append(f"sd.clip_id = ANY(${idx})")
            params.append(clip_ids)
            idx += 1
        if session_id is not None:
            conditions.append(f"c.session_id = ${idx}")
            params.append(session_id)
            idx += 1
        where = " AND ".join(conditions)
        params.append(int(limit))
        rows = await self._pool.fetch(
            f"""
            SELECT sd.clip_id, sd.dna_json
            FROM scenario_dna sd
            JOIN clips c ON c.clip_id = sd.clip_id
            WHERE {where}
            ORDER BY c.created_at
            LIMIT ${idx}
            """,
            *params,
        )
        return [{"clip_id": r["clip_id"], "dna_json": dict(r["dna_json"])} for r in rows]

    async def list_reembed_source(
        self,
        *,
        dna_version: str = CURRENT_DNA_VERSION,
        session_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Return clips with everything the re-embed pass needs in one query:
        ``[{clip_id, dna_json, frames_blob_uri, source_clip_id}]``.

        ``frames_blob_uri`` is NULL when the pipeline never captured frames for
        that segment (those clips get a text-only vector).  Schema-validity
        filtering is applied by the caller, not here.
        """
        conditions = ["sd.dna_version = $1"]
        params: list[Any] = [dna_version]
        idx = 2
        if session_id is not None:
            conditions.append(f"c.session_id = ${idx}")
            params.append(session_id)
            idx += 1
        where = " AND ".join(conditions)
        params.append(int(limit))
        rows = await self._pool.fetch(
            f"""
            SELECT sd.clip_id, sd.dna_json, c.frames_blob_uri, c.source_clip_id
            FROM scenario_dna sd
            JOIN clips c ON c.clip_id = sd.clip_id
            WHERE {where}
            ORDER BY c.created_at
            LIMIT ${idx}
            """,
            *params,
        )
        return [
            {
                "clip_id": r["clip_id"],
                "dna_json": dict(r["dna_json"]),
                "frames_blob_uri": r["frames_blob_uri"],
                "source_clip_id": r["source_clip_id"],
            }
            for r in rows
        ]

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

        conditions: list[str] = ["sd.clip_id = ANY($1)"]
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
        # P3-4: JOIN clips so search results carry the metadata UI needs in
        # a single round-trip (start_s/end_s, blob_uri, source_clip_id).
        sql = (
            "SELECT sd.clip_id, sd.dna_json, "
            "c.start_s, c.end_s, c.blob_uri, c.source_clip_id "
            "FROM scenario_dna sd "
            "JOIN clips c ON c.clip_id = sd.clip_id "
            f"WHERE {where} "
            f"LIMIT ${idx}"
        )

        rows = await self._pool.fetch(sql, *params)
        return [
            {
                "clip_id": r["clip_id"],
                "dna_json": dict(r["dna_json"]),
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "blob_uri": r["blob_uri"],
                "source_clip_id": r["source_clip_id"],
            }
            for r in rows
        ]

    async def get_clip_with_blob_uri(self, clip_id: UUID) -> dict[str, Any] | None:
        """Return clip metadata joined with its scenario_dna and sessions rows.

        Returns:
            Dict with keys: clip_id, session_id, blob_uri, frames_blob_uri,
            start_s, end_s, source_clip_id, dna_version, dna_json,
            dataset, subset, dataset_version.
            dataset/subset/dataset_version are None when the session row is
            missing (should not happen in practice).
            None if the clip does not exist.
        """
        row = await self._pool.fetchrow(
            """
            SELECT c.clip_id, c.session_id, c.blob_uri, c.frames_blob_uri,
                   c.start_s, c.end_s, c.source_clip_id,
                   sd.dna_version, sd.dna_json,
                   s.dataset, s.subset, s.dataset_version,
                   rq.state AS review_status
            FROM clips c
            LEFT JOIN scenario_dna sd ON c.clip_id = sd.clip_id
            LEFT JOIN sessions s ON s.session_id = c.session_id
            LEFT JOIN LATERAL (
                SELECT state FROM review_queue
                WHERE clip_id = c.clip_id
                ORDER BY created_at DESC
                LIMIT 1
            ) rq ON true
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
            "source_clip_id": row["source_clip_id"],
            "dna_version": row["dna_version"],
            "dna_json": dict(row["dna_json"]) if row["dna_json"] else None,
            "dataset": row["dataset"],
            "subset": row["subset"],
            "dataset_version": row["dataset_version"],
            "review_status": row["review_status"],
        }

    async def find_frames_sibling(self, source_clip_id: str, at_s: float) -> dict[str, Any] | None:
        """Return a sibling segment of the same source clip that has frames and
        whose window contains ``at_s``.

        Segments of one source overlap, so a segment without frames of its own is
        still covered by a neighbour — the frame served is a real frame of the
        requested time range, not a stand-in. Prefers the window whose midpoint
        sits closest to ``at_s``.

        Returns:
            Dict with keys: frames_blob_uri, start_s, end_s. None if no sibling
            with frames covers ``at_s``.
        """
        row = await self._pool.fetchrow(
            """
            SELECT frames_blob_uri, start_s, end_s
            FROM clips
            WHERE source_clip_id = $1
              AND frames_blob_uri IS NOT NULL
              AND start_s <= $2 AND end_s >= $2
            ORDER BY abs((start_s + end_s) / 2 - $2)
            LIMIT 1
            """,
            source_clip_id,
            at_s,
        )
        if row is None:
            return None
        return {
            "frames_blob_uri": row["frames_blob_uri"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
        }

    async def list_clips(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recently inserted clips joined with their DNA row.

        Drives the Dashboard "recent clips" widget (P3-4).  Ordered by
        ``clips.created_at DESC``.  DNA columns are nullable so a clip whose
        DNA is still being processed still appears in the list.

        Args:
            limit: maximum number of rows to return (1..100).

        Returns:
            List of dicts with the same shape as ``filter_dna_by_ids`` rows.
        """
        capped = max(1, min(int(limit), 100))
        rows = await self._pool.fetch(
            """
            SELECT c.clip_id, c.session_id, c.blob_uri, c.frames_blob_uri,
                   c.start_s, c.end_s, c.source_clip_id,
                   sd.dna_version, sd.dna_json
            FROM clips c
            LEFT JOIN scenario_dna sd ON c.clip_id = sd.clip_id
            ORDER BY c.created_at DESC
            LIMIT $1
            """,
            capped,
        )
        return [
            {
                "clip_id": r["clip_id"],
                "session_id": r["session_id"],
                "blob_uri": r["blob_uri"],
                "frames_blob_uri": r["frames_blob_uri"],
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "source_clip_id": r["source_clip_id"],
                "dna_version": r["dna_version"],
                "dna_json": dict(r["dna_json"]) if r["dna_json"] else None,
            }
            for r in rows
        ]

    async def ensure_sim_road_index(self) -> None:
        """Create ``sim_road_index`` if it is absent.

        The builder provisions its own table rather than relying on ``infra/init-sql``,
        which only runs on a fresh volume: ``judge_overrides`` shipped that way in P4-6
        and is still missing from the deployed database as a result.
        """
        await self._pool.execute(SIM_ROAD_INDEX_DDL)

    async def replace_sim_road_index(self, rows: list[dict[str, Any]]) -> int:
        """Swap in a freshly parsed index. Rebuilt whole; there is nothing to merge."""
        await self.ensure_sim_road_index()
        records = [
            (
                r["town"],
                int(r["road_id"]),
                int(r["lane_id"]),
                float(r["lane_section_s"]),
                float(r["lane_section_end_s"]),
                int(r["driving_lanes"]),
                float(r["speed_kph"]),
                sorted(r["lane_types"]),
                sorted(r["junction_forms"]),
                bool(r["in_junction"]),
            )
            for r in rows
        ]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("TRUNCATE sim_road_index")
            await conn.copy_records_to_table("sim_road_index", records=records)
        return len(records)

    async def list_sim_road_index(self, *, towns: list[str] | None = None) -> list[dict[str, Any]]:
        """Return the road index, optionally restricted to *towns*."""
        sql = """
            SELECT town, road_id, lane_id, lane_section_s, lane_section_end_s,
                   driving_lanes, speed_kph, lane_types, junction_forms, in_junction
            FROM sim_road_index
        """
        params: list[Any] = []
        if towns is not None:
            sql += " WHERE town = ANY($1)"
            params.append(towns)
        sql += " ORDER BY town, road_id, lane_section_s, lane_id"
        rows = await self._pool.fetch(sql, *params)
        return [dict(r) for r in rows]

    # ── sim render ledger (P5-4) ─────────────────────────────────────────────────

    async def ensure_sim_render(self) -> None:
        """Create ``sim_render`` if it is absent, for the same reason the road index does."""
        await self._pool.execute(SIM_RENDER_DDL)

    async def list_sim_segments(
        self,
        *,
        dna_version: str = CURRENT_DNA_VERSION,
        source_clip_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Segments with everything a render needs: their DNA and their source video.

        Ordered so ``segment_index`` is position within the source clip, which is what the
        ``--segment`` override selects on.
        """
        conditions = ["sd.dna_version = $1"]
        params: list[Any] = [dna_version]
        idx = 2
        if source_clip_id is not None:
            conditions.append(f"c.source_clip_id = ${idx}")
            params.append(source_clip_id)
            idx += 1
        params.append(int(limit))
        rows = await self._pool.fetch(
            f"""
            SELECT sd.clip_id, sd.dna_json, c.source_clip_id, c.session_id,
                   c.blob_uri, c.start_s, c.end_s,
                   row_number() OVER (PARTITION BY c.source_clip_id ORDER BY c.start_s) - 1
                       AS segment_index
            FROM scenario_dna sd
            JOIN clips c ON c.clip_id = sd.clip_id
            WHERE {" AND ".join(conditions)} AND NOT c.is_synthetic
            ORDER BY c.source_clip_id, c.start_s
            LIMIT ${idx}
            """,
            *params,
        )
        return [{**dict(r), "dna_json": dict(r["dna_json"])} for r in rows]

    async def record_sim_render(self, record: dict[str, Any]) -> int:
        """Append one render attempt, successful or not. Returns its ``render_id``."""
        await self.ensure_sim_render()
        return await self._pool.fetchval(
            """
            INSERT INTO sim_render (
                clip_id, source_clip_id, segment_index, status, failure_reason,
                town, road_id, lane_id, duration_s, ego_key, chase_key, compare_key
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING render_id
            """,
            record["clip_id"],
            record["source_clip_id"],
            int(record["segment_index"]),
            record["status"],
            record.get("failure_reason"),
            record.get("town"),
            record.get("road_id"),
            record.get("lane_id"),
            record.get("duration_s"),
            record.get("ego_key"),
            record.get("chase_key"),
            record.get("compare_key"),
        )

    async def list_sim_renders(
        self, *, source_clip_id: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sim_render"
        params: list[Any] = []
        if source_clip_id is not None:
            sql += " WHERE source_clip_id = $1"
            params.append(source_clip_id)
        sql += f" ORDER BY rendered_at DESC, render_id DESC LIMIT ${len(params) + 1}"
        params.append(int(limit))
        rows = await self._pool.fetch(sql, *params)
        return [dict(r) for r in rows]

    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate curation counts for the Dashboard (P3-4).

        Returns:
            ``{"total_clips": int, "scenario_dna_count": int,
                "review": {"pending": int, "approved": int, "rejected": int,
                           "rejected_schema_invalid": int},
                "dna_pass_rate": float}``.

            ``dna_pass_rate`` is ``approved / (approved + rejected + schema_invalid)``,
            or ``None`` when no decisions have been recorded yet.
        """
        total_clips = await self._pool.fetchval("SELECT count(*) FROM clips")
        scenario_dna_count = await self._pool.fetchval("SELECT count(*) FROM scenario_dna")
        rows = await self._pool.fetch(
            "SELECT state, count(*) AS n FROM review_queue GROUP BY state"
        )
        review = {
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "rejected_schema_invalid": 0,
        }
        for r in rows:
            state = r["state"]
            if state in review:
                review[state] = int(r["n"])
        decided = review["approved"] + review["rejected"] + review["rejected_schema_invalid"]
        dna_pass_rate = float(review["approved"]) / decided if decided else None
        return {
            "total_clips": int(total_clips or 0),
            "scenario_dna_count": int(scenario_dna_count or 0),
            "review": review,
            "dna_pass_rate": dna_pass_rate,
        }
