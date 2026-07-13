-- My-Curator base schema (consolidated from P1-2 + P2-6 + P3-1 + P3-4 + P3-5).
-- Mounted into Postgres at first boot via /docker-entrypoint-initdb.d.
-- Idempotent so re-applying on an existing volume is safe.
--
-- Migration history (for existing-volume manual upgrades, see
-- docs/db_migration_history.md):
--   P1-2  initial 4 tables + indexes
--   P2-6  scenario_dna.curation_meta JSONB
--   P3-1  clips.frames_blob_uri TEXT
--   P3-4  clips.source_clip_id TEXT + idx_clips_source_clip_id
--   P3-5  review_queue UNIQUE(clip_id) — review_status UPSERT
--   P4-6  judge_overrides (append-only Judge-vs-Scout override audit log)

BEGIN;

-- ──────────────────────────────────────────────────────────────────────
-- sessions — one row per recording session
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT        PRIMARY KEY,
    dataset         TEXT        NOT NULL,
    subset          TEXT        NOT NULL,
    dataset_version TEXT        NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL,
    source_kind     TEXT        NOT NULL CHECK (source_kind IN ('real', 'synthetic')),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ──────────────────────────────────────────────────────────────────────
-- clips — one row per segmented clip
-- ──────────────────────────────────────────────────────────────────────
-- Column order matches the historical ALTER TABLE sequence (P1-2 base, then
-- P3-1 frames_blob_uri appended, then P3-4 source_clip_id appended) so the
-- consolidated end-state matches ``pg_dump --schema-only`` byte-for-byte
-- against a database that was migrated through the original 002→004 files.
CREATE TABLE IF NOT EXISTS clips (
    clip_id         UUID        PRIMARY KEY,
    session_id      TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    blob_uri        TEXT        NOT NULL,
    start_s         DOUBLE PRECISION NOT NULL CHECK (start_s >= 0),
    end_s           DOUBLE PRECISION NOT NULL CHECK (end_s >= start_s),
    frame_count     INTEGER,
    is_gold         BOOLEAN     NOT NULL DEFAULT FALSE,
    is_synthetic    BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    frames_blob_uri TEXT,                                 -- P3-1 (ALTER-appended)
    source_clip_id  TEXT                                  -- P3-4 (ALTER-appended)
);
CREATE INDEX IF NOT EXISTS idx_clips_session         ON clips(session_id);
CREATE INDEX IF NOT EXISTS idx_clips_is_gold         ON clips(is_gold) WHERE is_gold;
CREATE INDEX IF NOT EXISTS idx_clips_source_clip_id  ON clips(source_clip_id);  -- P3-4

-- ──────────────────────────────────────────────────────────────────────
-- scenario_dna — JSONB with a GIN index so nested enum filters are fast
-- ──────────────────────────────────────────────────────────────────────
-- Column order matches the historical ALTER TABLE sequence (P1-2 base, then
-- P2-6 curation_meta appended) so the consolidated end-state matches
-- ``pg_dump --schema-only`` byte-for-byte.
CREATE TABLE IF NOT EXISTS scenario_dna (
    clip_id           UUID        PRIMARY KEY REFERENCES clips(clip_id) ON DELETE CASCADE,
    dna_version       TEXT        NOT NULL,
    dna_json          JSONB       NOT NULL,
    scout_prompt_hash TEXT        NOT NULL,
    judge_prompt_hash TEXT,                  -- nullable until Judge lands (post-v0.1)
    pipeline_version  TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    curation_meta     JSONB       NOT NULL DEFAULT '{}'   -- P2-6 (ALTER-appended)
);
CREATE INDEX IF NOT EXISTS idx_scenario_dna_json_gin
    ON scenario_dna USING GIN (dna_json jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_scenario_dna_version
    ON scenario_dna(dna_version);

-- ──────────────────────────────────────────────────────────────────────
-- review_queue — curation workflow state
-- UNIQUE(clip_id) lets set_review_status() UPSERT instead of appending.
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS review_queue (
    queue_id        BIGSERIAL   PRIMARY KEY,
    clip_id         UUID        NOT NULL REFERENCES clips(clip_id) ON DELETE CASCADE,
    state           TEXT        NOT NULL
                    CHECK (state IN ('pending', 'approved', 'rejected', 'rejected_schema_invalid')),
    reviewer        TEXT,
    reviewed_at     TIMESTAMPTZ,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT review_queue_clip_id_unique UNIQUE (clip_id)  -- P3-5
);
CREATE INDEX IF NOT EXISTS idx_review_state ON review_queue(state);
CREATE INDEX IF NOT EXISTS idx_review_clip  ON review_queue(clip_id);

-- ──────────────────────────────────────────────────────────────────────
-- judge_overrides — append-only audit of every Judge-vs-Scout override (P4-6)
-- One row per (clip, field) each time the Judge changes a Scout value; history
-- accumulates across re-runs, and callers take the latest per (clip_id, field).
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS judge_overrides (
    id          BIGSERIAL   PRIMARY KEY,
    clip_id     UUID        NOT NULL REFERENCES clips(clip_id) ON DELETE CASCADE,
    field       TEXT        NOT NULL
                CHECK (field IN ('risk_level', 'scene_description', 'safety_event_consistency')),
    scout_value TEXT,
    judge_value TEXT,
    gt_value    TEXT,                                 -- nullable (gold may be absent)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_judge_overrides_clip ON judge_overrides(clip_id);

COMMIT;
