-- My-Curator base schema (P1-2). Mounted into postgres at first boot via
-- /docker-entrypoint-initdb.d. Idempotent so re-applying on an existing
-- volume is safe.

BEGIN;

-- ──────────────────────────────────────────────────────────────────────
-- sessions — one row per recording session
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT        PRIMARY KEY,
    project         TEXT        NOT NULL,
    subproject      TEXT        NOT NULL,
    version         TEXT        NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL,
    source_kind     TEXT        NOT NULL CHECK (source_kind IN ('real', 'synthetic')),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ──────────────────────────────────────────────────────────────────────
-- clips — one row per segmented clip
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clips (
    clip_id         UUID        PRIMARY KEY,
    session_id      TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    blob_uri        TEXT        NOT NULL,
    start_s         DOUBLE PRECISION NOT NULL CHECK (start_s >= 0),
    end_s           DOUBLE PRECISION NOT NULL CHECK (end_s >= start_s),
    frame_count     INTEGER,
    is_gold         BOOLEAN     NOT NULL DEFAULT FALSE,
    is_synthetic    BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_clips_session ON clips(session_id);
CREATE INDEX IF NOT EXISTS idx_clips_is_gold ON clips(is_gold) WHERE is_gold;

-- ──────────────────────────────────────────────────────────────────────
-- scenario_dna — JSONB with a GIN index so nested enum filters are fast
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scenario_dna (
    clip_id         UUID        PRIMARY KEY REFERENCES clips(clip_id) ON DELETE CASCADE,
    dna_version     TEXT        NOT NULL,
    dna_json        JSONB       NOT NULL,
    scout_prompt_hash TEXT      NOT NULL,
    judge_prompt_hash TEXT,                   -- nullable until Judge lands (post-v0.1)
    pipeline_version TEXT       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_scenario_dna_json_gin
    ON scenario_dna USING GIN (dna_json jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_scenario_dna_version
    ON scenario_dna(dna_version);

-- ──────────────────────────────────────────────────────────────────────
-- review_queue — curation workflow state
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS review_queue (
    queue_id        BIGSERIAL   PRIMARY KEY,
    clip_id         UUID        NOT NULL REFERENCES clips(clip_id) ON DELETE CASCADE,
    state           TEXT        NOT NULL
                    CHECK (state IN ('pending', 'approved', 'rejected', 'rejected_schema_invalid')),
    reviewer        TEXT,
    reviewed_at     TIMESTAMPTZ,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_review_state ON review_queue(state);
CREATE INDEX IF NOT EXISTS idx_review_clip ON review_queue(clip_id);

COMMIT;
