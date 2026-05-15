-- P3-4: add source_clip_id column to clips.
-- Stores the identifier of the original source clip a segment was carved
-- out from.  Multiple My-Curator segments can share a single source_clip_id
-- (e.g. one source clip → six 5-second segments, each with its own UUID
-- in clips).  NULLABLE so existing rows remain valid; new rows from the
-- DS pipeline populates it automatically from the source file path.
-- For existing volumes: run manually as ALTER TABLE.
BEGIN;
ALTER TABLE clips
    ADD COLUMN IF NOT EXISTS source_clip_id TEXT;
CREATE INDEX IF NOT EXISTS idx_clips_source_clip_id ON clips(source_clip_id);
COMMIT;
