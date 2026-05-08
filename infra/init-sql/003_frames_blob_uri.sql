-- P3-2: add frames_blob_uri column to clips.
-- Stores the MinIO key prefix (e.g. "frames/<session>/<clip_id>") under which
-- the 8 JPEG video-tower frames live.  Sister to clips.blob_uri (MP4 location).
-- Required by /v1/search/video to locate frames per clip without relying on
-- a hardcoded path convention.  NULLABLE so existing rows remain valid; new
-- rows from the DS pipeline path populate it via CurationConsumer.
-- For existing volumes: run manually as ALTER TABLE.
BEGIN;
ALTER TABLE clips
    ADD COLUMN IF NOT EXISTS frames_blob_uri TEXT;
COMMIT;
