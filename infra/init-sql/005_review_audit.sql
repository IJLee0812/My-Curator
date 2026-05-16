-- P3-5: add UNIQUE(clip_id) to review_queue so set_review_status() can
-- UPSERT (overwrite) the state for a clip instead of appending rows.
-- Existing rows with duplicate clip_id are collapsed to the latest entry
-- before the constraint is applied.
--
-- Applied via cleanup_curator_db.py + volume recreate (same pattern as 004).

BEGIN;

DELETE FROM review_queue rq
USING review_queue rq2
WHERE rq.clip_id = rq2.clip_id
  AND rq.queue_id < rq2.queue_id;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'review_queue_clip_id_unique'
    ) THEN
        ALTER TABLE review_queue
            ADD CONSTRAINT review_queue_clip_id_unique UNIQUE (clip_id);
    END IF;
END
$$;

COMMIT;
