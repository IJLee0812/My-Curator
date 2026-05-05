-- P2-6: add curation_meta column to scenario_dna.
-- Applied after 001_schema.sql on fresh docker compose up (init-sql files
-- run in filename order).
-- For existing volumes: run manually as ALTER TABLE.
BEGIN;
ALTER TABLE scenario_dna
    ADD COLUMN IF NOT EXISTS curation_meta JSONB NOT NULL DEFAULT '{}';
COMMIT;
