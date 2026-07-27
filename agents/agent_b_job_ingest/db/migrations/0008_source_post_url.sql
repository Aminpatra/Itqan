-- source_post_url — the URL of the SOURCE POST a row came from.
--
-- Added when Agent B learned to SPLIT a multi-vacancy post (a "roundup" that
-- advertises several distinct roles in one article) into one row per vacancy.
-- A split vacancy's source_url becomes post_url#role — unique under
-- job_postings_source_url_uk — while this column keeps the roundup's own URL.
--
-- It is what lets change detection stay at the POST level: "has this post
-- changed since last cycle?" is decided once, before re-extracting the several
-- vacancies it splits into, so an unchanged roundup still costs zero LLM and
-- zero embedding — the property the whole warm-cycle design rests on.
--
-- For an ordinary single-vacancy posting source_post_url EQUALS source_url, so
-- existing rows are backfilled to source_url and nothing about their behaviour
-- changes.

ALTER TABLE job_postings
    ADD COLUMN source_post_url text;

UPDATE job_postings SET source_post_url = source_url WHERE source_post_url IS NULL;

ALTER TABLE job_postings
    ALTER COLUMN source_post_url SET NOT NULL,
    ALTER COLUMN source_post_url SET DEFAULT '';

COMMENT ON COLUMN job_postings.source_post_url IS
    'URL of the source post this row came from; equals source_url except for a '
    'vacancy split out of a multi-job roundup. Keyed on for post-level change detection.';

-- Change detection and per-post touch both look rows up by this column.
CREATE INDEX IF NOT EXISTS job_postings_source_post_url_idx
    ON job_postings (source, source_post_url);
