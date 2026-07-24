-- Courses — Agent D's primary table, the supply-side mirror of job_postings.
--
-- Same shape as job_postings MINUS the scam/legitimacy machinery (courses come
-- from vetted platforms, gated by a light quality check, not a fraud filter)
-- and PLUS course-specific fields: provider, level, primary_language,
-- taught_skills, and attribution/license (freeCodeCamp curriculum is CC-BY-SA,
-- which requires attribution — we catalog skill facts + links, never
-- redistribute content).

CREATE TABLE IF NOT EXISTS courses (
    -- sha256(source || US || canonical_url_or_slug)[:32]. Scoped by source, so
    -- the same slug from two providers stays two rows.
    course_id        text PRIMARY KEY,

    -- ---- provenance -------------------------------------------------------
    source           text        NOT NULL,
    source_group     text        NOT NULL,
    -- 'api' (Coursera) or 'html_scrape' (freeCodeCamp). text + CHECK, matching
    -- the job_postings convention (ALTER TYPE can't run in a transaction).
    source_type      text        NOT NULL,
    source_url       text        NOT NULL,

    -- ---- content ----------------------------------------------------------
    name             text        NOT NULL,
    raw_description  text        NOT NULL,

    -- ---- extracted --------------------------------------------------------
    -- Skills the course TEACHES, as short canonical English names — the supply
    -- analog of job_postings.required_skills. ESCO-mapped downstream.
    taught_skills    text[]      NOT NULL DEFAULT '{}',
    provider         text,
    level            text,          -- beginner / intermediate / advanced, if stated
    primary_language char(2),       -- ISO-639-1, or NULL
    subject          text,          -- broad domain, if stated
    -- Most courses are global; NULL means "not country-specific". Kept for
    -- symmetry with job_postings and possible future regional MOOCs.
    country          char(2),

    -- ---- classification ---------------------------------------------------
    status           text        NOT NULL DEFAULT 'active',
    review_reason    text,

    -- ---- dedup ------------------------------------------------------------
    content_hash     text        NOT NULL,
    duplicate_of     text,

    -- ---- lifecycle --------------------------------------------------------
    first_seen_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at     timestamptz NOT NULL DEFAULT now(),
    missed_cycles    integer     NOT NULL DEFAULT 0,
    stale_since      timestamptz,

    -- ---- licensing (freeCodeCamp CC-BY-SA requires attribution) -----------
    attribution      text,
    license          text,

    -- ---- provenance of the extraction itself ------------------------------
    extraction_model text,
    schema_version   text        NOT NULL DEFAULT 'itqan.course/1.0',

    embedding        vector(1536),

    CONSTRAINT courses_status_check
        CHECK (status IN ('active', 'stale', 'needs_review', 'rejected')),
    CONSTRAINT courses_source_type_check
        CHECK (source_type IN ('api', 'html_scrape')),
    CONSTRAINT courses_not_self_duplicate_check
        CHECK (duplicate_of IS DISTINCT FROM course_id),

    -- ON DELETE SET NULL, same reasoning as job_postings: the 60-day prune of a
    -- canonical must not cascade-delete its duplicates (still-valid audit
    -- evidence); promoting the orphan to standalone is exactly true once its
    -- canonical is gone.
    CONSTRAINT courses_duplicate_of_fk
        FOREIGN KEY (duplicate_of) REFERENCES courses (course_id) ON DELETE SET NULL
);

-- One row per source_url. A course reached twice (re-listed) must update one
-- row, not create a second and double-count its skills in supply stats.
CREATE UNIQUE INDEX IF NOT EXISTS courses_source_url_key ON courses (source_url);
