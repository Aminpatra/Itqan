-- Job postings — one of the two tables Agent C consumes.
--
-- Enums are `text` + CHECK rather than a real Postgres ENUM. ALTER TYPE ... ADD
-- VALUE cannot run inside a transaction block on older servers, and a value can
-- never be removed at all; a CHECK is dropped and recreated in one migration.
-- Reversibility beats type-system purity for a column whose value set is still
-- settling.

CREATE TABLE IF NOT EXISTS job_postings (
    -- sha256(source || US || canonical_url)[:32]. Canonicalization matters:
    -- the same posting reached from a listing page and from search differs only
    -- in tracking params, and ingesting it twice would inflate every
    -- frequency_count in skill_demand_stats.
    posting_id       text PRIMARY KEY,

    -- ---- provenance -------------------------------------------------------
    source           text        NOT NULL,
    -- Sources that are the same underlying operation published in several
    -- places (a blog and its companion Telegram channel) share a group, so
    -- aggregation can dedupe by operation rather than by URL.
    source_group     text        NOT NULL,
    source_type      text        NOT NULL,
    source_url       text        NOT NULL,

    -- ---- content ----------------------------------------------------------
    -- Required by skill_demand_stats.sample_postings, which is specified as
    -- "title + source_url"; without it that JSONB cannot be built without
    -- re-parsing raw_description.
    title            text        NOT NULL,
    raw_description  text        NOT NULL,

    -- ---- extracted --------------------------------------------------------
    -- ISCO-08 major group. NULL when extraction could not determine it — never
    -- guessed into a default, and NULL sectors are excluded from aggregation.
    sector           text,
    -- Free text today. ESCO codes are deferred to a future skill_esco_map side
    -- table rather than a parallel array here: parallel arrays must stay
    -- index-aligned forever and Postgres cannot enforce that.
    required_skills  text[]      NOT NULL DEFAULT '{}',
    seniority_level  text,
    location         text,
    -- ISO-3166 alpha-2. Deliberately NOT a legitimacy signal: a regional board
    -- carries legitimate postings for neighbouring countries, and scoring those
    -- as fraud would delete real jobs invisibly. Scope is filtered at
    -- aggregation instead.
    country          char(2),
    posted_date      date,

    -- ---- classification ---------------------------------------------------
    status           text        NOT NULL DEFAULT 'active',
    -- 1.000 = clean, 0.000 = certainly a scam. Stored as legitimacy (not risk)
    -- so the column name matches its direction; a misleading name outlives
    -- every comment explaining it.
    legitimacy_score numeric(4,3),
    -- Why a row is in needs_review. Without it the queue mixes "extraction
    -- found no sector" with "possible cross-group duplicate", which need
    -- completely different human follow-up.
    review_reason    text,

    -- ---- dedup ------------------------------------------------------------
    content_hash     text        NOT NULL,
    -- Points at the canonical posting when this one is a duplicate. Set by link
    -- extraction (deterministic) or embedding similarity (fallback). Duplicates
    -- are retained, never deleted — they are evidence that a posting was
    -- republished.
    duplicate_of     text,

    -- ---- lifecycle --------------------------------------------------------
    first_seen_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at     timestamptz NOT NULL DEFAULT now(),
    -- An explicit counter, not derived from last_seen_at. The staleness rule
    -- counts CYCLES; deriving from elapsed time breaks the moment a cycle is
    -- skipped (machine asleep, source blocked for a day) and would age every
    -- posting wrongly.
    missed_cycles    integer     NOT NULL DEFAULT 0,
    -- When staleness began. last_seen_at is not a substitute: it drifts the
    -- moment a stale posting is seen again and revives.
    stale_since      timestamptz,

    -- ---- provenance of the extraction itself ------------------------------
    extraction_model text,
    schema_version   text        NOT NULL DEFAULT 'itqan.job_posting/1.0',

    embedding        vector(1536),

    CONSTRAINT job_postings_status_check
        CHECK (status IN ('active', 'stale', 'needs_review', 'rejected')),
    CONSTRAINT job_postings_source_type_check
        CHECK (source_type IN ('html_scrape', 'blogger_feed', 'telegram')),
    CONSTRAINT job_postings_legitimacy_range_check
        CHECK (legitimacy_score IS NULL
               OR (legitimacy_score >= 0 AND legitimacy_score <= 1)),
    CONSTRAINT job_postings_not_self_duplicate_check
        CHECK (duplicate_of IS DISTINCT FROM posting_id),

    -- ON DELETE SET NULL, deliberately:
    --   CASCADE  -> the 60-day prune of a canonical would silently delete its
    --               duplicates, which may still be active and are independent
    --               audit evidence. Data loss on a cron job.
    --   RESTRICT -> the prune raises an FK violation and an unattended cycle
    --               dies.
    --   SET NULL -> the orphan is promoted back to a standalone posting, which
    --               is exactly true once its canonical no longer exists.
    CONSTRAINT job_postings_duplicate_of_fk
        FOREIGN KEY (duplicate_of) REFERENCES job_postings (posting_id)
        ON DELETE SET NULL
);
